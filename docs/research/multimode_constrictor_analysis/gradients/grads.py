"""Per-mode PPO gradients of the generalist (or another ppo_modes net) on fresh on-policy rollouts.

For one parameter set, collects R independent rollouts (the trainer's make_rollout over all four
modes, 64 envs x 128 turns, after `warm` discarded rollouts of 128 turns from fresh resets), computes
GAE / returns / value weights exactly as make_train_chunk does, then for each mode and each block of
`block_envs` envs computes the gradient of the three terms of pm.group_loss (pg_loss, v_loss,
entropy) at the current params (PPO ratio = 1). Advantages are normalised over the whole group
batch of the rollout (the trainer normalises per random minibatch; same expectation), and then
group_loss runs with norm_adv=False, so the count-weighted mean of the block gradients is exactly the
full-batch gradient of the trainer's group loss.

Saves (out npz): the Gram matrix of all block gradients per layer group (for bootstrap over env
blocks), block valid counts, value weights, full-rollout gradient vectors per (rollout, mode, part),
and rollout diagnostics.

Usage: python grads.py <run_dir> <params file> <out.npz> [--rollouts 2] [--warm 2] [--seed 0]
       [--cfg-run runs/ppo-modes-seed0]
The policy collecting the rollouts and the parameters differentiated are the same (on-policy);
the loss config (modes, value_norm, etc.) is --cfg-run's (G's by default).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

sys.path.insert(0, "baselines")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import ppo_modes as pm  # noqa: E402

PARTS = ("pg", "v", "ent")


def layer_slices(params):
    """Flat-vector slices of the layer groups, in jax.tree leaves order."""
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    out, start = {}, 0
    names = []
    for path, leaf in flat:
        name = "/".join(k.key for k in path)
        n = int(np.prod(leaf.shape))
        names.append((name, start, n, leaf.shape))
        start += n
    groups = {"conv": [], "hidden": [], "feat_onehot": [], "feat_shared": [], "policy": [], "value": []}
    for name, s, n, shape in names:
        layer = name.split("/")[0]
        if layer.startswith("conv"):
            groups["conv"].append(np.arange(s, s + n))
        elif layer == "hidden":
            groups["hidden"].append(np.arange(s, s + n))
        elif layer == "policy":
            groups["policy"].append(np.arange(s, s + n))
        elif layer == "value":
            groups["value"].append(np.arange(s, s + n))
        elif layer == "feat":
            idx = np.arange(s, s + n).reshape(shape)  # [7, hidden]
            groups["feat_onehot"].append(idx[:4].ravel())
            groups["feat_shared"].append(idx[4:].ravel())
        else:
            raise ValueError(name)
    return {k: np.concatenate(v) for k, v in groups.items() if v}, start, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("params_file")
    ap.add_argument("out")
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-envs", type=int, default=4)
    ap.add_argument("--cfg-run", default="runs/ppo-modes-seed0")
    ap.add_argument("--check", action="store_true", help="verify against full-batch jax.grad (duel)")
    args = ap.parse_args()
    skip = os.path.join(os.path.dirname(args.out), "SKIP_" + os.path.basename(args.out))
    if os.path.exists(skip):
        print(f"skipped ({skip} exists)", flush=True)
        return

    t0 = time.perf_counter()
    cfg = pm.load_config(args.cfg_run)
    pcfg = pm.load_config(args.run_dir)
    params = pm.load_params(args.run_dir, pcfg, name=args.params_file)
    assert pcfg.features == cfg.features and pcfg.hidden == cfg.hidden
    groups = pm.make_groups(cfg)
    E, T = cfg.num_envs, cfg.num_steps
    BE = args.block_envs
    nblocks = E // BE
    cfg_nn = dataclasses.replace(cfg, norm_adv=False)
    slices, P, names = layer_slices(params)

    rollout = jax.jit(pm.make_rollout(cfg, groups))
    values_of = jax.jit(pm.make_values(cfg, groups))

    @jax.jit
    def gae_all(trajs, lasts):
        outs = []
        for traj, last in zip(trajs, lasts, strict=True):
            adv, ret = pm.compute_gae(
                traj.rewards, traj.values, traj.valid, traj.terminal, traj.done,
                traj.boot_values, last, cfg.gamma, cfg.gae_lambda,
            )  # fmt: skip
            w = traj.valid.astype(jnp.float32)
            cnt = jnp.maximum(w.sum(), 1.0)
            mu = jnp.sum(w * adv) / cnt
            sd = jnp.sqrt(jnp.sum(w * (adv - mu) ** 2) / cnt)
            adv_n = (adv - mu) / (sd + 1e-8)
            rv = pm._weighted_var(ret, traj.valid)
            ev = pm.explained_variance(traj.values, ret, traj.valid)
            outs.append((adv_n, ret, rv, ev))
        return outs

    @jax.jit
    def block_grads(params, mb, value_weight):
        def f(p):
            logits, v = pm.network(p, mb.obs, mb.feats, cfg)
            loss, parts = pm.group_loss(logits, v, mb, value_weight, cfg_nn)
            return jnp.stack([parts["pg_loss"], parts["v_loss"], parts["entropy"]]), parts

        # Three backward passes from one forward pass (2.4x faster than jacrev on XLA:CPU).
        _, vjp, parts = jax.vjp(f, params, has_aux=True)
        rows = []
        for i in range(3):
            (gi,) = vjp(jnp.eye(3, dtype=jnp.float32)[i])
            rows.append(jnp.concatenate([x.ravel() for x in jax.tree.leaves(gi)]))
        flat = jnp.stack(rows)  # [3, P]
        count = mb.valid.sum()
        return flat, count, parts

    G = len(groups)
    R = args.rollouts
    X = np.zeros((R, G, nblocks, 3, P), np.float32)
    counts = np.zeros((R, G, nblocks), np.float64)
    value_weights = np.zeros((R, G))
    return_vars = np.zeros((R, G))
    expl_var = np.zeros((R, G))
    diag = {k: np.zeros((R, G, nblocks)) for k in ("approx_kl", "clip_frac", "entropy", "pg_loss", "v_loss")}
    roll_metrics = []
    base_key = jax.random.key(args.seed)
    for r in range(R):
        kr = jax.random.fold_in(base_key, r)
        k_init, k_warm, k_meas = jax.random.split(kr, 3)
        states = tuple(
            jax.vmap(g.sim.init_state)(jax.random.split(jax.random.fold_in(k_init, g.mode_id), E))
            for g in groups
        )
        t = time.perf_counter()
        for w in range(args.warm):
            states, trajs, _ = rollout(params, states, jax.random.fold_in(k_warm, w))
            del trajs
        jax.block_until_ready(states)
        t_warm = time.perf_counter() - t
        t = time.perf_counter()
        states, trajs, metrics = rollout(params, states, k_meas)
        lasts = values_of(params, states)
        outs = gae_all(trajs, lasts)
        jax.block_until_ready(outs)
        t_roll = time.perf_counter() - t
        rv = jnp.stack([o[2] for o in outs])
        vw = pm.value_loss_weights(rv, cfg)
        return_vars[r] = np.asarray(rv)
        value_weights[r] = np.asarray(vw)
        expl_var[r] = np.asarray([o[3] for o in outs])
        m = jax.tree.map(np.asarray, metrics)
        roll_metrics.append({k: v.tolist() for k, v in m.items()})
        t = time.perf_counter()
        for gi, (g, traj, o) in enumerate(zip(groups, trajs, outs, strict=True)):
            adv_n, ret = o[0], o[1]
            N = g.num_snakes
            for b in range(nblocks):
                sl = slice(b * BE, (b + 1) * BE)

                def fl(x):
                    x = x[:, sl]
                    return x.reshape(T * BE * N, *x.shape[3:])

                mb = pm.Batch(
                    fl(traj.obs), fl(traj.feats), fl(traj.mask), fl(traj.actions),
                    fl(traj.log_probs), fl(traj.values), fl(adv_n), fl(ret), fl(traj.valid),
                )  # fmt: skip
                flat, count, parts = block_grads(params, mb, vw[gi])
                X[r, gi, b] = np.asarray(flat)
                counts[r, gi, b] = float(count)
                for k in diag:
                    key = {"pg_loss": "pg_loss", "v_loss": "v_loss"}.get(k, k)
                    diag[k][r, gi, b] = float(parts[key])
        t_grad = time.perf_counter() - t
        if args.check and r == 0:
            # The duel group's full-batch trainer loss (norm_adv=True on the raw advantages, over
            # the whole group batch) vs the count-weighted mean of the block gradients.
            gi = [g.name for g in groups].index("duel")
            g, traj = groups[gi], trajs[gi]
            adv_raw, _ = pm.compute_gae(
                traj.rewards, traj.values, traj.valid, traj.terminal, traj.done,
                traj.boot_values, lasts[gi], cfg.gamma, cfg.gae_lambda,
            )  # fmt: skip
            S = T * E * g.num_snakes
            fl = lambda x: x.reshape(S, *x.shape[3:])  # noqa: E731
            mb = pm.Batch(
                fl(traj.obs), fl(traj.feats), fl(traj.mask), fl(traj.actions), fl(traj.log_probs),
                fl(traj.values), fl(adv_raw), fl(outs[gi][1]), fl(traj.valid),
            )  # fmt: skip

            def full_loss(p):
                logits, v = pm.network(p, mb.obs, mb.feats, cfg)
                return pm.group_loss(logits, v, mb, vw[gi], cfg)[0]

            gref = jax.jit(jax.grad(full_loss))(params)
            gref = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(gref)])
            wb = counts[r, gi] / counts[r, gi].sum()
            parts_full = np.einsum("b,bkp->kp", wb, X[r, gi].astype(np.float64))
            mine = parts_full[0] + cfg.vf_coef * float(vw[gi]) * parts_full[1] - cfg.ent_coef * parts_full[2]
            rel = np.linalg.norm(mine - gref) / np.linalg.norm(gref)
            cosv = mine @ gref / (np.linalg.norm(mine) * np.linalg.norm(gref))
            print(f"CHECK duel full-batch grad: rel err {rel:.2e}, cos {cosv:.6f}, |g| {np.linalg.norm(gref):.4e}", flush=True)
        del trajs, outs
        print(
            f"rollout {r}: warm {t_warm:.0f}s, rollout+gae {t_roll:.0f}s, grads {t_grad:.0f}s; "
            f"value_weights {np.round(value_weights[r], 3)}, return_var {np.round(return_vars[r], 3)}, "
            f"games {m['games']}, game_turns/games {np.round(m['game_turns'] / np.maximum(m['games'], 1), 1)}",
            flush=True,
        )

    # The block gradients themselves (float32), so files of the same params can be pooled.
    np.save(args.out.removesuffix(".npz") + "-blocks.npy", X)
    # Gram matrices per layer group over all (rollout, mode, block, part) vectors.
    flatX = X.reshape(-1, P)
    grams = {}
    for name, idx in slices.items():
        acc = np.zeros((flatX.shape[0], flatX.shape[0]))
        for c in range(0, len(idx), 16384):  # float64 accumulation, bounded memory
            sub = flatX[:, idx[c : c + 16384]].astype(np.float64)
            acc += sub @ sub.T
        grams[name] = acc
    # Full-rollout gradients (count-weighted means of block gradients): [R, G, 3, P]
    w = counts / counts.sum(axis=2, keepdims=True)
    full = np.zeros((R, G, 3, P), np.float32)
    for r in range(R):
        for g in range(G):
            full[r, g] = np.einsum("b,bkp->kp", w[r, g], X[r, g].astype(np.float64))
    np.savez_compressed(
        args.out,
        **{f"gram_{k}": v for k, v in grams.items()},
        counts=counts,
        value_weights=value_weights,
        return_vars=return_vars,
        explained_var=expl_var,
        full=full,
        modes=np.array([g.name for g in groups]),
        parts=np.array(PARTS),
        vf_coef=cfg.vf_coef,
        ent_coef=cfg.ent_coef,
        **{f"diag_{k}": v for k, v in diag.items()},
        roll_metrics=np.array(repr(roll_metrics)),
        layer_names=np.array([n for n, *_ in names]),
    )
    print(f"saved {args.out} in {time.perf_counter() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
