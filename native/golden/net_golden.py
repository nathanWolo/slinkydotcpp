"""Golden vectors for the Rust network crate (native/crates/net).

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \
        /home/nathan/slinky-native/native/golden/net_golden.py [--real-per-mode N] [--random N] [--out DIR]

Writes, for each network, a run directory under ``/dev/shm/naz-golden/net/<name>/`` (RAM, never
committed) holding

* ``config.json`` and ``params.npz``: a loadable run dir (``M`` links to the deployed
  /home/nathan/slinky-serve/az-modes-M-s0; ``S``, ``L``, ``S_nomode``, ``S_none`` are fresh
  ``az_modes.init_network`` weights with the zero ``feat`` layer and the near-zero heads replaced
  by random normals, so every part of the net matters; ``S_nomode``/``S_none`` use features=no_mode/none);
* ``cases.npz``: float32 arrays ``obs [n,21,21,13]`` (NHWC, the network's boards), ``feats [n,7]``
  (raw ``modes.mode_features``), ``logits [n,4]`` and ``value [n]`` (the jitted JAX
  ``az_modes.network``), and ``kind [n]`` float32 (0..3 = real states of standard, duel, royale,
  constrictor in ``modes.MODE_NAMES`` order, 4 = random observations). For ``no_mode`` the expected
  outputs use features multiplied by unified._NO_MODE_KEEP (the Rust net applies that mask itself).

Real cases are every living snake's egocentric observation (food hidden in constrictor) of random
play states in all four modes. Random cases mix sparse and dense planes, hazard counts above 1 and
random features.
"""
import argparse, dataclasses, json, os, shutil, sys
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, "/home/nathan/slinky-release/baselines")
import az_modes
from slinky import modes, unified
from slinky.env import BattlesnakeEnv

M_RUN = "/home/nathan/slinky-serve/az-modes-M-s0"


def real_cases(per_mode, cfg_all):
    obs, feats, kind = [], [], []
    for k, mode in enumerate(modes.MODE_NAMES):
        game = modes.mode_config(mode)
        env = BattlesnakeEnv(game, obs="egocentric")
        step = jax.jit(env.step)
        reset = jax.jit(env.reset)
        inputs = jax.jit(lambda s: unified.policy_inputs(env.observe(s), s, game, cfg_all, mode))
        got, seed = 0, 0
        while got < per_mode:
            key = jax.random.key(1000 * k + seed); seed += 1
            state, ts = reset(key)
            for t in range(120):
                if t % 4 == 1 or t < 3:
                    b, f = inputs(state)
                    for a in range(game.num_snakes):
                        if bool(state.alive[a]) and got < per_mode:
                            obs.append(np.asarray(b[a])); feats.append(np.asarray(f[a])); kind.append(k); got += 1
                key, k1, k2 = jax.random.split(key, 3)
                acts = jax.random.categorical(k1, jnp.where(ts.action_mask, 0., -jnp.inf))
                nstate, ts2 = step(k2, state, acts)
                if bool(nstate.done): break
                state, ts = nstate, ts2
    return np.stack(obs), np.stack(feats), np.array(kind, np.int32)


def random_cases(n, rng):
    obs = np.zeros((n, 21, 21, 13), np.float32)
    for i in range(n):
        dens = rng.choice([0.02, 0.1, 0.5, 1.0])
        x = (rng.random((21, 21, 13)) < dens).astype(np.float32)
        cont = rng.random((21, 21, 13)).astype(np.float32) * (rng.random() < 0.5)
        x = np.where(rng.random((21, 21, 13)) < 0.5, x, np.maximum(x, cont) if dens > .3 else x * cont)
        x[..., 0] = (rng.random((21, 21)) < 0.7)
        x[..., 2] = rng.integers(0, 4, (21, 21))
        obs[i] = x
    feats = rng.random((n, 7)).astype(np.float32)
    feats[:, :4] = np.eye(4, dtype=np.float32)[rng.integers(0, 4, n)]
    return obs, feats, np.full(n, 4, np.int32)


def make_net(name, cfg, key, noise):
    params = az_modes.init_network(key, cfg)
    rng = np.random.default_rng(sum(map(ord, name)))
    out = {}
    for layer, d in params.items():
        out[layer] = {}
        for k, v in d.items():
            v = np.asarray(v, np.float32)
            if layer in ("policy", "value", "feat") and k == "w":
                v = rng.normal(0, 0.3 if layer != "feat" else 1.0, v.shape).astype(np.float32)
            elif k == "b":
                v = rng.normal(0, 0.1, v.shape).astype(np.float32)
            out[layer][k] = v
    return {k: {kk: jnp.asarray(vv) for kk, vv in d.items()} for k, d in out.items()}


def write_run(d, cfg, params):
    os.makedirs(d, exist_ok=True)
    unified.save_config(d, "az_modes", cfg)
    unified.save_params(d, params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-per-mode", type=int, default=160)
    ap.add_argument("--random", type=int, default=300)
    ap.add_argument("--out", default="/dev/shm/naz-golden/net")
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    mcfg = az_modes.load_config(M_RUN)
    ro, rf, rk = real_cases(a.real_per_mode, mcfg)
    xo, xf, xk = random_cases(a.random, np.random.default_rng(5))
    obs = np.concatenate([ro, xo]).astype(np.float32)
    feats = np.concatenate([rf, xf]).astype(np.float32)
    kind = np.concatenate([rk, xk])
    print("cases", len(obs), "real", len(ro))

    nets = {"M": (mcfg, az_modes.load_params(M_RUN, mcfg))}
    for name, size, feat in [("S", "S", "all"), ("L", "L", "all"), ("S_nomode", "S", "no_mode"), ("S_none", "S", "none")]:
        ch, hid = unified.SIZES[size]
        cfg = dataclasses.replace(mcfg, conv_channels=ch, hidden=hid, features=feat)
        nets[name] = (cfg, make_net(name, cfg, jax.random.key(11), 0))
    for name, (cfg, params) in nets.items():
        f = feats * unified._NO_MODE_KEEP if cfg.features == "no_mode" else feats
        net = jax.jit(lambda b, f: az_modes.network(params, b, f, cfg))
        lg, vs = [], []
        for i in range(0, len(obs), 50):
            l, v = net(jnp.asarray(obs[i:i + 50]), jnp.asarray(f[i:i + 50]))
            lg.append(np.asarray(l)); vs.append(np.asarray(v))
        d = os.path.join(a.out, name)
        if name == "M":
            os.makedirs(d); [os.symlink(os.path.join(M_RUN, n), os.path.join(d, n)) for n in ("config.json", "params.npz")]
        else:
            write_run(d, cfg, params)
        np.savez(os.path.join(d, "cases.npz"), obs=obs, feats=feats, logits=np.concatenate(lg), value=np.concatenate(vs), kind=kind.astype(np.float32))
        print(name, "ok", sum(int(np.prod(np.shape(x))) for p in params.values() for x in p.values()), "params")


if __name__ == "__main__":
    main()
