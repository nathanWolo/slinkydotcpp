"""Play the generalist (G) against its mode specialist (S), seats G,S,G,S rotated, and record
every decision state, every death and who caused it.

Usage (from /home/user/slinkydotcpp):
    taskset -c 1 .venv/bin/python <this file> --mode constrictor --games 512 --out <npz>

Games are the *same* games as benchmarks/modes.py plays for the cell "G,S,G,S" of that mode
(same key derivation, same seat rotation, same per-turn env key), so the first N games here are
the first N games of the benchmark cell and the outcome can be checked against play_table.

Each game is played to its natural end (no early stop), so the contested part (the benchmark's
early-stop) can be recovered afterwards. Recorded per game and per turn t (state s_t before the
move; arrays are [games, T, ...]):
  alive, head, length, mask, done (before), applied actions, G's and S's unmasked logits for all
  seats (the counterfactual move of the other policy on the same state), the cell owner grid
  (constrictor only), the new head cell, who died this turn and the 'killers' matrix.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import sys
import time

sys.path.insert(0, "baselines")

import jax
import jax.numpy as jnp
import numpy as np

import ppo_modes as pm  # noqa: E402
from slinky import modes, rules  # noqa: E402
from slinky.env import BattlesnakeEnv  # noqa: E402
from slinky.types import Cause  # noqa: E402


def cell_key(seed: int, game_mode: str) -> jax.Array:
    # Copy of benchmarks/modes.py:cell_key (kept identical so the games coincide).
    digest = hashlib.sha256(f"modes\0{game_mode}".encode()).digest()
    return jax.random.fold_in(jax.random.key(seed), int.from_bytes(digest[:4], "big") >> 1)


def build(mode: str, tmax: int, detail: bool, run_g: str, run_s: str, snap_g=None, snap_s=None, hyb_first: str = 'G', sample: bool = False, eps: float = 0.0):
    game = modes.mode_config(mode)
    env = BattlesnakeEnv(game, obs="egocentric")
    cg, cs = pm.load_config(run_g), pm.load_config(run_s)
    pg = pm.load_params(run_g, cg, name=snap_g or "params.npz")
    ps = pm.load_params(run_s, cs, name=snap_s or "params.npz")
    logit_g = pm.make_logits_fn(pg, cg, game)
    logit_s = pm.make_logits_fn(ps, cs, game)
    n, h, w = game.num_snakes, game.height, game.width
    seats = jnp.arange(n)
    lineup = jnp.array([0, 1, 0, 1], jnp.int32)
    idx = jnp.arange(n)[:, None, None]
    constrictor = game.ruleset.constrictor

    def killers_of(state, actions, new):
        """Per snake: who caused its death this turn, as a bool matrix [i, j] (j killed i)."""
        alive0 = state.alive
        _, head = rules.next_heads(state, actions, game)
        body_dec = jnp.where(alive0[:, None, None], jnp.maximum(state.body - 1, 0), state.body)
        hits = rules.value_at(body_dec > 0, head, game).T  # [i, j]: head i on body of j
        died = alive0 & ~new.alive
        cause = new.elim_cause.astype(jnp.int32)
        pass1 = died & ((cause == Cause.OUT_OF_HEALTH) | (cause == Cause.OUT_OF_BOUNDS) | (cause == Cause.HAZARD))
        alive2 = alive0 & ~pass1
        others = ~jnp.eye(n, dtype=bool) & alive2[None, :]
        same_cell = jnp.all(head[:, None, :] == head[None, :, :], axis=-1) & others
        if constrictor:
            longer_or_equal = jnp.ones((n, n), bool)  # lengths stay equal; checked from the records
        else:
            longer_or_equal = new.length[:, None] <= new.length[None, :]
        k_body = hits & others
        k_h2h = same_cell & longer_or_equal
        k_self = jnp.eye(n, dtype=bool) & hits  # self collision
        killers = jnp.where((cause == Cause.COLLISION)[:, None], k_body, False)
        killers = killers | jnp.where((cause == Cause.HEAD_COLLISION)[:, None], k_h2h, False)
        killers = killers | jnp.where((cause == Cause.SELF_COLLISION)[:, None], k_self, False)
        killers = killers & died[:, None]
        return killers, head

    def one(state, ts, k_loop, g, K):
        lg = logit_g(state, ts.obs)
        ls = logit_s(state, ts.obs)
        mask = ts.action_mask
        ag = jnp.argmax(jnp.where(mask, lg, -jnp.inf), axis=-1).astype(jnp.int32)
        a_s = jnp.argmax(jnp.where(mask, ls, -jnp.inf), axis=-1).astype(jnp.int32)
        ks = jax.random.split(jax.random.fold_in(k_loop, state.turn), 3)
        if sample:
            # as play_table would with sampled policies: policy 0 uses key 0, policy 1 key 1, the env key 2
            pg_ = pm.sample_actions(ks[0], lg, mask)
            ps_ = pm.sample_actions(ks[1], ls, mask)
        elif eps > 0:
            # epsilon-greedy: with probability eps a uniformly random legal move instead of the greedy one
            def _eg(key, a):
                k1, k2 = jax.random.split(key)
                rnd = jax.random.categorical(k1, jnp.where(mask, 0.0, -1e9)).astype(jnp.int32)
                return jnp.where(jax.random.uniform(k2, a.shape) < eps, rnd, a)
            pg_, ps_ = _eg(ks[0], ag), _eg(ks[1], a_s)
        else:
            pg_, ps_ = ag, a_s
        owner = lineup[(seats - g) % n]  # 0 = G, 1 = S
        # Team 0 (the 'G' seats) plays hyb_first's move for turns < K and the other policy's afterwards;
        # with the defaults (hyb_first='G', K huge) it is simply G. Team 1 always plays S.
        first, second = (pg_, ps_) if hyb_first == "G" else (ps_, pg_)
        team0 = jnp.where(state.turn < K, first, second)
        actions = jnp.where(owner == 0, team0, ps_).astype(jnp.int32)
        new, nts = env.step(ks[2], state, actions)
        killers, head = killers_of(state, actions, new)
        rec = dict(
            alive=state.alive, head=state.head, length=state.length, done=state.done,
            mask=mask, act=actions, ag=ag, a_s=a_s, killers=killers, newhead=head,
            alive_after=new.alive, health=state.health, last_move=state.last_move,
        )  # fmt: skip
        if detail:
            rec["lg"], rec["ls"] = lg, ls
            own = jnp.max(jnp.where((state.body > 0) & state.alive[:, None, None], idx + 1, 0), axis=0)
            rec["owner_grid"] = (own - 1).astype(jnp.int8)
            rec["hazard"] = state.hazard.astype(jnp.int8)
        return new, nts, rec

    def run(key, game_ids, K):
        def init(g):
            k_reset, k_loop = jax.random.split(jax.random.fold_in(key, g))
            state, ts = env.reset(k_reset)
            return state, ts, k_loop

        states, tss, k_loops = jax.vmap(init)(game_ids)
        _, _, rec0 = jax.vmap(one, in_axes=(0, 0, 0, 0, None))(states, tss, k_loops, game_ids, K)
        bufs = jax.tree.map(lambda x: jnp.zeros((x.shape[0], tmax) + x.shape[1:], x.dtype), rec0)

        def cond(c):
            t, states = c[0], c[1]
            return (t < tmax) & jnp.any(~states.done)

        def body(c):
            t, states, tss, bufs = c
            new, nts, rec = jax.vmap(one, in_axes=(0, 0, 0, 0, None))(states, tss, k_loops, game_ids, K)
            bufs = jax.tree.map(lambda b, r: b.at[:, t].set(r), bufs, rec)
            return t + 1, new, nts, bufs

        t, states, _, bufs = jax.lax.while_loop(cond, body, (jnp.int32(0), states, tss, bufs))
        return t, states, bufs

    return game, jax.jit(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--tmax", type=int, default=80)
    ap.add_argument("--detail", type=int, default=1)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-g", default="runs/ppo-modes-seed0")
    ap.add_argument("--run-s", default=None)
    ap.add_argument("--snap-g", default=None)
    ap.add_argument("--snap-s", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_s = a.run_s or f"runs/ppo-modes-{a.mode}-seed0"
    t0 = time.time()
    game, run = build(a.mode, a.tmax, bool(a.detail), a.run_g, run_s, a.snap_g, a.snap_s, sample=bool(a.sample), eps=a.eps)
    ids = jnp.arange(a.games, dtype=jnp.int32)
    t, states, bufs = run(cell_key(a.seed, a.mode), ids, jnp.int32(10**6))
    jax.block_until_ready(bufs)
    t = int(t)
    print(f"played {a.games} games of {a.mode}: {t} loop turns, {time.time() - t0:.1f}s", flush=True)
    out = {k: np.asarray(v)[:, :t] for k, v in bufs.items()}
    out["final_elim_turn"] = np.asarray(states.elim_turn)
    out["final_elim_cause"] = np.asarray(states.elim_cause)
    out["final_length"] = np.asarray(states.length)
    out["final_alive"] = np.asarray(states.alive)
    out["final_turn"] = np.asarray(states.turn)
    out["final_done"] = np.asarray(states.done)
    np.savez_compressed(a.out, **out)
    print("unfinished games:", int((~out["final_done"]).sum()), "max turn", int(out["final_turn"].max()))


if __name__ == "__main__":
    main()
