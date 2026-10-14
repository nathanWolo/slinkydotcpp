"""Collect decision states from games between the generalist G and a specialist S.

For every turn of every game it stores the raw state (bodies, heads, lengths, alive,
health, hazards), the env action mask, the seat assignment, the move actually played,
and the *unmasked* logits of BOTH policies (G and S) for every seat, on the real
observation and on two counterfactual observations:

* ``far``: every body cell's countdown (own_countdown, opp_countdown channels) set to the
  maximum countdown value present in that observation (all bodies look 'far from
  clearing', like a head);
* ``near``: every body cell's countdown set to 2/121, the smallest countdown a
  constrictor body ever shows (all bodies look like stacked tails, 'about to clear').

Usage: python collect.py MODE LINEUP NUM_GAMES MAX_TURNS SEED OUT.npz [PLAY]
  PLAY: greedy (default) or sample (each seat samples from its own policy; diverse states)
  LINEUP: gsgs (G,S,G,S rotated by game), ssss, gggg
Run from the repo root.
"""

import sys
import time

sys.path.insert(0, "baselines")
sys.path.insert(0, "src")

import jax
import jax.numpy as jnp
import numpy as np

import ppo_modes as pm
from slinky.env import BattlesnakeEnv
from slinky.modes import mode_config
from slinky.observations import CHANNEL_NAMES

OWN_BODY = CHANNEL_NAMES.index("own_body")
OWN_CD = CHANNEL_NAMES.index("own_countdown")
OPP_BODY = CHANNEL_NAMES.index("opp_body")
OPP_CD = CHANNEL_NAMES.index("opp_countdown")
NEAR_VALUE = 2.0 / 121.0


def counterfactual(obs, kind):
    """obs f32[N, R, C, 13] -> the same with body countdowns replaced."""
    own_b = obs[..., OWN_BODY] > 0
    opp_b = obs[..., OPP_BODY] > 0
    if kind == "far":
        m = jnp.maximum(obs[..., OWN_CD].max(axis=(-2, -1)), obs[..., OPP_CD].max(axis=(-2, -1)))
        v = m[:, None, None]
    else:
        v = jnp.float32(NEAR_VALUE)
    own = jnp.where(own_b, v, obs[..., OWN_CD])
    opp = jnp.where(opp_b, v, obs[..., OPP_CD])
    return obs.at[..., OWN_CD].set(own).at[..., OPP_CD].set(opp)


def main():
    mode, lineup, num_games, max_turns, seed, out = sys.argv[1:7]
    import os
    if os.path.exists(out + ".skip"):
        print("skipped (marker file", out + ".skip)")
        return
    num_games, max_turns, seed = int(num_games), int(max_turns), int(seed)
    play = sys.argv[7] if len(sys.argv) > 7 else "greedy"
    # openN: each seat samples from its own policy for the first N turns, then plays greedy
    assert play in ("greedy", "sample") or play.startswith("open")
    open_turns = int(play[4:]) if play.startswith("open") else 0
    game = mode_config(mode)
    n = game.num_snakes
    env = BattlesnakeEnv(game, obs="egocentric")

    g_dir = "runs/ppo-modes-seed0"
    s_dir = f"runs/ppo-modes-{mode}-seed0"
    g_cfg, s_cfg = pm.load_config(g_dir), pm.load_config(s_dir)
    g_par = pm.load_params(g_dir, g_cfg, name="params-5242880.npz")
    s_par = pm.load_params(s_dir, s_cfg, name="params-5242880.npz")
    lg_fn = pm.make_logits_fn(g_par, g_cfg, game)
    ls_fn = pm.make_logits_fn(s_par, s_cfg, game)

    if lineup == "gsgs":
        base = np.array([1, 0] * (n // 2) + [1] * (n % 2), bool)
        seats = np.stack([base if g % 2 == 0 else ~base for g in range(num_games)])
    elif lineup == "ssss":
        seats = np.zeros((num_games, n), bool)
    elif lineup == "gggg":
        seats = np.ones((num_games, n), bool)
    else:
        raise ValueError(lineup)

    def greedy(logits, mask):
        return jnp.argmax(jnp.where(mask, logits, -jnp.inf), axis=-1).astype(jnp.int32)

    def one_game(key, seat_g):
        k0, key = jax.random.split(key)
        state, ts = env.reset(k0)

        def step(carry, _):
            state, ts, key = carry
            key, k = jax.random.split(key)
            obs = ts.obs
            lg = lg_fn(state, obs)
            ls = ls_fn(state, obs)
            far = counterfactual(obs, "far")
            near = counterfactual(obs, "near")
            rec = dict(
                body=state.body,
                head=state.head,
                length=state.length,
                health=state.health,
                alive=state.alive,
                hazard=state.hazard,
                food=state.food,
                turn=state.turn,
                done=state.done,
                mask=ts.action_mask,
                lg=lg,
                ls=ls,
                lg_far=lg_fn(state, far),
                ls_far=ls_fn(state, far),
                lg_near=lg_fn(state, near),
                ls_near=ls_fn(state, near),
            )
            greedy_act = jnp.where(seat_g, greedy(lg, ts.action_mask), greedy(ls, ts.action_mask))
            if play == "greedy":
                act = greedy_act
            else:
                ka, kb = jax.random.split(jax.random.fold_in(k, 7))
                act = jnp.where(
                    seat_g,
                    pm.sample_actions(ka, lg, ts.action_mask),
                    pm.sample_actions(kb, ls, ts.action_mask),
                )
                if open_turns:
                    act = jnp.where(state.turn < open_turns, act, greedy_act)
            rec["action"] = act
            state2, ts2 = env.step(k, state, act)
            return (state2, ts2, key), rec

        (final, _, _), recs = jax.lax.scan(step, (state, ts, key), None, length=max_turns)
        recs["final_alive"] = final.alive
        recs["final_elim_turn"] = final.elim_turn
        recs["final_done"] = final.done
        recs["final_turn"] = final.turn
        return recs

    run = jax.jit(jax.vmap(one_game))
    batch = 64
    keys = jax.random.split(jax.random.key(seed), num_games)
    chunks = []
    t0 = time.time()
    for b in range(0, num_games, batch):
        r = run(keys[b : b + batch], jnp.asarray(seats[b : b + batch]))
        r = jax.tree.map(np.asarray, r)
        chunks.append(r)
        print(f"{mode} {lineup}: {b + batch}/{num_games} games, {time.time() - t0:.0f}s", flush=True)
    out_d = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
    out_d["seat_g"] = seats
    # Trim the time axis to the longest game.
    alive_t = ~out_d["done"]  # [G, T]
    last = int(np.max(np.nonzero(alive_t.any(axis=0))[0])) + 1
    for k in list(out_d):
        v = out_d[k]
        if v.ndim >= 2 and v.shape[1] == max_turns and k not in ("seat_g",):
            out_d[k] = v[:, :last]
    print("not finished games:", int((~out_d["final_done"]).sum()), "last turn", last)
    np.savez_compressed(out, **out_d)


if __name__ == "__main__":
    main()
