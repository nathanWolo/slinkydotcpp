"""Counterfactual countdown inputs, recomputed offline from stored states.

Usage: python cf_offline.py MODE games.npz decisions.npz OUT.npz

For every decision, rebuild the state, compute the mover's egocentric observation and
the greedy (legal argmax) choice of G and S under input variants:
  real     unchanged (must reproduce the stored choices: a check)
  far      every body countdown := the observation's max countdown (as in collect.py)
  near     every body countdown := 2/121 (as in collect.py)
  shiftK   every body countdown += K/121, clipped at 1 (all bodies clear K turns later
           than shown; keeps the order and gradients along each body)
Run from the repo root.
"""

import sys

sys.path.insert(0, "baselines")
sys.path.insert(0, "src")

import jax
import jax.numpy as jnp
import numpy as np

import ppo_modes as pm
from slinky.modes import mode_config
from slinky.observations import CHANNEL_NAMES, make_obs_fn
from slinky.types import empty_state

OWN_BODY = CHANNEL_NAMES.index("own_body")
OWN_CD = CHANNEL_NAMES.index("own_countdown")
OPP_BODY = CHANNEL_NAMES.index("opp_body")
OPP_CD = CHANNEL_NAMES.index("opp_countdown")
VARIANTS = ("real", "far", "near", "shift5", "shift10", "shift30")


def transform(obs, variant):
    """obs f32[R, C, 13] (one snake)."""
    if variant == "real":
        return obs
    own_b = obs[..., OWN_BODY] > 0
    opp_b = obs[..., OPP_BODY] > 0
    own, opp = obs[..., OWN_CD], obs[..., OPP_CD]
    if variant == "far":
        v = jnp.maximum(own.max(), opp.max())
        own, opp = jnp.where(own_b, v, own), jnp.where(opp_b, v, opp)
    elif variant == "near":
        v = jnp.float32(2.0 / 121.0)
        own, opp = jnp.where(own_b, v, own), jnp.where(opp_b, v, opp)
    else:
        k = int(variant[5:]) / 121.0
        own = jnp.where(own_b, jnp.minimum(own + k, 1.0), own)
        opp = jnp.where(opp_b, jnp.minimum(opp + k, 1.0), opp)
    return obs.at[..., OWN_CD].set(own).at[..., OPP_CD].set(opp)


def main():
    mode, games_f, dec_f, out = sys.argv[1:5]
    game = mode_config(mode)
    obs_fn = make_obs_fn(game, "egocentric")
    g = np.load(games_f)
    d = np.load(dec_f)
    gi, ti, si, legal = d["game"], d["turn"], d["snake"], d["legal"]
    pars = {}
    for name, run in (("g", "runs/ppo-modes-seed0"), ("s", f"runs/ppo-modes-{mode}-seed0")):
        cfg = pm.load_config(run)
        pars[name] = (pm.load_params(run, cfg, name="params-5242880.npz"), cfg)
    template = empty_state(game)
    fields = ("body", "head", "length", "health", "alive", "hazard", "food", "turn")

    def one(st_fields, snake, lg):
        state = template._replace(**st_fields)
        obs = obs_fn(state)  # [N, R, C, 13]
        res = {}
        for name, (params, cfg) in pars.items():
            boards, feats = pm.policy_inputs(obs, state, game, cfg)
            b, f = boards[snake], feats[snake]
            for v in VARIANTS:
                logits, _ = pm.network(params, transform(b, v)[None], f[None], cfg)
                res[f"c_{name}_{v}"] = jnp.argmax(jnp.where(lg, logits[0], -jnp.inf))
        return res

    run = jax.jit(jax.vmap(one))
    chunk = 4096
    outs = []
    for b in range(0, len(gi), chunk):
        sl = slice(b, b + chunk)
        st = {f: jnp.asarray(g[f][gi[sl], ti[sl]]) for f in fields}
        st["body"] = st["body"].astype(jnp.int16)
        outs.append(jax.tree.map(np.asarray, run(st, jnp.asarray(si[sl]), jnp.asarray(legal[sl]))))
        print(f"{b + chunk}/{len(gi)}", flush=True)
    res = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    print("real==stored G", np.mean(res["c_g_real"] == d["c_lg"]), "S", np.mean(res["c_s_real"] == d["c_ls"]))
    print("far==stored G", np.mean(res["c_g_far"] == d["c_lg_far"]), "S", np.mean(res["c_s_far"] == d["c_ls_far"]))
    print("near==stored G", np.mean(res["c_g_near"] == d["c_lg_near"]), "S", np.mean(res["c_s_near"] == d["c_ls_near"]))
    np.savez_compressed(out, **res)


if __name__ == "__main__":
    main()
