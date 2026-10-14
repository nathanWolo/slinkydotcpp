"""How often does telling the generalist a wrong mode change its greedy move?

States from the generalist's own sampled self-play in each mode (64 games x 200 turns,
autoreset); decisions are live snakes with at least two legal moves. Also the same with
the right mode but each feature group zeroed, to see what the features carry.
"""
import sys
sys.path.insert(0, "/home/user/slinkydotcpp/baselines")
import jax, jax.numpy as jnp, numpy as np
import ppo_modes as pm
from slinky import modes
from slinky.env import BattlesnakeEnv

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo-modes-seed0"
cfg = pm.load_config(RUN); params = pm.load_params(RUN, cfg)
B, T = 64, 200
for mode in modes.MODE_NAMES:
    game = modes.mode_config(mode)
    env = BattlesnakeEnv(game, obs="egocentric")
    pol = pm.make_policy(params, cfg, game, greedy=False)

    def step(carry, key):
        states, ts = carry
        k1, k2 = jax.random.split(key)
        acts = jax.vmap(pol)(jax.random.split(k1, B), states, ts)
        nstates, nts = jax.vmap(env.step_autoreset)(jax.random.split(k2, B), states, acts)
        return (nstates, nts), (states, ts)

    k0, k1 = jax.random.split(jax.random.key(7))
    states, ts = jax.vmap(env.reset)(jax.random.split(k0, B))
    (states, ts), _ = step((states, ts), k0)  # an autoreset step first: the carry's structure
    _, (S, TS) = jax.jit(lambda c, k: jax.lax.scan(step, c, jax.random.split(k, T)))((states, ts), k1)
    flat = lambda x: x.reshape(-1, *x.shape[2:])
    S, TS = jax.tree.map(flat, S), jax.tree.map(flat, TS)
    mask = np.asarray(TS.action_mask)  # [M, N, 4]
    alive = np.asarray(S.alive) & ~np.asarray(S.done)[:, None]
    choose = alive & (mask.sum(-1) >= 2)

    def greedy(told):
        f = pm.make_logits_fn(params, cfg, game, told)
        lg = jax.jit(jax.vmap(f))(S, TS.obs)
        return np.asarray(jnp.argmax(jnp.where(TS.action_mask, lg, -jnp.inf), -1))

    right = greedy(None)
    parts = [f"{mode:12} decisions {choose.sum():6d}"]
    for other in modes.MODE_NAMES:
        if other == mode:
            continue
        diff = (greedy(other) != right)[choose].mean()
        parts.append(f"told {other}: {100 * diff:4.1f}%")
    print("  ".join(parts), flush=True)
