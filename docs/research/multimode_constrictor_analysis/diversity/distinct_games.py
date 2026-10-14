"""How many distinct games does a line-up produce? Plays B games with independent keys, seats
rotated as in play_table (seat s holds lineup[(s - g) % N]), and counts distinct trajectories
(start heads + every snake's action sequence while alive)."""
import sys, hashlib
import jax, jax.numpy as jnp, numpy as np
from slinky import agents as registry, modes

mode, seats, B = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
names = dict(x.split("=", 1) for x in sys.argv[4:])
game = modes.mode_config(mode)
specs = {s: registry.parse_agent(names[s]) for s in dict.fromkeys(seats)}
built = {s: registry.make_agent(specs[s], game) for s in specs}
env = registry.make_env(game, any(a.needs_obs for a in built.values()))
syms = list(specs); N = game.num_snakes; T = 400

def one_game(key, g):
    owner = jnp.array([syms.index(seats[(s - g) % N]) for s in range(N)]) if False else None
    return None

def play(keys):
    def single(key, g):
        k0, k1 = jax.random.split(key)
        state, ts = env.reset(k0)
        own = jnp.stack([jnp.array([syms.index(x) for x in seats])[(s - g) % N] for s in range(N)])
        def step(carry, k):
            state, ts = carry
            ks = jax.random.split(k, len(syms) + 1)
            acts = jnp.stack([built[s].policy(ks[i], state, ts) for i, s in enumerate(syms)])  # [P, N]
            a = acts[own, jnp.arange(N)]
            alive = state.alive & ~state.done
            nstate, nts = env.step(ks[-1], state, a)
            return (nstate, nts), (jnp.where(alive, a, -1), state.head[:, 0], state.head[:, 1], state.done)
        _, (acts, hx, hy, done) = jax.lax.scan(step, (state, ts), jax.random.split(k1, T))
        return acts, hx[0], hy[0], done
    return jax.jit(jax.vmap(single))(keys, jnp.arange(B))

acts, hx, hy, done = map(np.asarray, play(jax.random.split(jax.random.key(123), B)))
sigs = set(); outs = {}
for g in range(B):
    own = [seats[(i - g) % N] for i in range(N)]
    # canonical: per snake (start cell, its symbol, its moves), as a set: seat labels do not matter
    key = tuple(sorted((int(hx[g, i]), int(hy[g, i]), own[i], tuple(acts[g, :, i][acts[g, :, i] >= 0])) for i in range(N)))
    sigs.add(key)
print(f"{mode} {','.join(seats)}: {len(sigs)} distinct games of {B} (all done by turn {T}: {bool(done[:, -1].all())})")
