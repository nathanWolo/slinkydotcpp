"""How many DISTINCT games do the constrictor evaluations contain? Count distinct trajectories among
games of one network (G or S) against 3 heuristics, with play_table's seat rotation and key derivation."""
import sys, hashlib, time
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
sys.path.insert(0, "/home/user/slinkydotcpp/baselines")
import jax, jax.numpy as jnp, numpy as np
import ppo_modes as pm
from slinky import modes
from slinky.env import BattlesnakeEnv
from slinky.heuristic import heuristic
import simulate as sim
mode = "constrictor"; game = modes.mode_config(mode); env = BattlesnakeEnv(game, obs="egocentric")
n_games = int(sys.argv[1]); which = sys.argv[2]  # "G" or "S" or "random"
run = "runs/ppo-modes-seed0" if which == "G" else f"runs/ppo-modes-{mode}-seed0"
cfg = pm.load_config(run); params = pm.load_params(run, cfg)
net = pm.make_policy(params, cfg, game, greedy=True)
heur = heuristic(env)
from slinky.evaluate import random_legal
rnd = random_legal(env)
other = rnd if which == "random" else heur
lineup = jnp.array([0, 1, 1, 1])
key = sim.cell_key(0, mode)
N = 4
def one(g):
    k_reset, k_loop = jax.random.split(jax.random.fold_in(key, g))
    state, ts = env.reset(k_reset)
    def body(c, _):
        state, ts = c
        ks = jax.random.split(jax.random.fold_in(k_loop, state.turn), 3)
        a0 = net(ks[0], state, ts); a1 = other(ks[1], state, ts)
        owner = lineup[(jnp.arange(N) - g) % N]
        act = jnp.where(owner == 0, a0, a1).astype(jnp.int32)
        new, nts = env.step(ks[2], state, act)
        return (new, nts), (act, state.alive & ~state.done)
    (state, _), (acts, alive) = jax.lax.scan(body, (state, ts), None, length=60)
    return acts, alive, state.done
t0 = time.time()
acts, alive, done = jax.jit(jax.vmap(one))(jnp.arange(n_games, dtype=jnp.int32))
acts, alive, done = np.asarray(acts), np.asarray(alive), np.asarray(done)
print("all games finished:", bool(done.all()), f"[{time.time()-t0:.0f}s]")
sig = set(); 
for g in range(n_games):
    a = np.where(alive[g], acts[g], -1)
    sig.add(hashlib.md5(a.tobytes()).hexdigest())
print(f"{which} vs 3 {'random_legal' if which=='random' else 'heuristic'} snakes, {n_games} games: {len(sig)} distinct trajectories (action sequences incl. seat order)")
