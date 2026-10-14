import sys
sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
sys.path.insert(0, "/home/user/slinkydotcpp/baselines")
import numpy as np, jax, jax.numpy as jnp
import ppo_modes as pm
from slinky import modes
from slinky.env import BattlesnakeEnv
from slinky.evaluate import play_table
import simulate as sim
from common import *
mode = sys.argv[1]; path = sys.argv[2]; n = int(sys.argv[3])
game = modes.mode_config(mode); env = BattlesnakeEnv(game, obs="egocentric")
cg = pm.load_config("runs/ppo-modes-seed0"); cs = pm.load_config(f"runs/ppo-modes-{mode}-seed0")
pg = pm.make_policy(pm.load_params("runs/ppo-modes-seed0", cg), cg, game, greedy=False)
ps = pm.make_policy(pm.load_params(f"runs/ppo-modes-{mode}-seed0", cs), cs, game, greedy=False)
r = play_table(env, (pg, ps), (0, 1, 0, 1), sim.cell_key(0, mode), n, max_turns=1000, batch_size=n, per_game=True)
d = contest_outcome(load(path, dedupe=False))
mine = d["share"][:n, 0]
print("play_table (sampled policies) share", r.share[0], "per-game vs mine max abs diff", np.abs(r.game_share[:, 0] - mine).max(), " mine mean", mine.mean())
