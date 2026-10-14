"""Constrictor self-play with the trainer's own rollout code (sampled, final params), G and the constrictor
specialist S. For each: explained variance and MSE of V against (a) the trainer's GAE(0.95) lambda-returns and
(b) Monte Carlo discounted returns (lambda=1), plus the variance of each return. Tests whether G's lower logged
return_var (and hence its EV gap decomposition) is a property of its games or of its critic.
Usage: python value_mc.py RUN_DIR [chunks]"""
import sys, dataclasses, time
sys.path.insert(0, '/home/user/slinkydotcpp/baselines')
import jax, jax.numpy as jnp, numpy as np
import ppo_modes as pm
run = sys.argv[1]; chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 20
cfg = pm.load_config(run)
params = pm.load_params(run, cfg)
cfg = dataclasses.replace(cfg, game_modes=('constrictor',), num_envs=32, num_steps=32)
groups = pm.make_groups(cfg)
assert len(groups) == 1 and groups[0].name == 'constrictor'
rollout = jax.jit(pm.make_rollout(cfg, groups)); values_of = jax.jit(pm.make_values(cfg, groups))
runner = pm.init_runner(cfg, groups, jax.random.key(123))
states = runner.env_states
key = jax.random.key(42)
keep = ('values', 'rewards', 'valid', 'terminal', 'done', 'boot_values')
buf = {k: [] for k in keep}
t0 = time.time()
for c in range(chunks + 3):  # 3 warm-up chunks (96 turns) discarded
    key, k = jax.random.split(key)
    states, trajs, _ = rollout(params, states, k)
    tr = trajs[0]
    if c >= 3:
        for k_ in keep: buf[k_].append(np.asarray(getattr(tr, k_)))
last = np.asarray(values_of(params, states)[0])
arr = {k: np.concatenate(v, 0) for k, v in buf.items()}
print(f'{run}: {arr["values"].shape} collected in {time.time()-t0:.0f}s')
def gae(lam):
    adv, ret = pm.compute_gae(jnp.asarray(arr['rewards']), jnp.asarray(arr['values']), jnp.asarray(arr['valid']),
                              jnp.asarray(arr['terminal']), jnp.asarray(arr['done']), jnp.asarray(arr['boot_values']),
                              jnp.asarray(last), cfg.gamma, lam)
    return np.asarray(ret)
T = arr['values'].shape[0]
v = arr['values']; valid = arr['valid'].copy()
valid[T - 64:] = False  # drop the last 64 turns: their MC returns are cut by the end of collection
r95, r1 = gae(cfg.gae_lambda), gae(1.0)
rng = np.random.default_rng(0)
E = v.shape[1]
def summ(ret):
    x, y = ret[valid], v[valid]
    ev = 1 - np.var(x - y) / np.var(x)
    # bootstrap over envs (each env is a long sequence of games)
    eidx = np.broadcast_to(np.arange(E)[None, :, None], valid.shape)[valid]
    bs = []
    for _ in range(500):
        w = np.bincount(rng.integers(0, E, E), minlength=E)[eidx].astype(float)
        def wv(z): m = (w * z).sum() / w.sum(); return (w * (z - m) ** 2).sum() / w.sum()
        bs.append((1 - wv(x - y) / wv(x), wv(x), (w * (x - y) ** 2).sum() / w.sum()))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return ev, np.var(x), np.mean((x - y) ** 2), lo, hi
for nm, ret in (('GAE lambda=0.95 (trainer targets)', r95), ('Monte Carlo lambda=1', r1)):
    ev, var, mse, lo, hi = summ(ret)
    print(f'  {nm:34s} EV {ev:.3f} [{lo[0]:.3f},{hi[0]:.3f}]  Var(ret) {var:.4f} [{lo[1]:.4f},{hi[1]:.4f}]  MSE {mse:.4f} [{lo[2]:.4f},{hi[2]:.4f}]')
print(f'  Var(V) {np.var(v[valid]):.4f}  samples {valid.sum()}')
np.savez_compressed(sys.argv[3] if len(sys.argv) > 3 else '/dev/null', **arr) if len(sys.argv) > 3 else None
