import sys, time, dataclasses
sys.path.insert(0, 'baselines')
import jax, jax.numpy as jnp, numpy as np
import ppo_modes as pm
cfg = pm.load_config('runs/ppo-modes-seed0')
cfg_nn = dataclasses.replace(cfg, norm_adv=False)
params = pm.load_params('runs/ppo-modes-seed0', cfg, name='params-1048576.npz')
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
k = jax.random.key(0)
obs = (jax.random.uniform(k, (B, 21, 21, 13)) > 0.8).astype(jnp.float32)
feats = jax.random.uniform(k, (B, 7))
mask = jnp.ones((B, 4), bool)
actions = jnp.zeros((B,), jnp.int32)
logits, v = pm.network(params, obs, feats, cfg)
lp = jnp.take_along_axis(pm.masked_log_probs(logits, mask), actions[:, None], -1)[:, 0]
mb = pm.Batch(obs, feats, mask, actions, lp, v, jax.random.normal(k, (B,)), jax.random.normal(k, (B,)), jnp.ones((B,), bool))
def f(p, mb):
    lg, vv = pm.network(p, mb.obs, mb.feats, cfg)
    loss, parts = pm.group_loss(lg, vv, mb, 1.0, cfg_nn)
    return jnp.stack([parts['pg_loss'], parts['v_loss'], parts['entropy']])
g1 = jax.jit(jax.grad(lambda p, mb: f(p, mb).sum()))
g3 = jax.jit(jax.jacrev(f))
def vjp3(p, mb):
    out, vjp = jax.vjp(lambda q: f(q, mb), p)
    return [vjp(jnp.eye(3)[i])[0] for i in range(3)]
g3v = jax.jit(vjp3)
for name, fn in [('grad1', g1), ('jacrev3', g3), ('vjp3', g3v)]:
    t = time.perf_counter(); jax.block_until_ready(fn(params, mb)); tc = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(3): jax.block_until_ready(fn(params, mb))
    print(name, f'compile+run {tc:.1f}s, run {(time.perf_counter()-t)/3:.2f}s', flush=True)
