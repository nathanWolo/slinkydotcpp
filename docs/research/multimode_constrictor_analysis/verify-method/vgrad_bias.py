"""Is the value-gradient alignment among food modes carried by a shared calibration offset (value-head bias and
the mean direction), and what does the value-head bias gradient say about over/under-prediction per mode?
Uses the gradients team's full-rollout gradients ('full' [K, G, part, P]); split-half cosines (k != k')."""
import sys, numpy as np
sys.path.insert(0, '/home/user/slinkydotcpp/baselines')
GD = 'docs/research/multimode_constrictor_analysis/gradients'
sys.path.insert(0, GD)
import jax, ppo_modes as pm
from grads import layer_slices
cfg = pm.load_config('/home/user/slinkydotcpp/runs/ppo-modes-seed0')
sl, P, names = layer_slices(pm.init_network(jax.random.key(0), cfg))
vb = [s for (n, s, k, sh) in names if n == 'value/b']
print('value leaves', [(n, s, k) for (n, s, k, sh) in names if n.startswith('value') or n.startswith('hidden')])
vbias = [s for (n, s, k, sh) in names if n.startswith('value') and k == 1][0]
hb = [np.arange(s, s + k) for (n, s, k, sh) in names if n.startswith('hidden') and len(sh) == 1][0]
val = sl['value']; torso = np.concatenate([sl['conv'], sl['hidden']])
mask_novh = np.ones(P, bool); mask_novh[val] = False
def cos(a, b): return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
M = ['standard', 'duel', 'royale', 'constrictor']
for f in ['G-1048576', 'G-2097152', 'G-3145728', 'G-4194304', 'G-5242880', 'Sconstrictor-5242880']:
    z = np.load(f'{GD}/{f}.npz'); parts = list(z['parts']); vi = parts.index('v') if 'v' in parts else 1
    g = z['full'][:, :, vi, :].astype(np.float64)  # [K, G, P]
    K = g.shape[0]
    print(f'\n{f} parts={parts}')
    print('  value-head bias grad (sign>0 => V above targets on average), per rollout:',
          np.round(g[:, :, vbias], 4).tolist())
    print('  share of |g_v|^2 in value head:', np.round((g[:, :, val] ** 2).sum(-1) / (g ** 2).sum(-1), 3).tolist())
    for lab, idx in (('all', slice(None)), ('no value head', mask_novh), ('torso', torso)):
        C = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                C[a, b] = np.mean([cos(g[k, a, idx], g[l, b, idx]) for k in range(K) for l in range(K) if k != l])
        cf = np.mean([C[3, 0], C[3, 1], C[3, 2]]); fp = np.mean([C[0, 1], C[0, 2], C[1, 2]])
        print(f'  raw split-half cosines [{lab}]: food pairs {fp:+.3f}, C vs food {cf:+.3f}, ceilings {np.round(np.diag(C),2).tolist()}, C|S {C[3,0]:+.3f}')
