"""C14: G told a wrong mode vs told constrictor, on the same 256 constrictor game ids (vs 3 heuristics); pooled."""
import json, numpy as np
L = [json.loads(l) for l in open('/home/user/slinkydotcpp/benchmarks/results/modes.jsonl')]
def sh(r, seat):
    p = r['per_game']; t = p['share'][seat]; w = p['width']
    return np.array([int(t[i:i+w], 36) for i in range(0, len(t), w)]) / p['share_scale']
def cell(mode, lineup): return [r for r in L if r['game_mode'] == mode and r['lineup'] == lineup][0]
rng = np.random.default_rng(1)
for mode in ['constrictor', 'royale', 'standard']:
    base = sh(cell(mode, 'G,H,H,H'), 'G')[:256]
    others = [m for m in ['standard', 'duel', 'royale', 'constrictor'] if m != mode]
    W = np.stack([sh(cell(mode, f'G@{m},H,H,H'), f'G@{m}') for m in others])
    d = W.mean(0) - base
    idx = rng.integers(0, 256, (4000, 256)); bs = d[idx].mean(1)
    print(f'{mode}: told-own (first 256) {base.mean():.3f}; wrong modes {dict(zip(others, W.mean(1).round(3)))}; pooled wrong-own {d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]; full 1024 own {sh(cell(mode,"G,H,H,H"),"G").mean():.3f}')
