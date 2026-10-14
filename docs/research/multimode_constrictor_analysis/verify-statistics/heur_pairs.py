"""Recompute G-S vs 3 heuristics (paired over game ids) from benchmarks/results/modes.jsonl."""
import json, numpy as np
L = [json.loads(l) for l in open('/home/user/slinkydotcpp/benchmarks/results/modes.jsonl')]
def sh(r, seat):
    p = r['per_game']; t = p['share'][seat]; w = p['width']
    return np.array([int(t[i:i+w], 36) for i in range(0, len(t), w)]) / p['share_scale']
def cell(mode, lineup):
    out = [r for r in L if r['game_mode'] == mode and r['lineup'] == lineup]
    assert len(out) == 1, (mode, lineup, len(out)); return out[0]
rng = np.random.default_rng(0)
for mode in ['royale', 'constrictor', 'standard']:
    g = cell(mode, 'G,H,H,H'); s = cell(mode, 'S,H,H,H')
    a, b = sh(g, 'G'), sh(s, 'S')
    print(mode, 'G', a.mean(), g['results']['G']['share'], 'S', b.mean(), s['results']['S']['share'], 'n', len(a), len(b), 'seeds', g['seed'], s['seed'])
    d = a - b; n = len(d)
    idx = rng.integers(0, n, (4000, n))
    bs = d[idx].mean(1)
    # unpaired
    bu = a[idx].mean(1) - b[rng.integers(0, n, (4000, n))].mean(1)
    print(f'  G-S {d.mean():+.4f} paired boot [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] unpaired [{np.percentile(bu,2.5):+.4f},{np.percentile(bu,97.5):+.4f}] corr(a,b)={np.corrcoef(a,b)[0,1]:.3f}')
# royale S16
s16 = cell('royale', 'S16,H,H,H'); print('royale S16', s16['results']['S16']['share'], s16['seat_names'][0], s16['agents']['S16']['config'].get('checkpoint'), s16['date'])
s16s = cell('standard', 'S16,H,H,H'); print('standard S16', s16s['results']['S16']['share'])
x = sh(s16, 'S16'); y = sh(cell('royale','S,H,H,H'), 'S'); d = x - y; idx = rng.integers(0, len(d), (4000, len(d))); bs = d[idx].mean(1)
print(f'royale S16-S8 {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]')
# G told other modes in constrictor
gc = sh(cell('constrictor','G,H,H,H'),'G')
for m in ['standard','duel','royale']:
    r = cell('constrictor', f'G@{m},H,H,H'); x = sh(r, f'G@{m}'); n = len(x)
    d = x - gc[:n]; idx = rng.integers(0, n, (4000, n)); bs = d[idx].mean(1)
    print(f'constrictor G@{m} {x.mean():.4f} vs G(first {n}) {gc[:n].mean():.4f}, diff {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]')
# cross-mode specialists
for lu in ['S_standard,H,H,H','S_royale,H,H,H','S_duel,H,H,H']:
    for mode in ['constrictor','royale']:
        try:
            r = cell(mode, lu); print(mode, lu, round(r['results'][lu.split(',')[0]]['share'],4), r['games'])
        except AssertionError: pass
