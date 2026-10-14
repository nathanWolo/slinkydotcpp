"""Recompute snapshot cell (d) paired differences and the royale matched-snapshot numbers from snapshots/cells.jsonl;
check overlap of (d) 5.24M games with the benchmark cell."""
import json, numpy as np
D = "docs/research/multimode_constrictor_analysis/snapshots/cells.jsonl"
C = [json.loads(l) for l in open(D)]
L = [json.loads(l) for l in open('/home/user/slinkydotcpp/benchmarks/results/modes.jsonl')]
def sh(r, seat):
    p = r['per_game']; t = p['share'][seat]; w = p['width']
    return np.array([int(t[i:i+w], 36) for i in range(0, len(t), w)]) / p['share_scale']
SN = [1048576, 2097152, 3145728, 4194304, 5242880]
def find(mode, lineup, n0, n1=None):
    out = []
    for r in C:
        if r['game_mode'] != mode or r['lineup'] != lineup: continue
        if r['seat_names'][0].endswith(n0) and (n1 is None or r['seat_names'][1].endswith(n1)): out.append(r)
    return out
print('lineups present:', sorted({(r['game_mode'], r['lineup']) for r in C}))
names = sorted({r['seat_names'][0] for r in C})
for n in names: print(' ', n)
G = 'runs/ppo-modes-seed0'; S = 'runs/ppo-modes-constrictor-seed0'
def key(r): return (r['seat_names'][0], r['seat_names'][1] if len(set(r['seat_names']))>1 else None)
heur = {}
for r in C:
    if r['lineup'] == 'A,H,H,H':
        heur[r['seat_names'][0]] = sh(r, 'A')
def nm(run, k): return f'ppo_modes:{run}:snapshot={k}'
def pd(a, b):
    n = min(len(a), len(b)); d = a[:n] - b[:n]
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(n), n
print('(d) constrictor vs 3 heuristics')
for k in SN:
    g = heur.get(nm(G, k)); s = heur.get(nm(S, k)) if nm(S,k) in heur else heur.get(f'ppo_modes:{S}')
    print(k, 'G', None if g is None else (round(g.mean(),3), len(g)), 'S', None if s is None else (round(s.mean(),3), len(s)), 'G-S %+.3f +-%.3f n=%d' % pd(g, s) if g is not None and s is not None else '')
g1, g3, g5 = heur[nm(G,SN[0])], heur[nm(G,SN[2])], heur[nm(G,SN[4])]
print('G 1M->3M %+.3f +-%.3f' % pd(g3, g1)[:2], ' 3M->5M %+.3f +-%.3f' % pd(g5, g3)[:2], ' 1M->5M %+.3f +-%.3f' % pd(g5, g1)[:2])
s1 = heur[nm(S,SN[0])]; s5 = heur.get(nm(S,SN[4]), heur.get(f'ppo_modes:{S}'))
print('S 1M->5M %+.3f +-%.3f' % pd(s5, s1)[:2])
# overlap with benchmark cell
bc = [r for r in L if r['game_mode']=='constrictor' and r['lineup']=='G,H,H,H'][0]; bg = sh(bc,'G')
bs = [r for r in L if r['game_mode']=='constrictor' and r['lineup']=='S,H,H,H'][0]; bsS = sh(bs,'S')
print('snapshot G5 vs benchmark G first 512: identical frac', np.mean(g5 == bg[:len(g5)]), 'means', g5.mean(), bg[:len(g5)].mean())
print('snapshot S5 vs benchmark S first 512: identical frac', np.mean(s5 == bsS[:len(s5)]), s5.mean(), bsS[:len(s5)].mean())
print('benchmark second half G-S: %+.3f +-%.3f' % pd(bg[512:], bsS[512:])[:2])
# correlation between consecutive snapshot cells (pairing benefit)
print('corr G1,G3', np.corrcoef(g1, g3)[0,1])
print('royale A,B,A,B')
R = 'runs/ppo-modes-royale-seed0'
for r in C:
    if r['game_mode']=='royale':
        a = sh(r,'A'); print(r['seat_names'][0].split(':',1)[1], 'vs', r['seat_names'][1].split(':',1)[1], round(a.mean(),3), len(a), 'distinct-ish ties', np.mean(a==0.5).round(3))
