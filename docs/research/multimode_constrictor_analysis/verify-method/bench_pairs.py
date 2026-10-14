"""Paired comparisons from benchmarks/results/modes.jsonl (read only)."""
import json, numpy as np
rows=[json.loads(l) for l in open('/home/user/slinkydotcpp/benchmarks/results/modes.jsonl')]
def dec(pg,x):
    w=pg['width']; t=pg['share'][x]; return np.array([int(t[i:i+w],36) for i in range(0,len(t),w)])/pg['share_scale']
def get(mode, lineup):
    for r in rows:
        if r['game_mode']==mode and r['lineup']==lineup: return r
def sh(mode, lu):
    r=get(mode,lu); pg=r['per_game']; return dec(pg, list(pg['share'].keys())[0])
rng=np.random.default_rng(0)
def pboot(a,b,B=4000):
    n=min(len(a),len(b)); a=a[:n]; b=b[:n]; d=a-b
    idx=rng.integers(0,n,(B,n)); bs=d[idx].mean(1)
    return d.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5), n
for mode in ['constrictor','royale']:
    g=sh(mode,'G,H,H,H'); s=sh(mode,'S,H,H,H')
    print(mode,'G-S vs 3H paired', [round(x,4) for x in pboot(g,s)[:3]], 'first512', [round(x,4) for x in pboot(g[:512],s[:512])[:3]], 'second512', [round(x,4) for x in pboot(g[512:],s[512:])[:3]])
    for lu in ['G@standard,H,H,H','G@duel,H,H,H','G@royale,H,H,H','G@constrictor,H,H,H']:
        r=get(mode,lu)
        if r is None: continue
        w=sh(mode,lu); n=len(w)
        print('  ',lu, 'mean',round(w.mean(),4),'vs G first',n, round(g[:n].mean(),4), 'identical games', int((w==g[:n]).sum()), 'of', n, 'paired diff', [round(x,4) for x in pboot(w,g[:n])[:3]])
print('royale G - S16 vs 3H paired', [round(x,4) for x in pboot(sh('royale','G,H,H,H'), sh('royale','S16,H,H,H'))[:3]])
print('royale S - S16 vs 3H paired', [round(x,4) for x in pboot(sh('royale','S,H,H,H'), sh('royale','S16,H,H,H'))[:3]])
for lu in ['G,S,G,S','G,S16,G,S16','S16,S,S16,S']:
    r=get('royale',lu); pg=r['per_game']; k=list(pg['share'].keys()); a=dec(pg,k[0]); print('royale',lu,k,round(a.mean(),4), len(a))
