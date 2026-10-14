from common import *
import sys
rng=np.random.default_rng(0)
trG,evG=load('ppo-modes-seed0')
SP={m:load(f'ppo-modes-{m}-seed0') for m in MODES}
M16={m:load(f'ppo-modes-{m}-m16-seed0') for m in ['standard','royale']}
NW=10; PER=16  # 16 records x 32768 = 524288 env steps per window
KEYS=['approx_kl','clip_frac','entropy','explained_var','value_mean','return_var','value_weight','v_loss','pg_loss','mean_game_len','draw_rate','deaths_total','death_self','death_wall','death_body','death_h2h','death_starve','death_hazard','valid_frac']
def wmean(x,wts=None):
    return np.average(x,weights=wts)
def boot(x,wts=None,B=2000):
    n=len(x); idx=rng.integers(0,n,(B,n))
    if wts is None: v=x[idx].mean(1)
    else: v=(x[idx]*wts[idx]).sum(1)/wts[idx].sum(1)
    return np.percentile(v,[2.5,97.5])
def winstats(tr,mode,key):
    x=series(tr,mode,key); g=series(tr,mode,'games')
    # game-length and per-game rates weighted by games
    wts=g if (key in('mean_game_len','draw_rate','deaths_total') or key.startswith('death_')) else None
    out=[]
    for w in range(NW):
        sl=slice(w*PER,(w+1)*PER)
        xs=x[sl]; ws=None if wts is None else wts[sl]
        m=wmean(xs,ws); lo,hi=boot(xs,ws)
        out.append((m,lo,hi))
    return out
if __name__=='__main__':
    res={}
    for key in KEYS:
        for mode in MODES:
            res[f'G|{mode}|{key}']=winstats(trG,mode,key)
            res[f'S|{mode}|{key}']=winstats(SP[mode][0],mode,key)
        for mode in M16:
            res[f'S16|{mode}|{key}']=winstats(M16[mode][0],mode,key) if len(M16[mode][0])>=NW*PER else None
    json.dump(res,open('windows.json','w'))
    def fmt(t,p=3): return f'{t[0]:.{p}f}'
    for key in KEYS:
        print(f'\n### {key}  (window mean over 16 log records; [95% bootstrap CI over records] for last window)')
        hdr='mode run '+' '.join(f'w{w}' for w in range(NW))
        print(hdr)
        for mode in MODES:
            for run in ['G','S','S16']:
                v=res.get(f'{run}|{mode}|{key}')
                if v is None: continue
                print(f'{mode:12s} {run:3s} '+' '.join(f'{t[0]:.3f}' for t in v), ' last CI [%.3f,%.3f]'%(v[-1][1],v[-1][2]))
