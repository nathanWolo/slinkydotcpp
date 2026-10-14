from common import *
rng=np.random.default_rng(1)
trG,evG=load('ppo-modes-seed0')
SP={m:load(f'ppo-modes-{m}-seed0') for m in MODES}
M16={m:load(f'ppo-modes-{m}-m16-seed0') for m in ['standard','royale']}
KEYS=['approx_kl','clip_frac','entropy','explained_var','return_var','v_loss','mean_game_len','draw_rate','death_self','death_wall','death_body','death_h2h','death_starve','death_hazard']
def blockboot(x,g,lo,hi,B=2000,blk=4):
    """mean over records lo:hi, weights g (or None), block bootstrap CI (blocks of `blk` consecutive records)"""
    x=x[lo:hi]; w=np.ones_like(x) if g is None else g[lo:hi]
    n=len(x); nb=n//blk
    xb=x[:nb*blk].reshape(nb,blk); wb=w[:nb*blk].reshape(nb,blk)
    idx=rng.integers(0,nb,(B,nb))
    num=(xb*wb).sum(1)[idx].sum(1); den=wb.sum(1)[idx].sum(1)
    return (xb*wb).sum()/wb.sum(), num/den
def stat(tr,mode,key,lo,hi):
    x=series(tr,mode,key); g=series(tr,mode,'games')
    wts=g if (key in('mean_game_len','draw_rate') or key.startswith('death_')) else None
    return blockboot(x,wts,lo,hi)
for (lo,hi,lab) in [(144,160,'last window (4.72-5.24M)'),(80,160,'second half (2.6-5.24M)')]:
    print(f'\n##### G minus S, {lab}: diff of means [95% block-bootstrap CI, blocks of 4 records]')
    print('metric           '+'  '.join(f'{m:>26s}' for m in MODES))
    for key in KEYS:
        row=[]
        for m in MODES:
            a,ba=stat(trG,m,key,lo,hi); b,bb=stat(SP[m][0],m,key,lo,hi)
            d=ba-bb; row.append(f'{a-b:+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]')
        print(f'{key:16s} '+'  '.join(f'{r:>26s}' for r in row))
    print('G absolute / S absolute:')
    for key in KEYS:
        print(f'{key:16s} '+'  '.join(f'{stat(trG,m,key,lo,hi)[0]:.3f}/{stat(SP[m][0],m,key,lo,hi)[0]:.3f}'.rjust(26) for m in MODES))
# DiD: constrictor (G-S) minus mean of other three (G-S)
print('\n##### difference-in-differences second half: (G-S)_constrictor - mean over 3 other modes of (G-S)  [95% CI]')
lo,hi=80,160
for key in KEYS:
    d={}
    for m in MODES:
        a,ba=stat(trG,m,key,lo,hi); b,bb=stat(SP[m][0],m,key,lo,hi); d[m]=(a-b,ba-bb)
    others=np.mean([d[m][0] for m in MODES[:3]]); ob=np.mean([d[m][1] for m in MODES[:3]],0)
    did=d['constrictor'][0]-others; db=d['constrictor'][1]-ob
    print(f'{key:16s} {did:+.3f} [{np.percentile(db,2.5):+.3f},{np.percentile(db,97.5):+.3f}]  (constrictor G-S {d["constrictor"][0]:+.3f}, others avg {others:+.3f})')
