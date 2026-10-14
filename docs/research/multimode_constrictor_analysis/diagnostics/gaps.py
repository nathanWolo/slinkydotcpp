from common import *
rng=np.random.default_rng(2)
trG,evG=load('ppo-modes-seed0')
SP={m:load(f'ppo-modes-{m}-seed0')[0] for m in MODES}
def arr(tr,mode,key):
    x=series(tr,mode,key); g=series(tr,mode,'games')
    w=g if (key in('mean_game_len','draw_rate') or key.startswith('death_')) else np.ones_like(g)
    return x,w
def boot_diff(trA,trB,mode,key,lo,hi,B=2000,blk=4,ratio=False):
    xa,wa=arr(trA,mode,key); xb,wb=arr(trB,mode,key)
    xa,wa,xb,wb=[v[lo:hi] for v in (xa,wa,xb,wb)]
    n=(hi-lo)//blk
    def reshape(v): return v[:n*blk].reshape(n,blk)
    na,da=reshape(xa*wa).sum(1),reshape(wa).sum(1); nb,db=reshape(xb*wb).sum(1),reshape(wb).sum(1)
    ia=rng.integers(0,n,(B,n)); ib=rng.integers(0,n,(B,n))
    ma=na[ia].sum(1)/da[ia].sum(1); mb=nb[ib].sum(1)/db[ib].sum(1)
    pa=na.sum()/da.sum(); pb=nb.sum()/db.sum()
    if ratio: d=ma/mb; p=pa/pb
    else: d=ma-mb; p=pa-pb
    return p,np.percentile(d,2.5),np.percentile(d,97.5)
print('### self-collision deaths per game, G minus S, by 0.52M window (CI: block bootstrap, block=4 records)')
for key in ['death_self','death_wall','death_h2h','mean_game_len']:
    print('--',key)
    print('window(M)  '+'  '.join(f'{(w*16+16)*32768/1e6:>16.2f}' for w in range(10)))
    for m in MODES:
        row=[]
        for w in range(10):
            p,lo,hi=boot_diff(trG,SP[m],m,key,w*16,w*16+16)
            row.append(f'{p:+.2f}[{lo:+.2f},{hi:+.2f}]' if key!='mean_game_len' else f'{p:+.1f}[{lo:+.1f},{hi:+.1f}]')
        print(f'{m:11s}'+'  '.join(f'{r:>16s}' for r in row))
print('\n### ratio G/S of approx_kl and clip_frac, second half (records 80:160) and last third (records 107:160)')
for lo,hi in [(80,160),(107,160),(0,80)]:
    print(f'records {lo}:{hi}')
    for key in ['approx_kl','clip_frac']:
        for m in MODES:
            p,a,b=boot_diff(trG,SP[m],m,key,lo,hi,ratio=True)
            print(f'  {key:10s} {m:12s} G/S = {p:.2f} [{a:.2f},{b:.2f}]')
print('\n### EV slope per 1M steps on records 64:160 (2.1-5.24M), OLS with block-bootstrap CI (blocks of 8)')
def slope(x,lo,hi,blk=8,B=2000):
    t=np.arange(lo,hi)*32768/1e6; y=x[lo:hi]
    def ols(tt,yy): 
        A=np.c_[np.ones_like(tt),tt]; return np.linalg.lstsq(A,yy,rcond=None)[0][1]
    n=(hi-lo)//blk; ti=t[:n*blk].reshape(n,blk); yi=y[:n*blk].reshape(n,blk)
    out=[]
    for _ in range(B):
        idx=rng.integers(0,n,n); out.append(ols(ti[idx].ravel(),yi[idx].ravel()))
    return ols(t,y),np.percentile(out,2.5),np.percentile(out,97.5)
for key in ['explained_var','death_self','entropy']:
    for m in MODES:
        for lab,tr in [('G',trG),('S',SP[m])]:
            x,_=arr(tr,m,key); s,a,b=slope(x,64,160)
            print(f'  {key:14s} {m:12s} {lab}: slope {s:+.3f}/M [{a:+.3f},{b:+.3f}]')
