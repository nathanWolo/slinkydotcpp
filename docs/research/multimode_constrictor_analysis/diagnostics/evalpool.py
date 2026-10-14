from common import *
trG,evG=load('ppo-modes-seed0')
SPc=load('ppo-modes-constrictor-seed0')[1]
def get(ev,m,opp):
    t={r['env_steps']:r for r in ev if r['game_mode']==m and r['opponent']==opp}; k=sorted(t)
    return np.array(k),np.array([t[i]['score'] for i in k]),np.array([t[i]['score_ci95'] for i in k])/1.96
def pool(y,se):
    w=1/se**2; return (w*y).sum()/w.sum(), 1/np.sqrt(w.sum())
for lab,ev in [('G',evG),('S',SPc)]:
    k,y,se=get(ev,'constrictor','heuristic')
    m,s=pool(y,se); print(f'{lab} constrictor vs heuristic pooled 5 pts: {m:.3f} ± {1.96*s:.3f}; chi2 vs const {(((y-m)/se)**2).sum():.1f} (4 dof)')
    mid=[1,2]; oth=[0,3,4]
    a,sa=pool(y[mid],se[mid]); b,sb=pool(y[oth],se[oth])
    print(f'   mid (2.10,3.15M) {a:.3f}±{1.96*sa:.3f}  vs others (1.05,4.19,5.24M) {b:.3f}±{1.96*sb:.3f}  diff {a-b:+.3f}±{1.96*np.hypot(sa,sb):.3f}  (post hoc)')
    d=y[4]-y[0]; sd=np.hypot(se[0],se[4]); print(f'   5.24M minus 1.05M {d:+.3f} ± {1.96*sd:.3f};  5.24 minus 2.10: {y[4]-y[1]:+.3f}±{1.96*np.hypot(se[1],se[4]):.3f}')
    l,sl=pool(y[3:],se[3:]); print(f'   last two (4.19,5.24M) pooled {l:.3f}±{1.96*sl:.3f}')
# G-S head to head difference on matched training-time points, pooled over the last 3
k,yg,sg=get(evG,'constrictor','heuristic'); k,ys,ss=get(SPc,'constrictor','heuristic')
d=yg-ys; sd=np.hypot(sg,ss)
m,s=pool(d,sd); print(f'G-S constrictor vs heuristic pooled over all 5: {m:+.3f} ± {1.96*s:.3f};  last 3: {pool(d[2:],sd[2:])[0]:+.3f} ± {1.96*pool(d[2:],sd[2:])[1]:.3f}; first: {d[0]:+.3f}±{1.96*sd[0]:.3f}')
# same for other modes (last 3 points)
for m_ in MODES:
    SPm=load(f'ppo-modes-{m_}-seed0')[1]
    k,yg,sg=get(evG,m_,'heuristic'); k,ys,ss=get(SPm,m_,'heuristic'); d=yg-ys; sd=np.hypot(sg,ss)
    p=pool(d[2:],sd[2:]); q=pool(d,sd)
    print(f'{m_:12s} G-S vs heuristic: last 3 pooled {p[0]:+.3f} ± {1.96*p[1]:.3f}; all 5 pooled {q[0]:+.3f} ± {1.96*q[1]:.3f}')
