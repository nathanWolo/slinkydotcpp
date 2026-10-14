from common import *
from scipy import stats
trG,evG=load('ppo-modes-seed0')
SP={m:load(f'ppo-modes-{m}-seed0') for m in MODES}
def table(ev,mode,opp):
    return {r['env_steps']:r for r in ev if r['game_mode']==mode and r['opponent']==opp}
def wls(x,y,se):
    w=1/se**2; X=np.c_[np.ones_like(x),x]; A=X.T@(X*w[:,None]); b=np.linalg.solve(A,X.T@(w*y)); cov=np.linalg.inv(A)
    return b[1],np.sqrt(cov[1,1]), (w*(y-X@b)**2).sum()
for opp in ['heuristic','random_legal']:
    print(f'\n=== vs {opp}: weighted linear slope of score vs env steps (per 1M steps/mode), 5 eval points, SE from logged ci95/1.96; chi2 for constancy (4 dof)')
    for m in MODES:
        for lab,ev in [('G',evG),('S',SP[m][1])]:
            t=table(ev,m,opp); k=sorted(t); x=np.array(k)/1e6
            y=np.array([t[i]['score'] for i in k]); se=np.array([t[i]['score_ci95'] for i in k])/1.96
            # guard against zero se (score 1.0)
            se=np.maximum(se,0.004)
            s,ss,chi=wls(x,y,se)
            mu=np.average(y,weights=1/se**2); chic=(((y-mu)/se)**2).sum()
            early=np.average(y[:2],weights=1/se[:2]**2); se_e=1/np.sqrt((1/se[:2]**2).sum())
            late=np.average(y[2:],weights=1/se[2:]**2); se_l=1/np.sqrt((1/se[2:]**2).sum())
            dl=late-early; sdl=np.hypot(se_e,se_l)
            print(f'{m:12s} {lab}: slope {s:+.4f}/M ± {1.96*ss:.4f}   chi2(const) {chic:5.1f} (p={1-stats.chi2.cdf(chic,4):.3f})   late(3.15-5.24)-early(1.05-2.10) = {dl:+.3f} ± {1.96*sdl:.3f}   [early {early:.3f}, late {late:.3f}]')
