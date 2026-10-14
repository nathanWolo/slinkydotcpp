from common import *
trG,evG=load('ppo-modes-seed0')
print([r['env_steps'] for r in trG][:5],[r['env_steps'] for r in trG][-3:])
for m in MODES:
    print(m,'games/record',series(trG,m,'games')[:5],series(trG,m,'games')[-5:])
# check value_weight formula
vw={m:series(trG,m,'value_weight') for m in MODES}
rv={m:series(trG,m,'return_var') for m in MODES}
inv=np.array([1/np.maximum(rv[m],0.01) for m in MODES])
w=inv/inv.mean(0)
for i,m in enumerate(MODES):
    print(m,'max abs diff formula vs logged',np.abs(w[i]-vw[m]).max(), 'mean ratio',(vw[m]/w[i]).mean())
print('sum vw',sum(vw[m] for m in MODES)[:5])
for r in trG[:2]+trG[-2:]:
    print(r['env_steps'],{m:(round(r['game_modes'][m]['return_var'],4),round(r['game_modes'][m]['value_weight'],3)) for m in MODES})
