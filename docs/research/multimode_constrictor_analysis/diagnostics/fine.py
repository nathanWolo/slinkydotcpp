from common import *
trG,evG=load('ppo-modes-seed0')
SP={m:load(f'ppo-modes-{m}-seed0')[0] for m in MODES}
def win(tr,mode,key,k=8):
    x=series(tr,mode,key); g=series(tr,mode,'games')
    w=g if (key in('mean_game_len','draw_rate') or key.startswith('death_')) else np.ones_like(g)
    return np.array([np.average(x[i:i+k],weights=w[i:i+k]) for i in range(0,len(x),k)])
print('G constrictor vs S constrictor in 8-record (0.262M) windows; end-of-window env steps (M):')
print('end(M)   '+' '.join(f'{(i+8)*32768/1e6:6.2f}' for i in range(0,160,8)))
for key in ['explained_var','entropy','approx_kl','clip_frac','death_self','death_h2h','death_body','death_wall','draw_rate','mean_game_len','return_var','value_weight']:
    print(f'G {key:14s}'+' '.join(f'{v:6.3f}' for v in win(trG,'constrictor',key)))
    print(f'S {key:14s}'+' '.join(f'{v:6.3f}' for v in win(SP['constrictor'],'constrictor',key)))
# top-level grad norm and lr
def topwin(tr,key,k=16): 
    x=np.array([r[key] for r in tr]); return np.array([x[i:i+k].mean() for i in range(0,len(x),k)])
print('\nTop-level grad_norm (0.5 global clip) by 0.52M windows')
print('G        ',' '.join(f'{v:.3f}' for v in topwin(trG,'grad_norm')))
for m in MODES: print(f'S {m:11s}',' '.join(f'{v:.3f}' for v in topwin(SP[m],'grad_norm')))
print('lr first/last G:',trG[0]['lr'],trG[-1]['lr'],' S const:',SP['constrictor'][0]['lr'],SP['constrictor'][-1]['lr'])
print('total_updates G',trG[-1]['total_updates'],' S',SP['constrictor'][-1]['total_updates'])
# fraction of value-loss weight
print('\nShare of total value-loss weight (w_g/4) by 0.52M window, G:')
for m in MODES:
    print(f'{m:12s}',' '.join(f'{v/4:.3f}' for v in [np.mean(series(trG,m,"value_weight")[i:i+16]) for i in range(0,160,16)]))
# value-gradient magnitude proxy w*sqrt(2*v_loss), and relative to policy gradient
print('\nValue gradient proxy w*sqrt(2*v_loss) (G), and v-term w*v_loss:')
for m in MODES:
    vw=series(trG,m,'value_weight'); vl=series(trG,m,'v_loss')
    print(f'{m:12s} proxy',' '.join(f'{np.mean((vw*np.sqrt(2*vl))[i:i+16]):.3f}' for i in range(0,160,16)),'   w*v_loss',' '.join(f'{np.mean((vw*vl)[i:i+16]):.4f}' for i in range(0,160,16)))
# value weight vs 1/Var: constrictor weight share if only var mattered
