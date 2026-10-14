from common import *
runs={'seed0':'ppo-modes-duel-seed0','seed1':'ppo-modes-duel-seed1','seed2':'ppo-modes-duel-seed2'}
trs={k:load(v) for k,v in runs.items()}
for k,(tr,ev) in trs.items(): print(k,len(tr),'train',len(ev),'eval')
print('\nDuel specialist across seeds: mean over records 144:160 (last 0.52M) and 80:160')
for key in ['explained_var','entropy','approx_kl','clip_frac','return_var','mean_game_len','draw_rate','death_self','death_h2h','death_body','death_wall']:
    for lo,hi in [(144,160),(80,160)]:
        row=[]
        for k,(tr,ev) in trs.items():
            x=series(tr,'duel',key); g=series(tr,'duel','games')
            w=g if (key in('mean_game_len','draw_rate') or key.startswith('death_')) else np.ones_like(g)
            row.append(np.average(x[lo:hi],weights=w[lo:hi]))
        print(f'{key:14s} rec {lo}:{hi}  '+'  '.join(f'{v:.3f}' for v in row))
print('\nDuel specialist eval vs heuristic across seeds (score, w/d/l):')
for k,(tr,ev) in trs.items():
    e=[r for r in ev if r['opponent']=='heuristic']
    print(k,[ (round(r['env_steps']/1e6,2), round(r['score'],3), f"{r['wins']}/{r['draws']}/{r['losses']}") for r in e])
