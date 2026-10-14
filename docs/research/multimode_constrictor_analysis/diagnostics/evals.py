from common import *
trG,evG=load('ppo-modes-seed0')
SP={m:load(f'ppo-modes-{m}-seed0') for m in MODES}
M16={m:load(f'ppo-modes-{m}-m16-seed0') for m in ['standard','royale']}
def table(ev,mode,opp):
    return {r['env_steps']:r for r in ev if r['game_mode']==mode and r['opponent']==opp}
out={}
for opp in ['heuristic','random_legal']:
    print(f'\n=== vs {opp}: score (95% CI half-width; normal approx as logged), parity = 1/N; 256 games each')
    for m in MODES:
        g=table(evG,m,opp); s=table(SP[m][1],m,opp)
        steps=sorted(g)
        print(f'-- {m} (parity {1/ g[steps[0]]["num_snakes"]:.2f})   steps(M): '+'  '.join(f'{k/1e6:5.2f}' for k in steps))
        print('  G   '+'  '.join(f'{g[k]["score"]:.3f}±{g[k]["score_ci95"]:.3f}' for k in steps))
        print('  S   '+'  '.join(f'{s[k]["score"]:.3f}±{s[k]["score_ci95"]:.3f}' for k in steps))
        d=[(g[k]['score']-s[k]['score'], 1.96*np.sqrt((g[k]['score_ci95']/1.96)**2+(s[k]['score_ci95']/1.96)**2)) for k in steps]
        print('  G-S '+'  '.join(f'{a:+.3f}±{b:.3f}' for a,b in d))
        print('  G w/d/l '+'  '.join(f'{g[k]["wins"]}/{g[k]["draws"]}/{g[k]["losses"]}' for k in steps))
        print('  S w/d/l '+'  '.join(f'{s[k]["wins"]}/{s[k]["draws"]}/{s[k]["losses"]}' for k in steps))
        print('  G plc '+'  '.join(f'{g[k]["placement"]:.2f}' for k in steps)+'   S plc '+'  '.join(f'{s[k]["placement"]:.2f}' for k in steps))
        print('  G turns '+'  '.join(f'{g[k]["mean_turns"]:.1f}' for k in steps)+'   S turns '+'  '.join(f'{s[k]["mean_turns"]:.1f}' for k in steps))
print(set(r['env_steps'] for r in evG))
print(evG[0].keys())
