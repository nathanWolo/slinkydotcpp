from common import *
trG,_=load('ppo-modes-seed0')
for run in ['ppo-modes-royale-m16-seed0','ppo-modes-standard-m16-seed0','ppo-modes-duel-m16-seed0']:
    tr,ev=load(run); print(run,len(tr),'train recs, last env_steps',tr[-1]['env_steps'] if tr else None,'evals',len(ev))
    cfg=json.load(open(R+run+'/config.json')); print('  minibatches',cfg['num_minibatches'])
# compare windows w5-w8 (records 80:144, 2.6-4.7M) for royale: G, S8, S16
S={m:load(f'ppo-modes-{m}-seed0')[0] for m in MODES}
M16={m:load(f'ppo-modes-{m}-m16-seed0')[0] for m in ['standard','royale']}
for mode in ['standard','royale']:
    n=len(M16[mode]); lo,hi=80,min(n,144)
    print(f'\n{mode}: records {lo}:{hi} (env steps {(lo+1)*32768/1e6:.2f}-{hi*32768/1e6:.2f}M)')
    for key in ['approx_kl','clip_frac','entropy','explained_var','return_var','mean_game_len','draw_rate','death_self','death_wall','death_body','death_h2h','death_starve','death_hazard']:
        vals=[]
        for lab,tr in [('G',trG),('S8',S[mode]),('S16',M16[mode])]:
            x=series(tr,mode,key); g=series(tr,mode,'games')
            wts=g if (key in('mean_game_len','draw_rate') or key.startswith('death_')) else np.ones_like(g)
            vals.append(np.average(x[lo:hi],weights=wts[lo:hi]))
        print(f'  {key:14s} G {vals[0]:.3f}  S8 {vals[1]:.3f}  S16 {vals[2]:.3f}')
