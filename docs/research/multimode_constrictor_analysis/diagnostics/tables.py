from common import *
import json
res=json.load(open('windows.json'))
def row(run,mode,key,p=3): return ' | '.join(f'{t[0]:.{p}f}' for t in res[f'{run}|{mode}|{key}'])
hdr='| quantity | '+' | '.join(f'{(w+1)*0.524288:.2f}' for w in range(10))+' |'
sep='|---|'+'---|'*10
for key in ['value_weight','explained_var','entropy','approx_kl','clip_frac','return_var','death_self','mean_game_len','draw_rate']:
    print(f'\n{key} (window end, M env steps per mode)'); print(hdr); print(sep)
    for m in MODES:
        for run in ['G','S']:
            if key=='value_weight' and run=='S': continue
            print(f'| {m} {run} | '+row(run,m,key,4 if key in('approx_kl','clip_frac') else 3)+' |')
