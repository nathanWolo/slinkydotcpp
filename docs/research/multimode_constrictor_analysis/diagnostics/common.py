import json, numpy as np
R='/home/user/slinkydotcpp/runs/'
MODES=['standard','duel','royale','constrictor']
def load(run):
    recs=[json.loads(l) for l in open(R+run+'/metrics.jsonl')]
    return [r for r in recs if r['type']=='train'],[r for r in recs if r['type']=='eval']
def series(tr, mode, key, game_modes=True):
    out=[]
    for r in tr:
        gm=r['game_modes'][mode]
        if key.startswith('death_'):
            out.append(gm['deaths'][key[6:]]/gm['games'])
        elif key=='draw_rate':
            out.append(gm['draws']/gm['games'])
        elif key=='deaths_total':
            out.append(sum(gm['deaths'].values())/gm['games'])
        else:
            out.append(gm[key])
    return np.array(out,float)
