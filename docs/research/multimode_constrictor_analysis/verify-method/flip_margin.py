"""Is G's 5x higher flip rate under countdown-only changes explained by G simply having smaller logit margins
(less peaked policy) than S in constrictor? Compare flip rates within bins of the masked top-2 log-prob margin."""
import numpy as np
TP = 'docs/research/multimode_constrictor_analysis/tailprobe'
z = np.load(f'{TP}/con_gsgs_s.npz')
al, done, mask = z['alive'], z['done'], z['mask']
valid = al & ~done[:, :, None] & (mask.sum(-1) >= 2)
def stats(lab, l, lf, ln):
    m = np.where(mask, l, -np.inf); a = m.argmax(-1)
    srt = np.sort(m, -1); margin = srt[..., -1] - srt[..., -2]
    flips = {}
    for nm, x in (('far', lf), ('near', ln)):
        flips[nm] = np.where(mask, x, -np.inf).argmax(-1) != a
    return margin[valid], {k: v[valid] for k, v in flips.items()}
mg, fg = stats('G', z['lg'], z['lg_far'], z['lg_near'])
ms, fs = stats('S', z['ls'], z['ls_far'], z['ls_near'])
print('decisions with >=2 legal moves:', valid.sum())
print('overall flip far G %.3f S %.3f | near G %.3f S %.3f' % (fg['far'].mean(), fs['far'].mean(), fg['near'].mean(), fs['near'].mean()))
print('median margin (logit units) G %.2f S %.2f' % (np.median(mg), np.median(ms)))
edges = [0, 0.5, 1, 2, 3, 4, 6, 8, 1e9]
for lo, hi in zip(edges[:-1], edges[1:]):
    ig, is_ = (mg >= lo) & (mg < hi), (ms >= lo) & (ms < hi)
    print(f'margin [{lo},{hi}): n G {ig.sum():6d} S {is_.sum():6d} | far G {fg["far"][ig].mean():.3f} S {fs["far"][is_].mean():.3f} | near G {fg["near"][ig].mean():.3f} S {fs["near"][is_].mean():.3f}')
# margin-matched overall: reweight S's flip rates to G's margin distribution
bins = np.digitize(mg, edges) ; bins_s = np.digitize(ms, edges)
for nm in ('far', 'near'):
    rw = sum((bins == b).mean() * fs[nm][bins_s == b].mean() for b in np.unique(bins) if (bins_s == b).any())
    print(f'{nm}: G {fg[nm].mean():.3f}, S {fs[nm].mean():.3f}, S reweighted to G margin distribution {rw:.3f}')
