"""C1: correct uncertainty for the 12 distinct greedy constrictor games (4 W, 2 D, 6 L)."""
import numpy as np
from scipy import stats
x = np.array([1]*4 + [0.5]*2 + [0]*6, float)
rng = np.random.default_rng(0)
B = x[rng.integers(0, 12, (100000, 12))].mean(1)
print('mean', x.mean(), 'boot over 12 distinct games [%.3f, %.3f]' % tuple(np.percentile(B, [2.5, 97.5])))
# stratified by layout (corners: 1,1,.5,.5,0,0 ; cardinal: 1,1,0,0,0,0)
c = np.array([1,1,.5,.5,0,0.]); d = np.array([1,1,0,0,0,0.])
Bs = 0.5*(c[rng.integers(0,6,(100000,6))].mean(1) + d[rng.integers(0,6,(100000,6))].mean(1))
print('stratified boot [%.3f, %.3f]' % tuple(np.percentile(Bs, [2.5, 97.5])))
se = x.std(ddof=1)/np.sqrt(12); print('normal t-interval', x.mean() - stats.t.ppf(.975, 11)*se, x.mean() + stats.t.ppf(.975, 11)*se)
print('sign test 4 vs 6 two-sided p =', stats.binomtest(4, 10, 0.5).pvalue)
print('sign test 16 vs 24 (what was reported, 4x pseudo-replicated) p =', stats.binomtest(16, 40, 0.5).pvalue)
B96 = np.repeat(x, 8)[rng.integers(0, 96, (100000, 96))].mean(1)
print('boot over 96 configs (as reported) [%.3f, %.3f]' % tuple(np.percentile(B96, [2.5, 97.5])))
