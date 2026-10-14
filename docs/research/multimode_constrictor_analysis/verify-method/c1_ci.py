"""Uncertainty for the greedy constrictor G-vs-S head-to-head using the 12 distinct games (orbits.py output)
instead of the 48 placements (which are replicates: each placement = mean of 2 complementary games, and only
6 distinct complementary pairs exist)."""
import numpy as np
from math import comb
# orbits.py output, order: corners x6, cardinal x6
games = {('c',((1,1),(1,9))):0.0, ('c',((1,1),(9,1))):1.0, ('c',((1,1),(9,9))):0.0, ('c',((1,9),(9,1))):0.5,
         ('c',((1,9),(9,9))):0.5, ('c',((9,1),(9,9))):1.0,
         ('x',((1,5),(5,1))):0.0, ('x',((1,5),(5,9))):1.0, ('x',((1,5),(9,5))):0.0, ('x',((5,1),(5,9))):0.0,
         ('x',((5,1),(9,5))):1.0, ('x',((5,9),(9,5))):0.0}
v = np.array(list(games.values()))
print('12 games mean', v.mean())
rng = np.random.default_rng(0)
bs = v[rng.integers(0, 12, (20000, 12))].mean(1)
print('bootstrap over 12 games 95%:', np.percentile(bs, [2.5, 97.5]))
w, l = int((v == 1).sum()), int((v == 0).sum()); n = w + l
p = min(1, 2 * sum(comb(n, k) for k in range(0, min(w, l) + 1)) / 2 ** n)
print(f'sign test over games {w} vs {l}: p={p:.3f}')
# complementary pairs: G holds A in one game, S holds A in the other (each policy plays both sides)
cells = {'c': [(1,1),(1,9),(9,1),(9,9)], 'x': [(1,5),(5,1),(5,9),(9,5)]}
pairs = []
for lay, cs in cells.items():
    seen = set()
    for (L, gs), s in games.items():
        if L != lay or gs in seen: continue
        comp = tuple(sorted(c for c in cs if c not in gs))
        seen |= {gs, comp}
        pairs.append(((lay, gs, comp), s, games[(lay, comp)]))
for p_ in pairs: print(p_)
pm = np.array([(a + b) / 2 for _, a, b in pairs])
print('pair means', pm, 'mean', pm.mean())
bs = pm[rng.integers(0, 6, (20000, 6))].mean(1)
print('bootstrap over 6 complementary pairs 95%:', np.percentile(bs, [2.5, 97.5]))
a, b = int((pm > .5).sum()), int((pm < .5).sum())
p = min(1, 2 * sum(comb(a + b, k) for k in range(0, min(a, b) + 1)) / 2 ** (a + b))
print(f'pairs: G ahead {a}, level {int((pm==.5).sum())}, behind {b}; sign test p={p:.3f}')
