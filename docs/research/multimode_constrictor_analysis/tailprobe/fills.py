"""Flood fills for the tail probe (own numpy copy of the ideas in slinky/heuristic.py).

All functions work on batches: cd is int[K, H, W] (the countdown until each cell is
vacated, max over living snakes; 0 = free), seeds is bool[K, H, W] (one cell each, the
cell a move leads to). A seed is never tested for passability (as in heuristic.distances).

* area_con: static flood fill where every currently occupied cell (cd > 0) is a wall
  forever: the constrictor truth, ignoring future growth and deaths.
* area_std: time-aware fill (tail-moving semantics, ignoring eating). The seed is
  entered on move 1; a cell with countdown c can be entered on move d iff c <= d.
  wait=False (heuristic.arrival_times default): only the newest frontier expands, so a
  cell blocked when first reached is retried only from later frontier cells.
  wait=True: every reached cell keeps expanding (an optimistic bound: cells that open
  later than the frontier dies are still counted).
"""

import numpy as np


def neigh(m):
    p = np.pad(m, ((0, 0), (1, 1), (1, 1)))
    return p[:, :-2, 1:-1] | p[:, 2:, 1:-1] | p[:, 1:-1, :-2] | p[:, 1:-1, 2:]


def area_con(cd, seeds):
    free = cd == 0
    reached = seeds.copy()
    while True:
        new = neigh(reached) & free & ~reached
        if not new.any():
            break
        reached |= new
    return reached.sum(axis=(1, 2))


def area_std(cd, seeds, wait=False, max_depth=None):
    if max_depth is None:
        max_depth = int(cd.max()) + cd.shape[1] * cd.shape[2] + 2
    reached = seeds.copy()
    frontier = seeds.copy()
    d = 1
    maxcd = int(cd.max())
    while d < max_depth:
        d += 1
        grow = reached if wait else frontier
        new = neigh(grow) & ~reached & (cd <= d)
        reached |= new
        frontier = new
        if not wait and not frontier.any():
            break
        if wait and d > maxcd and not new.any():
            break
    return reached.sum(axis=(1, 2))


def _self_test():
    # 1x5 corridor: seed at x=0, a body cell with countdown 3 at x=2, free x=3,4.
    cd = np.zeros((1, 1, 5), np.int32)
    cd[0, 0, 2] = 3
    seeds = np.zeros((1, 1, 5), bool)
    seeds[0, 0, 0] = True
    assert area_con(cd, seeds)[0] == 2
    # frontier dies at move 2 (x=1 reached on move 2, x=2 opens on move 3 but frontier
    # x=1 expands on move 3: x=1 is the frontier from move 2 -> tries x=2 on move 3 ok).
    assert area_std(cd, seeds)[0] == 5, area_std(cd, seeds)
    cd[0, 0, 2] = 5
    assert area_std(cd, seeds)[0] == 2
    assert area_std(cd, seeds, wait=True)[0] == 5
    print("fills self-test ok")


if __name__ == "__main__":
    _self_test()
