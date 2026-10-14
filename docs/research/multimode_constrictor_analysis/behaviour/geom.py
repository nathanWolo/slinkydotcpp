"""Board geometry helpers for the behaviour analysis (numpy, batched)."""
import numpy as np

DX = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])  # (dx, dy) for UP, DOWN, LEFT, RIGHT
H = W = 11


def dilate(x):
    """4-neighbour dilation of bool [..., H, W]."""
    o = x.copy()
    o[..., 1:, :] |= x[..., :-1, :]
    o[..., :-1, :] |= x[..., 1:, :]
    o[..., :, 1:] |= x[..., :, :-1]
    o[..., :, :-1] |= x[..., :, 1:]
    return o


def onehot(xy, valid):
    """bool [B, K, H, W] one-hot of xy [B, K, 2] (x, y); all-False where invalid or off board."""
    B, K, _ = xy.shape
    out = np.zeros((B, K, H, W), bool)
    ok = valid & (xy[..., 0] >= 0) & (xy[..., 0] < W) & (xy[..., 1] >= 0) & (xy[..., 1] < H)
    b, k = np.nonzero(ok)
    out[b, k, xy[b, k, 1], xy[b, k, 0]] = True
    return out


def voronoi(free, src, alive, delay=None, iters=60):
    """Cells strictly closer to each source than to every other (BFS over free cells).

    free: bool [B, H, W] (enterable cells); src: int [B, K, 2] (x, y); alive: bool [B, K];
    delay: int [B, K] in {0, 1}; a delay-1 source (a snake that moves first) starts claiming at
    step 1 *at* its source cell (the cell it is moving into), a delay-0 source at step 1 claims
    its neighbours. Cells reached at the same step by two sources are neutral (and block).
    Returns counts [B, K].
    """
    B, K, _ = src.shape
    delay = np.zeros((B, K), int) if delay is None else delay
    oh = onehot(src, alive)
    frontier = oh & (delay == 0)[..., None, None]
    claimed = np.zeros((B, H, W), bool)
    counts = np.zeros((B, K), int)
    for step in range(1, iters + 1):
        cand = dilate(frontier)
        if step == 1:
            cand = cand | (oh & (delay == 1)[..., None, None])
        cand = cand & free[:, None] & ~claimed[:, None]
        n = cand.sum(1)  # [B, H, W]
        uniq = n == 1
        counts += (cand & uniq[:, None]).sum((2, 3))
        claimed |= n >= 1
        frontier = cand & uniq[:, None]
        if not frontier.any() and step > 1:
            break
    return counts


def region_sizes(free, head, alive):
    """Connected free region size reachable from each snake's head (ignoring other snakes' moves).

    free: bool [B, H, W]; head: int [B, K, 2]. Returns the number of free cells reachable from the
    head's neighbours, [B, K].
    """
    B, K, _ = head.shape
    reach = onehot(head, alive)
    f = free[:, None]
    for _ in range(H * W):
        nxt = (dilate(reach) & f) | reach
        if (nxt == reach).all():
            break
        reach = nxt
    reach &= f
    return reach.sum((2, 3))
