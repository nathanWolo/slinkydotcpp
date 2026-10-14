"""Turn a collected .npz of games into a table of decisions with per-move fill areas.

Usage: python decisions.py IN.npz OUT.npz

A decision is (game, turn, snake) with the snake alive, the game not over, and at least
two legal moves (env action mask, destination on the board; an all-True mask row after
turn 0 is the env's 'no safe move' convention and is skipped).
Per decision and move: area_con, area_std (frontier), area_stdw (wait), legal, plus the
greedy choices of G and S on the real and counterfactual inputs and the move played.
"""

import sys

import numpy as np

from fills import area_con, area_std

DELTAS = np.array([(0, 1), (0, -1), (-1, 0), (1, 0)])  # (dx, dy) UP DOWN LEFT RIGHT


def greedy(logits, legal):
    return np.argmax(np.where(legal, logits, -np.inf), axis=-1)


def main():
    src, out = sys.argv[1:3]
    d = np.load(src)
    body, head, alive, done, mask, turn = (d[k] for k in ("body", "head", "alive", "done", "mask", "turn"))
    G, T, N, H, W = body.shape
    gi, ti, si = np.nonzero(alive & ~done[:, :, None])
    hd = head[gi, ti, si]  # [K, 2] (x, y)
    dest = hd[:, None, :] + DELTAS[None]  # [K, 4, 2]
    onb = (dest[..., 0] >= 0) & (dest[..., 0] < W) & (dest[..., 1] >= 0) & (dest[..., 1] < H)
    m = mask[gi, ti, si]
    legal = m & onb
    trivial = m.all(axis=1) & (turn[gi, ti] > 0)
    keep = (legal.sum(axis=1) >= 2) & ~trivial
    gi, ti, si, dest, legal = gi[keep], ti[keep], si[keep], dest[keep], legal[keep]
    K = len(gi)
    print(f"{src}: {K} decisions")

    # countdown grid: max over living snakes
    cdall = np.where(alive[:, :, :, None, None], body, 0).max(axis=2).astype(np.int32)  # [G, T, H, W]

    a_con = np.zeros((K, 4), np.int32)
    a_std = np.zeros((K, 4), np.int32)
    a_stdw = np.zeros((K, 4), np.int32)
    kk, mm = np.nonzero(legal)
    chunk = 40000
    for b in range(0, len(kk), chunk):
        k, mv = kk[b : b + chunk], mm[b : b + chunk]
        cd = cdall[gi[k], ti[k]]
        seeds = np.zeros(cd.shape, bool)
        xy = dest[k, mv]
        seeds[np.arange(len(k)), xy[:, 1], xy[:, 0]] = True
        a_con[k, mv] = area_con(cd, seeds)
        a_std[k, mv] = area_std(cd, seeds, wait=False)
        a_stdw[k, mv] = area_std(cd, seeds, wait=True)
        print(f"  fills {min(b + chunk, len(kk))}/{len(kk)}", flush=True)

    # per-game signature (whole trajectory + seats) to find duplicated deterministic games
    import hashlib
    sig = np.array([int(hashlib.md5(body[g].tobytes() + d["seat_g"][g].tobytes()).hexdigest()[:15], 16)
                    for g in range(G)], np.int64)
    res = dict(
        game_sig=sig[gi],
        game=gi, turn=ti, snake=si, legal=legal, a_con=a_con, a_std=a_std, a_stdw=a_stdw,
        played=d["action"][gi, ti, si],
        mover_is_g=d["seat_g"][gi, si],
        length=d["length"][gi, ti, si],
        n_alive=alive[gi, ti].sum(axis=1),
        game_turn=turn[gi, ti],
        elim_turn=d["final_elim_turn"][gi, si],
        final_alive=d["final_alive"][gi, si],
    )
    # own tail cell (smallest positive countdown of the mover's own body)
    own = body[gi, ti, si].astype(np.int32)  # [K, H, W]
    ownpos = np.where(own > 0, own, 1 << 20)
    flat = ownpos.reshape(K, -1).argmin(axis=1)
    tail_xy = np.stack([flat % W, flat // W], axis=1)
    res["tail_adj"] = (np.abs(dest - tail_xy[:, None, :]).sum(axis=-1) == 1) & legal
    for key in ("lg", "ls", "lg_far", "ls_far", "lg_near", "ls_near"):
        res["c_" + key] = greedy(d[key][gi, ti, si], legal)
    np.savez_compressed(out, **res)


if __name__ == "__main__":
    main()
