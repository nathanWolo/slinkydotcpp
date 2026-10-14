"""Attribute deaths to avoidable trap choices, by who held the seat (played moves).

Usage: python deaths.py decisions.npz [dedup]

For every snake that died, look at its decisions in the last WINDOW turns before the
fatal move. A 'trap choice' is a played move with area_con < T while some legal move had
area_con >= 2T (T=8). It is 'masked' if area_std of the played move is >= 2T (the
pocket looks roomy under tail-moving semantics), else 'plain'.
Rates are per seat-game; CIs bootstrap games (B=2000).
"""

import sys

import numpy as np

T, WINDOW, B = 8, 12, 2000
rng = np.random.default_rng(7)

d = dict(np.load(sys.argv[1]))
if len(sys.argv) > 2 and sys.argv[2] == "dedup":
    _, first = np.unique(d["game_sig"], return_index=True)
    keep_games = np.unique(d["game"][first])
    sel = np.isin(d["game"], keep_games)
    d = {k: v[sel] for k, v in d.items()}

L, ac, ast, pl = d["legal"], d["a_con"], d["a_std"], d["played"]
a_pl = np.take_along_axis(ac, pl[:, None], 1)[:, 0]
s_pl = np.take_along_axis(ast, pl[:, None], 1)[:, 0]
safe = (L & (ac >= 2 * T)).any(1)
trap = (a_pl < T) & safe
masked = trap & (s_pl >= 2 * T)
plain = trap & ~masked

games = np.unique(d["game"])
gidx = {g: i for i, g in enumerate(games)}
# per game, per policy (0=S, 1=G): seats, deaths, deaths after masked trap, after plain trap
acc = np.zeros((len(games), 2, 4))
seen = set()
for g, s, is_g, died, et in zip(d["game"], d["snake"], d["mover_is_g"], ~d["final_alive"], d["elim_turn"]):
    if (g, s) in seen:
        continue
    seen.add((g, s))
    sel = (d["game"] == g) & (d["snake"] == s)
    row = acc[gidx[g], int(is_g)]
    row[0] += 1
    if died:
        row[1] += 1
        recent = sel & (d["game_turn"] >= et - WINDOW) & (d["game_turn"] < et)
        if masked[recent].any():
            row[2] += 1
        elif plain[recent].any():
            row[3] += 1

W = rng.multinomial(len(games), np.full(len(games), 1 / len(games)), size=B)


def show(name, num, den):
    est = num.sum() / den.sum()
    bs = (W @ num) / (W @ den)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{name}={est:.3f} [{lo:.3f},{hi:.3f}]"


print(f"{sys.argv[1]} games={len(games)}")
for p, name in ((1, "G"), (0, "S")):
    seats, deaths, dm, dp = (acc[:, p, i] for i in range(4))
    print(f" {name}: seats={int(seats.sum())} deaths={int(deaths.sum())}  "
          + show("deaths/seat", deaths, seats) + "  "
          + show("masked-trap deaths/seat", dm, seats) + "  "
          + show("plain-trap deaths/seat", dp, seats))
# paired difference G - S per seat
for i, name in ((1, "deaths"), (2, "masked"), (3, "plain")):
    est = acc[:, 1, i].sum() / acc[:, 1, 0].sum() - acc[:, 0, i].sum() / acc[:, 0, 0].sum()
    bs = (W @ acc[:, 1, i]) / (W @ acc[:, 1, 0]) - (W @ acc[:, 0, i]) / (W @ acc[:, 0, 0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f" G-S {name}/seat = {est:+.3f} [{lo:+.3f},{hi:+.3f}]")
