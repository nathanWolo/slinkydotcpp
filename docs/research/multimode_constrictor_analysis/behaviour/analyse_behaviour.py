"""Opening / move-level behaviour, policy disagreement, territory, for the constrictor G,S,G,S run.

Usage: python analyse_behaviour.py <npz> <out.json>
Decision states = (game, move t, living snake) with the contest still undecided at the start of t
(both policies have a snake alive). G and S are both evaluated on every such state (the move the
other policy would have made is the counterfactual). Intervals: cluster bootstrap over games.
"""
import json
import sys

sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from common import *
from geom import DX, H, W, voronoi, region_sizes, dilate

path, out_path = sys.argv[1], sys.argv[2]
d = contest_outcome(load(path))
Gm, T, N = d["G"], d["T"], d["N"]
pol = d["pol"]
boot = Boot(Gm, cluster=d.get("cluster"))
res = {"games": Gm}


def bp(x, p):
    """Per-game sum over the seats of policy p (and over the time axis)."""
    r = by_pol(x, pol, p)
    return r.sum(tuple(range(1, r.ndim)))

tt = np.arange(T)[None, :, None]
valid = d["alive"] & ~d["done"][:, :, None]
cv = valid & (np.arange(T)[None, :] < d["K_end"][:, None])[:, :, None]  # contested decision state
head = d["head"].astype(int)
act = d["act"].astype(int)
new = head + DX[act]
og = d["owner_grid"].astype(int)  # [G, T, H, W]
ar = np.arange(Gm)

# ------------------------------------------------------------------ per-state features
x, y = head[..., 0], head[..., 1]
c_l1 = np.abs(x - 5) + np.abs(y - 5)
c_l2 = np.hypot(x - 5, y - 5)
wall = np.minimum(np.minimum(x, 10 - x), np.minimum(y, 10 - y))
nx, ny = new[..., 0], new[..., 1]
inb = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
c_l1_new = np.abs(nx - 5) + np.abs(ny - 5)
toward_centre = c_l1_new < c_l1
away_centre = c_l1_new > c_l1
straight = act == d["last_move"].astype(int)
straight_ok = tt > 0


def nbr_owner(cellx, celly):
    """owner ids of the 4 neighbours of cells [G,T,N] -> [G,T,N,4] (-2 off board)."""
    out = np.full(cellx.shape + (4,), -2, int)
    for k in range(4):
        px, py = cellx + DX[k, 0], celly + DX[k, 1]
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        pxc, pyc = np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)
        v = og[ar[:, None, None], np.arange(T)[None, :, None], pyc, pxc]
        out[..., k] = np.where(ok, v, -2)
    return out, None


no_new, _ = nbr_owner(nx, ny)  # owners around the new head cell
me = np.arange(N)[None, None, :, None]
# neighbours of new head owned by me, excluding the cell I came from (old head)
nbx = nx[..., None] + DX[:, 0]
nby = ny[..., None] + DX[:, 1]
came_from = (nbx == x[..., None]) & (nby == y[..., None])
hug_own = ((no_new == me) & ~came_from).any(-1)
hug_opp = ((no_new >= 0) & (no_new != me)).any(-1)
on_wall_new = inb & ((nx == 0) | (nx == W - 1) | (ny == 0) | (ny == H - 1))
# L1 distance to nearest living opponent head, before / after
ohd = np.abs(head[:, :, :, None, :] - head[:, :, None, :, :]).sum(-1)  # [G,T,N,N] old
ohd_new = np.abs(new[:, :, :, None, :] - head[:, :, None, :, :]).sum(-1)
other_alive = d["alive"][:, :, None, :] & ~np.eye(N, dtype=bool)[None, None]
INF = 99
dmin_old = np.where(other_alive, ohd, INF).min(-1)
dmin_new = np.where(other_alive, ohd_new, INF).min(-1)
toward_opp = (dmin_new < dmin_old) & (dmin_old < INF)
# cell contested: another living snake has a legal move into the same cell
tgt = head[:, :, :, None, :] + DX[None, None, None, :, :]  # [G,T,N,4,2] cells reachable per snake
mask = d["mask"]
# me (i, a) vs other (j, b): same cell and other legal and alive
same = (tgt[:, :, :, :, None, None, :] == tgt[:, :, None, None, :, :, :]).all(-1)  # [G,T,N,4,N,4]
oth_ok = d["alive"][:, :, None, None, :, None] & mask[:, :, None, None, :, :] & ~np.eye(N, dtype=bool)[None, None, :, None, :, None]
contested_cell = (same & oth_ok).any((-1, -2))  # [G,T,N,4]
choose_contested = np.take_along_axis(contested_cell, act[..., None], -1)[..., 0]
legal = mask
has_alt_free = (legal & ~contested_cell & (np.arange(4)[None, None, None, :] != act[..., None])).any(-1)  # a legal, uncontested alternative exists
n_legal = legal.sum(-1)
trapped = ((n_legal == 4) & (tt > 0)) | (n_legal == 0)
forced = trapped | (n_legal == 1)

# ------------------------------------------------------------------ helpers
def series(feat, sel, label, ts, scope_text="", digits=3, pct=False, store=None):
    """Per policy mean of feat over selected snake-states for each t in ts (list of ranges), with G-S."""
    out = {}
    for name, tsel in ts:
        s = sel & tsel
        ng, ns = bp(feat * s, 0), bp(feat * s, 1)
        dg, ds = bp(s.astype(float), 0), bp(s.astype(float), 1)
        a, b, df = boot.ratio(ng, dg), boot.ratio(ns, ds), boot.diff(ng, dg, ns, ds)
        out[name] = {"G": a, "S": b, "diff": df, "nG": int(dg.sum()), "nS": int(ds.sum())}
    return out


def show(title, out, pct=False, digits=3):
    print(f"-- {title}")
    for name, v in out.items():
        print(f"   {name:10s} G {fmt(v['G'], digits, pct):>28s}  S {fmt(v['S'], digits, pct):>28s}  G-S {fmt(v['diff'], digits, pct):>26s} {'*' if sig(v['diff']) else ' '} (n={v['nG']},{v['nS']})")


# turn windows
per_turn = [(f"t={k}", (tt == k)) for k in range(10)]
open_win = [("t<10", tt < 10), ("t<5", tt < 5), ("t 5-9", (tt >= 5) & (tt < 10)), ("t>=10", tt >= 10)]
res["features"] = {}
print(f"Decision states used: {int(cv.sum())} (G {int(bp(cv.astype(float), 0).sum())}, S {int(bp(cv.astype(float), 1).sum())}); of which in t<10: {int((cv & (tt < 10)).sum())}")
feats = [
    ("head L1 distance to centre", c_l1.astype(float), cv, False, 2),
    ("head Euclid distance to centre", c_l2, cv, False, 2),
    ("head distance to nearest wall", wall.astype(float), cv, False, 2),
    ("move goes toward centre (L1)", toward_centre.astype(float), cv, True, 3),
    ("move goes away from centre (L1)", away_centre.astype(float), cv, True, 3),
    ("move keeps direction (t>=1)", straight.astype(float), cv & straight_ok, True, 3),
    ("new head touches own body (not cell just left)", hug_own.astype(float), cv & (tt >= 2), True, 3),
    ("new head touches an opponent body", hug_opp.astype(float), cv, True, 3),
    ("new head on the board edge", on_wall_new.astype(float), cv, True, 3),
    ("move toward nearest opp head (L1)", toward_opp.astype(float), cv, True, 3),
    ("min L1 distance to an opponent head", np.where(dmin_old < INF, dmin_old, 0).astype(float), cv & (dmin_old < INF), False, 2),
    ("chooses a cell another snake can also enter", choose_contested.astype(float), cv & ~trapped, True, 3),
    ("... among states with an uncontested alternative", choose_contested.astype(float), cv & ~trapped & has_alt_free, True, 3),
    ("legal moves available", n_legal.astype(float), cv, False, 2),
]
for title, f, sel, pct, dg in feats:
    out = series(f, sel, title, open_win, digits=dg)
    res["features"][title] = out
    show(title + "  [pooled windows]", out, pct, dg)

print("\n== per-turn series for the opening (t = move number, 0 = first decision)")
for title, f, sel, pct, dg in feats[:3] + feats[6:7] + feats[11:12]:
    out = series(f, sel, title, per_turn)
    res["features"][title + " (per turn)"] = out
    show(title, out, pct, dg)

# ------------------------------------------------------------------ disagreement
print("\n== disagreement of greedy moves on identical states")
ag, a_s = d["ag"].astype(int), d["a_s"].astype(int)
dis = (ag != a_s).astype(float)
res["disagree"] = {}
for lab, sel in (("all contested states", cv), ("states with >=2 legal moves", cv & (n_legal >= 2) & ~trapped), ("t<10, >=2 legal", cv & (n_legal >= 2) & (tt < 10)), ("t>=10, >=2 legal", cv & (n_legal >= 2) & ~trapped & (tt >= 10))):
    n_ = (dis * sel).sum((1, 2)); dn = sel.sum((1, 2)).astype(float)
    res["disagree"][lab] = boot.ratio(n_, dn)
    print(f"   {lab:32s} {fmt(res['disagree'][lab], pct=True)}   (n states={int(sel.sum())})")
print("   by who actually moves (state distribution produced by G's play vs S's play), >=2 legal moves:")
for p, nm in enumerate(POLICY):
    sel = cv & (n_legal >= 2) & ~trapped & (pol[:, None, :] == p)
    n_ = (dis * sel).sum((1, 2)); dn = sel.sum((1, 2)).astype(float)
    res["disagree"][f"seat of {nm}"] = boot.ratio(n_, dn)
    print(f"      snake played by {nm}: {fmt(res['disagree'][f'seat of {nm}'], pct=True)} (n={int(sel.sum())})")
sel = cv & (n_legal >= 2) & ~trapped
pg_ = bp(dis * sel, 0); dg_ = bp(sel.astype(float), 0); ps_ = bp(dis * sel, 1); ds_ = bp(sel.astype(float), 1)
res["disagree"]["G-seat minus S-seat"] = boot.diff(pg_, dg_, ps_, ds_)
print("      difference (G-played minus S-played):", fmt(res["disagree"]["G-seat minus S-seat"], 3), "*" if sig(res["disagree"]["G-seat minus S-seat"]) else "")
print("   by move number (>=2 legal moves):")
for k in list(range(0, 12)) + [12, 15, 20, 25, 30]:
    sk = sel & (tt == k)
    if sk.sum() < 10:
        continue
    ci = boot.ratio((dis * sk).sum((1, 2)), sk.sum((1, 2)).astype(float))
    res["disagree"][f"t={k}"] = ci
    print(f"      t={k:2d}: {fmt(ci, pct=True)}  (n={int(sk.sum())})")

# logits-based confidence of each policy on the same states
lg, ls = d["lg"].astype(np.float64), d["ls"].astype(np.float64)
def masked_softmax(l, m):
    z = np.where(m, l, -1e9)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z) * m
    return e / e.sum(-1, keepdims=True)
pgm, psm = masked_softmax(lg, d["mask"]), masked_softmax(ls, d["mask"])
ent = lambda p_: -(np.where(p_ > 0, p_ * np.log(np.maximum(p_, 1e-30)), 0)).sum(-1)
top_g, top_s = pgm.max(-1), psm.max(-1)
ent_g, ent_s = ent(pgm), ent(psm)
sel2 = cv & (n_legal >= 2) & ~trapped
res["confidence"] = {}
for lab, sel_ in (("t<10", sel2 & (tt < 10)), ("t>=10", sel2 & (tt >= 10)), ("all", sel2)):
    n_ = sel_.sum((1, 2)).astype(float)
    for nm, arr in (("top_prob_G", top_g), ("top_prob_S", top_s), ("entropy_G", ent_g), ("entropy_S", ent_s)):
        res["confidence"][f"{lab} {nm}"] = boot.ratio((arr * sel_).sum((1, 2)), n_)
    res["confidence"][f"{lab} top diff G-S"] = boot.diff((top_g * sel_).sum((1, 2)), n_, (top_s * sel_).sum((1, 2)), n_)
    res["confidence"][f"{lab} entropy diff G-S"] = boot.diff((ent_g * sel_).sum((1, 2)), n_, (ent_s * sel_).sum((1, 2)), n_)
    print(f"   [{lab}] mean top-move prob: G {fmt(res['confidence'][f'{lab} top_prob_G'])} S {fmt(res['confidence'][f'{lab} top_prob_S'])} diff {fmt(res['confidence'][f'{lab} top diff G-S'])};  entropy(nats): G {fmt(res['confidence'][f'{lab} entropy_G'])} S {fmt(res['confidence'][f'{lab} entropy_S'])} diff {fmt(res['confidence'][f'{lab} entropy diff G-S'])}")
# total-variation distance between G's and S's masked move distributions (0 = identical policies on that state)
tv = 0.5 * np.abs(pgm - psm).sum(-1)
res["tv"] = {}
for lab, sel_ in (("t<10", sel2 & (tt < 10)), ("t>=10", sel2 & (tt >= 10)), ("all", sel2)):
    n_ = sel_.sum((1, 2)).astype(float)
    res["tv"][lab] = boot.ratio((tv * sel_).sum((1, 2)), n_)
    print(f"   [{lab}] mean total-variation distance between the two masked move distributions (>=2 legal moves): {fmt(res['tv'][lab])}")
# cross-likelihood: probability each policy gives to the other's move at disagreements
dsel = sel2 & (ag != a_s)
pG_of_aS = np.take_along_axis(pgm, a_s[..., None], -1)[..., 0]
pS_of_aG = np.take_along_axis(psm, ag[..., None], -1)[..., 0]
pG_of_aG = top_g
pS_of_aS = top_s
n_ = dsel.sum((1, 2)).astype(float)
for nm, arr in (("P_G(G's own move)", pG_of_aG), ("P_G(S's move)", pG_of_aS), ("P_S(S's own move)", pS_of_aS), ("P_S(G's move)", pS_of_aG)):
    res["confidence"][f"disagree {nm}"] = boot.ratio((arr * dsel).sum((1, 2)), n_)
    print(f"   at disagreements, {nm}: {fmt(res['confidence'][f'disagree {nm}'])}")

# ------------------------------------------------------------------ territory (Voronoi) over the game
print("\n== Voronoi territory: cells closer to a snake's head than to any other's (ties neutral)")
sidx = np.argwhere(~d["done"] & (np.arange(T)[None, :] < d["K_end"][:, None]))  # contested (g, t) states
gs, ts_ = sidx[:, 0], sidx[:, 1]
free = (og[gs, ts_] == -1)
src = head[gs, ts_]
alv = d["alive"][gs, ts_]
terr = voronoi(free, src, alv)  # [S, N]
region = region_sizes(free, src, alv)
terr_full = np.zeros((Gm, T, N)); terr_full[gs, ts_] = terr
reg_full = np.zeros((Gm, T, N)); reg_full[gs, ts_] = region
tot = terr_full.sum(-1)
team_share = np.stack([by_pol(terr_full, pol, 0) / np.maximum(tot, 1), by_pol(terr_full, pol, 1) / np.maximum(tot, 1)], -1)  # [G,T,2]
both_alive = np.zeros((Gm, T), bool); both_alive[gs, ts_] = True
# only count states where both policies still have a snake and territory exists
res["territory"] = {}
print("   G team's share of all claimed cells at move t (parity 0.5), states where both teams alive and tot>0:")
for k in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]:
    s_ = both_alive[:, k] & (tot[:, k] > 0)
    if s_.sum() < 20:
        continue
    ci = boot.ratio(team_share[:, k, 0] * s_, s_.astype(float))
    res["territory"][f"G team share t={k}"] = ci
    print(f"   t={k:2d}: {fmt(ci)}  (games {int(s_.sum())}){' *' if (ci[1] > 0.5 or ci[2] < 0.5) else ''}")
# per snake absolute counts, opening
for lab, tsel in (("t<10", tt < 10), ("t 10-19", (tt >= 10) & (tt < 20)), ("t>=20", tt >= 20)):
    s_ = cv & tsel
    ng, ns = bp(terr_full * s_, 0), bp(terr_full * s_, 1)
    dg, ds = bp(s_.astype(float), 0), bp(s_.astype(float), 1)
    out = {"G": boot.ratio(ng, dg), "S": boot.ratio(ns, ds), "diff": boot.diff(ng, dg, ns, ds)}
    res["territory"][f"cells per snake {lab}"] = out
    print(f"   mean Voronoi cells per living snake, {lab:8s}: G {fmt(out['G'], 2)}  S {fmt(out['S'], 2)}  diff {fmt(out['diff'], 2)} {'*' if sig(out['diff']) else ''}")
    ng, ns = bp(reg_full * s_, 0), bp(reg_full * s_, 1)
    out = {"G": boot.ratio(ng, dg), "S": boot.ratio(ns, ds), "diff": boot.diff(ng, dg, ns, ds)}
    res["territory"][f"region per snake {lab}"] = out
    print(f"   mean reachable free region per living snake, {lab:8s}: G {fmt(out['G'], 2)}  S {fmt(out['S'], 2)}  diff {fmt(out['diff'], 2)} {'*' if sig(out['diff']) else ''}")

# ------------------------------------------------------------------ do G's and S's moves differ in territory when they disagree?
print("\n== at disagreement states: one-move lookahead (my Voronoi cells if I play the move; others at their current heads, delay-1 handicap for me)")
dsel_idx = np.argwhere(dsel)
gg, tt_, ii = dsel_idx[:, 0], dsel_idx[:, 1], dsel_idx[:, 2]


def terr_of_move(a_arr):
    free_ = (og[gg, tt_] == -1)
    # my new cell; others' current heads
    srcs = head[gg, tt_].copy()  # [S, N, 2]
    srcs[np.arange(len(gg)), ii] = head[gg, tt_, ii] + DX[a_arr]
    alive_ = d["alive"][gg, tt_]
    delay = np.zeros((len(gg), N), int); delay[np.arange(len(gg)), ii] = 1
    cnt = voronoi(free_, srcs, alive_, delay)
    return cnt[np.arange(len(gg)), ii], cnt.sum(1) - cnt[np.arange(len(gg)), ii]

tg, og_sum = terr_of_move(ag[gg, tt_, ii])
ts2, os_sum = terr_of_move(a_s[gg, tt_, ii])
actor_is_G = pol[gg, ii] == 0
# for each disagreement state the two candidate moves; value = my Voronoi cells
better_S = (ts2 > tg).astype(float); better_G = (tg > ts2).astype(float); equal = (tg == ts2).astype(float)
res["lookahead"] = {}
dstate_games = gg
def per_game(vals, mask_=None):
    out = np.zeros(Gm); cnt = np.zeros(Gm)
    m_ = np.ones(len(vals), bool) if mask_ is None else mask_
    np.add.at(out, gg[m_], vals[m_]); np.add.at(cnt, gg[m_], 1)
    return out, cnt
for lab, m_ in (("all disagreements", None), ("t<10", tt_ < 10), ("t>=10", tt_ >= 10)):
    nS, c_ = per_game(better_S, m_); nG, _ = per_game(better_G, m_); nE, _ = per_game(equal, m_)
    dterr, _ = per_game(ts2 - tg, m_)
    res["lookahead"][lab] = {"n": int(c_.sum()), "S_move_more_cells": boot.ratio(nS, c_), "G_move_more_cells": boot.ratio(nG, c_), "equal": boot.ratio(nE, c_), "mean_cells_S_minus_G": boot.ratio(dterr, c_)}
    r = res["lookahead"][lab]
    print(f"   {lab:18s} n={r['n']:5d}: S's move gives more cells {fmt(r['S_move_more_cells'], pct=True)}, G's move more {fmt(r['G_move_more_cells'], pct=True)}, equal {fmt(r['equal'], pct=True)}; mean cells (S move - G move) {fmt(r['mean_cells_S_minus_G'], 2)}")
# contested-cell choice at disagreements
cg = np.take_along_axis(contested_cell[gg, tt_, ii], ag[gg, tt_, ii][:, None], -1)[:, 0]
cs_ = np.take_along_axis(contested_cell[gg, tt_, ii], a_s[gg, tt_, ii][:, None], -1)[:, 0]
nG_, c_ = per_game(cg.astype(float)); nS_, _ = per_game(cs_.astype(float))
res["lookahead"]["contested_choice"] = {"G_move_contested": boot.ratio(nG_, c_), "S_move_contested": boot.ratio(nS_, c_), "diff": boot.diff(nG_, c_, nS_, c_)}
r = res["lookahead"]["contested_choice"]
print(f"   at disagreements, the move is a cell another snake can enter: G's {fmt(r['G_move_contested'], pct=True)}, S's {fmt(r['S_move_contested'], pct=True)}, diff {fmt(r['diff'], 3)} {'*' if sig(r['diff']) else ''}")
# G-played vs S-played snakes: who gets the better of it (acted move territory, using actual move)
print("   (the move actually played is the playing policy's own, so the comparison above is the counterfactual one)")

# ------------------------------------------------------------------ the 1 G vs 1 S endgame
print("\n== 1 G vs 1 S duels (first move at which exactly one snake of each policy is alive)")
seq = d["alive_seq"]
ng_alive = np.stack([(seq & (pol[:, None, :] == p)).sum(-1) for p in (0, 1)], -1)  # [G,T+1,2]
duel_k = np.full(Gm, -1)
for g in range(Gm):
    w = np.where((ng_alive[g, :, 0] == 1) & (ng_alive[g, :, 1] == 1))[0]
    if len(w):
        duel_k[g] = w[0]
has = duel_k >= 0
print(f"   games reaching a 1G-vs-1S state: {int(has.sum())} of {Gm}; G share there {fmt(boot.ratio(d['share'][:, 0] * has, has.astype(float)))}; draws {fmt(boot.ratio(d['draw'] * has, has.astype(float)), pct=True)}")
res["duel"] = {"n": int(has.sum()), "G_share": boot.ratio(d["share"][:, 0] * has, has.astype(float))}
# route into the duel and territory at the duel start
gi_ = np.where(has & (duel_k < T))[0]
k0 = duel_k[gi_]
free_ = (og[gi_, k0] == -1)
alv_ = d["alive"][gi_, k0]
terr_ = voronoi(free_, head[gi_, k0], alv_)
reg_ = region_sizes(free_, head[gi_, k0], alv_)
gi_seat = np.array([np.where(alv_[j] & (pol[g] == 0))[0][0] for j, g in enumerate(gi_)])
si_seat = np.array([np.where(alv_[j] & (pol[g] == 1))[0][0] for j, g in enumerate(gi_)])
tg_, ts_v = terr_[np.arange(len(gi_)), gi_seat], terr_[np.arange(len(gi_)), si_seat]
rg_, rs_ = reg_[np.arange(len(gi_)), gi_seat], reg_[np.arange(len(gi_)), si_seat]
adv = tg_ - ts_v
shr = d["share"][gi_, 0]
print("   at the duel start: mean Voronoi cells G", f"{tg_.mean():.2f}", "S", f"{ts_v.mean():.2f}", " (G - S mean", f"{(tg_-ts_v).mean():.2f})")
res["duel"]["terr_diff_mean"] = float((tg_ - ts_v).mean())
bins = [(-999, -3, "G has >=3 fewer cells"), (-2, 2, "within 2 cells"), (3, 999, "G has >=3 more cells")]
res["duel"]["by_territory"] = {}
for lo, hi, lab in bins:
    s_ = (adv >= lo) & (adv <= hi)
    num = np.zeros(Gm); den = np.zeros(Gm); np.add.at(num, gi_[s_], shr[s_]); np.add.at(den, gi_[s_], 1)
    ci = boot.ratio(num, den)
    res["duel"]["by_territory"][lab] = {"n": int(s_.sum()), "G_share": ci}
    print(f"   {lab:24s} n={int(s_.sum()):3d}  G share {fmt(ci)}")
# mean G share by sign of territory advantage with draws separated
print("   spearman(territory advantage, G share):", f"{np.corrcoef(np.argsort(np.argsort(adv)), np.argsort(np.argsort(shr)))[0,1]:.3f}")
# how the duel began: the preceding death events (head-on exchange vs trap)
tab = np.load(out_path.replace("behaviour.json", "core_constrictor_tables.npz"), allow_pickle=True) if False else None

json.dump(res, open(out_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
np.savez_compressed(out_path.replace(".json", "_arrays.npz"), terr=terr_full, reg=reg_full, cv=cv)
