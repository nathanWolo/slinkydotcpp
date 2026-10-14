"""Outcomes, survival, death causes, killers, head-on events and draws for a G,S,G,S run.

Usage: python analyse_core.py <mode> <npz> <out.json>
All intervals are 95% percentile intervals of a cluster bootstrap over games (4000 resamples),
so the pairing of G and S within a game is respected. 'Contested' = counted while snakes of both
policies were alive at the start of the move (the benchmark's early-stop rule).
"""
import json
import sys

sys.path.insert(0, "docs/research/multimode_constrictor_analysis/behaviour")
import numpy as np
from common import *

mode, path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
d = contest_outcome(load(path))
Gm, T, N = d["G"], d["T"], d["N"]
pol = d["pol"]
boot = Boot(Gm, cluster=d.get("cluster"))
res = {"mode": mode, "games": Gm}
ar = np.arange(Gm)

# --- outcomes
for p, nm in enumerate(POLICY):
    res[f"share_{nm}"] = boot.ratio(d["share"][:, p])
res["share_G_minus_parity"] = boot.ratio(d["share"][:, 0] - 0.5)
res["win_G"] = boot.ratio((d["share"][:, 0] == 1).astype(float))
res["win_S"] = boot.ratio((d["share"][:, 1] == 1).astype(float))
res["draw"] = boot.ratio(d["draw"].astype(float))
print(f"== {mode}: {Gm} games, contest ends after mean {d['K_end'].mean():.1f} moves; natural game length mean {d['final_turn'].mean():.1f}")
print("G share", fmt(res["share_G"]), " S share", fmt(res["share_S"]))
print("G wins", fmt(res["win_G"], pct=True), " S wins", fmt(res["win_S"], pct=True), " draws", fmt(res["draw"], pct=True))
pl = np.stack([(d["place"] * (pol == p)).sum(1) / 2 for p in (0, 1)], 1)
res["place_G"], res["place_S"] = boot.ratio(pl[:, 0]), boot.ratio(pl[:, 1])
res["place_diff"] = boot.diff(pl[:, 0], np.ones(Gm), pl[:, 1], np.ones(Gm))
print("mean placement G", fmt(res["place_G"]), "S", fmt(res["place_S"]), "diff(G-S)", fmt(res["place_diff"]))

# --- survival curves
Kmax = min(T, 60 if mode == "constrictor" else 250)
seq = d["alive_seq"]  # [G, T+1, N] alive after k moves
ks = np.arange(0, T + 1)
seq_c = d["alive_seq"][ar[:, None], np.minimum(ks[None, :], d["K_end"][:, None])]  # frozen at contest end
surv_full = {nm: np.stack([by_pol(seq[:, k], pol, p) for k in ks], 1) for p, nm in enumerate(POLICY)}  # [G, T+1] counts of alive seats
surv_con = {nm: np.stack([by_pol(seq_c[:, k], pol, p) for k in ks], 1) for p, nm in enumerate(POLICY)}
res["surv"] = {}
show = [5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40] if mode == "constrictor" else [25, 50, 75, 100, 125, 150, 175, 200]
for label, sv in (("full", surv_full), ("contested", surv_con)):
    res["surv"][label] = {}
    print(f"-- survival (fraction of seats alive after k moves), {label} (frozen at contest end)" if label == "contested" else f"-- survival, {label} game")
    rows = []
    for k in ks:
        if k >= T:
            break
        a = boot.ratio(sv["G"][:, k], np.full(Gm, 2.0)); b = boot.ratio(sv["S"][:, k], np.full(Gm, 2.0))
        df = boot.diff(sv["G"][:, k], np.full(Gm, 2.0), sv["S"][:, k], np.full(Gm, 2.0))
        res["surv"][label][int(k)] = {"G": a, "S": b, "diff": df}
        if k in show:
            print(f"  k={k:3d}  G {fmt(a)}  S {fmt(b)}  G-S {fmt(df)} {'*' if sig(df) else ''}")
    # restricted mean survival: area under curve over the whole horizon
    area_g = sv["G"][:, 1:T].sum(1) / 2.0; area_s = sv["S"][:, 1:T].sum(1) / 2.0
    res["surv"][label]["rmst"] = {"G": boot.ratio(area_g), "S": boot.ratio(area_s), "diff": boot.diff(area_g, np.ones(Gm), area_s, np.ones(Gm))}
    print("  area under curve (mean moves alive per seat): G", fmt(res["surv"][label]["rmst"]["G"]), "S", fmt(res["surv"][label]["rmst"]["S"]), "diff", fmt(res["surv"][label]["rmst"]["diff"]))
n_sig = sum(sig(res["surv"]["contested"][int(k)]["diff"]) for k in ks if k < T and k > 0)
print(f"  contested curve: G-S interval excludes 0 at {n_sig} of {T-1} moves")

# --- per-snake death table
dm = d["move_dead"]  # [G, N] fatal move index (10**6 if never)
dead = dm < 10**5
contested_death = dead & (dm < d["K_end"][:, None])
gi = ar[:, None].repeat(N, 1)
si = np.arange(N)[None, :].repeat(Gm, 0)
dmc = np.where(dead, dm, 0)
cause = np.where(dead, d["final_elim_cause"], 0)
kill = d["killers"][gi, dmc, si]  # [G, N(i), N(j)]
kill = kill & dead[:, :, None]
team_of = pol  # [G, N]
same_team = (pol[:, :, None] == pol[:, None, :]) & ~np.eye(N, dtype=bool)[None]
opp_team = pol[:, :, None] != pol[:, None, :]
# killer class
cls = np.empty((Gm, N), object)
cls[:] = "alive"
for c in range(1, 7):
    pass
k_mate = (kill & same_team).any(-1)
k_opp = (kill & opp_team).any(-1)
k_self = kill[:, np.arange(N), np.arange(N)] if False else np.stack([kill[:, i, i] for i in range(N)], 1)
for g in range(Gm):
    for i in range(N):
        if not dead[g, i]:
            continue
        c = cause[g, i]
        if c == 2:
            cls[g, i] = "wall"
        elif c == 1:
            cls[g, i] = "starved"
        elif c == 6:
            cls[g, i] = "hazard"
        elif c == 3:
            cls[g, i] = "self-body"
        elif c == 4:
            cls[g, i] = "body of teammate" if k_mate[g, i] and not k_opp[g, i] else ("body of opponent" if k_opp[g, i] and not k_mate[g, i] else "body ?")
        elif c == 5:
            cls[g, i] = ("head-on with teammate" if k_mate[g, i] and not k_opp[g, i] else
                         "head-on with opponent" if k_opp[g, i] and not k_mate[g, i] else
                         "head-on with both" if k_mate[g, i] and k_opp[g, i] else "head-on ?")
# validation: every collision death has an identified killer
unattrib = ((cause == 4) | (cause == 5)) & dead & ~(k_mate | k_opp)
print("deaths with unidentified killer:", int(unattrib.sum()), "of", int(dead.sum()))
res["unattributed_deaths"] = int(unattrib.sum())
# lengths at death
if mode == "constrictor":
    # equal-length check for head-ons: lengths at the fatal move of all snakes alive
    ln = d["length"][gi, dmc]  # length of snake i at start of fatal move
    mx = np.where(d["alive"][ar[:, None], dmc], d["length"][ar[:, None], dmc], -1)
    # any snake alive at that move with different length
    diff_len = 0
    for g in range(Gm):
        for t_ in range(d["T"]):
            if d["done"][g, t_]:
                break
            L = d["length"][g, t_][d["alive"][g, t_]]
            if L.min() != L.max():
                diff_len += 1
    res["states_with_unequal_lengths"] = diff_len
    print("decision states (game,move) with unequal lengths among living snakes:", diff_len)

cats = ["wall", "starved", "hazard", "self-body", "body of teammate", "body of opponent", "head-on with teammate", "head-on with opponent", "head-on with both"]
print(f"-- death causes per seat, contested deaths only (deaths while both policies alive); denominator = seats")
res["cause_contested"] = {}
res["cause_all"] = {}
for scope, mask_ in (("contested", contested_death), ("all", dead)):
    print(f"  [{scope}]  {'cause':26s} {'G':>26s} {'S':>26s} {'G-S':>26s}")
    for cat in cats + ["survived"]:
        if cat == "survived":
            ind = ~mask_ & True  # seat was not eliminated in scope
        else:
            ind = (cls == cat) & mask_
        ng, ns = by_pol(ind, pol, 0), by_pol(ind, pol, 1)
        a, b, df = boot.ratio(ng, np.full(Gm, 2.0)), boot.ratio(ns, np.full(Gm, 2.0)), boot.diff(ng, np.full(Gm, 2.0), ns, np.full(Gm, 2.0))
        res[f"cause_{scope}"][cat] = {"G": a, "S": b, "diff": df, "nG": int(ng.sum()), "nS": int(ns.sum())}
        if ng.sum() + ns.sum() > 0:
            print(f"        {cat:26s} {fmt(a, pct=True):>26s} {fmt(b, pct=True):>26s} {fmt(df, 3):>22s} {'*' if sig(df) else ' '}  (n={int(ng.sum())},{int(ns.sum())})")

# --- death moment and length at death (contested deaths)
dl = np.where(dead, d["final_length"], 0)
print("-- length at death (contested deaths), mean per dying seat")
for p, nm in enumerate(POLICY):
    pass
num_g = by_pol(dl * contested_death, pol, 0); den_g = by_pol(contested_death.astype(float), pol, 0)
num_s = by_pol(dl * contested_death, pol, 1); den_s = by_pol(contested_death.astype(float), pol, 1)
res["len_at_death"] = {"G": boot.ratio(num_g, den_g), "S": boot.ratio(num_s, den_s), "diff": boot.diff(num_g, den_g, num_s, den_s)}
print("  G", fmt(res["len_at_death"]["G"], 2), " S", fmt(res["len_at_death"]["S"], 2), " diff", fmt(res["len_at_death"]["diff"], 2))
num_g = by_pol((dm + 1) * contested_death, pol, 0); num_s = by_pol((dm + 1) * contested_death, pol, 1)
res["move_at_death"] = {"G": boot.ratio(num_g, den_g), "S": boot.ratio(num_s, den_s), "diff": boot.diff(num_g, den_g, num_s, den_s)}
print("-- mean death move number (contested deaths): G", fmt(res["move_at_death"]["G"], 2), "S", fmt(res["move_at_death"]["S"], 2), "diff", fmt(res["move_at_death"]["diff"], 2))
# first death in the game
first_move = np.where(dead, dm, 10**6).min(1)
first_is = (dm == first_move[:, None]) & dead
fg = by_pol(first_is.astype(float) , pol, 0) > 0
fs = by_pol(first_is.astype(float), pol, 1) > 0
res["first_death"] = {"G": boot.ratio(by_pol(first_is.astype(float), pol, 0) / first_is.sum(1)), "S": boot.ratio(by_pol(first_is.astype(float), pol, 1) / first_is.sum(1))}
print("-- share of first-death-of-the-game seats (ties split) : G", fmt(res["first_death"]["G"], 3), "S", fmt(res["first_death"]["S"], 3))
fm = first_move.astype(float)
res["first_death_move"] = boot.ratio(fm + 1)
print("   mean move of first death", fmt(res["first_death_move"], 2))

# --- forced (no safe move) vs chosen: fatal move legal-move count
nleg = d["mask"].sum(-1)  # [G, T, N]
trapped = (nleg == 4) & (np.arange(T)[None, :, None] > 0) | (nleg == 0)  # all-True mask after turn 0 = no safe move
n_legal_fatal = np.where(dead, nleg[gi, dmc, si], -1)
trapped_fatal = np.where(dead, trapped[gi, dmc, si], False)
res["trapped_at_nonh2h_death"] = {}
print("-- non-head-on deaths (wall/self/body) that were forced (mask all-True at t>0 = no safe move): ", end="")
nh = dead & (cause >= 1) & (cause != 5) & (cause != 1) & (cause != 6)
if nh.sum():
    print(f"{int((trapped_fatal & nh).sum())} of {int(nh.sum())}")
else:
    print("none")

# --- head-on events (contested moves only: move index < K_end)
print("-- head-on events: >=2 living snakes enter the same on-board cell on the same move")
ev = {}
H = 11
rows = []
for g in range(Gm):
    for t_ in range(min(T, int(d["K_end"][g]))):
        if d["done"][g, t_]:
            break
        al = d["alive"][g, t_]
        nh_ = d["newhead"][g, t_]
        onb = (nh_[:, 0] >= 0) & (nh_[:, 0] < H) & (nh_[:, 1] >= 0) & (nh_[:, 1] < H)
        seen = {}
        for i in range(N):
            if al[i] and onb[i]:
                seen.setdefault((nh_[i, 0], nh_[i, 1]), []).append(i)
        for cell, members in seen.items():
            if len(members) >= 2:
                comp = "".join(sorted(POLICY[pol[g, i]] for i in members))
                survivors = int(sum(d["alive_after"][g, t_, i] for i in members))
                rows.append((g, t_, comp, len(members), survivors, [int(dead[g, i] and cause[g, i] == 5) for i in members]))
comps = sorted({r[2] for r in rows})
res["headon"] = {}
for comp in comps:
    cnt = np.array([sum(1 for r in rows if r[0] == g and r[2] == comp) for g in range(Gm)], float)
    res["headon"][comp] = boot.ratio(cnt)
    print(f"  composition {comp:5s}: {int(cnt.sum()):4d} events, per game {fmt(res['headon'][comp], 3)}")
print("  events where some participant survived the collision:", sum(1 for r in rows if r[4] > 0), "of", len(rows))
cnt_gg = np.array([sum(1 for r in rows if r[0] == g and r[2] == "GG") for g in range(Gm)], float)
cnt_ss = np.array([sum(1 for r in rows if r[0] == g and r[2] == "SS") for g in range(Gm)], float)
res["headon_GG_minus_SS"] = boot.diff(cnt_gg, np.ones(Gm), cnt_ss, np.ones(Gm))
print("  GG minus SS events per game:", fmt(res["headon_GG_minus_SS"], 3), "*" if sig(res["headon_GG_minus_SS"]) else "")
# rate per seat: head-on deaths of each side (any composition)
for scope in ("contested",):
    ho = (cause == 5) & contested_death
    ng, ns = by_pol(ho.astype(float), pol, 0), by_pol(ho.astype(float), pol, 1)
    res["headon_death_rate"] = {"G": boot.ratio(ng, np.full(Gm, 2.0)), "S": boot.ratio(ns, np.full(Gm, 2.0)), "diff": boot.diff(ng, np.full(Gm, 2.0), ns, np.full(Gm, 2.0))}
    print("  seats dying in a head-on: G", fmt(res["headon_death_rate"]["G"], pct=True), "S", fmt(res["headon_death_rate"]["S"], pct=True), "diff", fmt(res["headon_death_rate"]["diff"], 3))

# head-to-head outcome when exactly one G and one S snake meet (composition 'GS'): who survives
print("-- head-on between one G and one S snake: who survives (length decides in the rules; ties kill both)")
ho_out = {"G only survives": 0, "S only survives": 0, "both die": 0}
lens = {"G longer": 0, "S longer": 0, "equal": 0}
gnum = {"G only survives": np.zeros(Gm), "S only survives": np.zeros(Gm), "both die": np.zeros(Gm)}
lnum = {"G longer": np.zeros(Gm), "S longer": np.zeros(Gm), "equal": np.zeros(Gm)}
for r in rows:
    g, t_, comp, nm_, surv, _ = r
    if comp != "GS" or nm_ != 2:
        continue
    cellh = None
    for cell_i in range(N):
        pass
    nh_ = d["newhead"][g, t_]
    al = d["alive"][g, t_]
    mem = [i for i in range(N) if al[i] and tuple(nh_[i]) == tuple(nh_[[j for j in range(N) if al[j] and pol[g, j] == 0 and tuple(nh_[j]) == tuple(nh_[i])][:1] + [i]][0]) ]
    gi_ = [i for i in range(N) if al[i] and pol[g, i] == 0 and any(al[j] and pol[g, j] == 1 and (nh_[j] == nh_[i]).all() for j in range(N))]
    si_ = [j for j in range(N) if al[j] and pol[g, j] == 1 and any(al[i] and pol[g, i] == 0 and (nh_[j] == nh_[i]).all() for i in range(N))]
    if len(gi_) != 1 or len(si_) != 1:
        continue
    ga, sa = d["alive_after"][g, t_, gi_[0]], d["alive_after"][g, t_, si_[0]]
    key_ = "G only survives" if ga and not sa else "S only survives" if sa and not ga else "both die" if not ga and not sa else None
    if key_ is None:
        continue
    ho_out[key_] += 1; gnum[key_][g] += 1
    lg_, ls_ = d["length"][g, t_, gi_[0]], d["length"][g, t_, si_[0]]
    lk = "G longer" if lg_ > ls_ else "S longer" if ls_ > lg_ else "equal"
    lens[lk] += 1; lnum[lk][g] += 1
res["headon_GS_outcome"] = {k: {"n": v, "per_game": boot.ratio(gnum[k])} for k, v in ho_out.items()}
res["headon_GS_lengths"] = {k: {"n": v, "per_game": boot.ratio(lnum[k])} for k, v in lens.items()}
tot_ = sum(ho_out.values())
print("   events (one G vs one S, two snakes only):", tot_, ho_out, " lengths:", lens)
if tot_:
    ge = np.array([gnum["G only survives"].sum(), gnum["S only survives"].sum()])
    nz = ge.sum()
    if nz:
        pg_ = boot.ratio(gnum["G only survives"], gnum["G only survives"] + gnum["S only survives"])
        res["headon_G_win_share_of_decisive"] = pg_
        print("   among head-ons with exactly one survivor, G is the survivor in", fmt(pg_, pct=True))
    for k in ho_out:
        print(f"   {k:16s} {fmt(boot.ratio(gnum[k], np.array([sum(gnum[kk][g] for kk in gnum) for g in range(Gm)])), pct=True) if tot_ else ''}")

# what a G-S head-on leaves behind: position after the collision (both die): count living G / S snakes after it
print("-- after a G-S head-on that kills both (composition 'GS', both die): who is left, and who wins")
left = {"G:S alive": [], }
outs = []
for r in rows:
    g, t_, comp, nm_, surv, _ = r
    if comp != "GS" or surv != 0:
        continue
    al_after = d["alive_after"][g, t_]
    ng_, ns_ = int((al_after & (pol[g] == 0)).sum()), int((al_after & (pol[g] == 1)).sum())
    outs.append((g, ng_, ns_, d["share"][g, 0]))
outs = np.array(outs)
res["GS_headon_aftermath"] = {}
for (a_, b_) in sorted({(int(o[1]), int(o[2])) for o in outs}):
    sel = [o for o in outs if int(o[1]) == a_ and int(o[2]) == b_]
    gs = np.array([o[0] for o in sel]); sh = np.array([o[3] for o in sel])
    # game-level cluster: weights by game
    num = np.zeros(Gm); den = np.zeros(Gm)
    for g_, s_ in zip(gs, sh):
        num[int(g_)] += s_; den[int(g_)] += 1
    ci = boot.ratio(num, den)
    res["GS_headon_aftermath"][f"{a_}G+{b_}S"] = {"n": len(sel), "G_share": ci}
    print(f"  left {a_} G + {b_} S alive: {len(sel):3d} events, G share of the game {fmt(ci, 3)}")

# --- how decisive games were lost: the last snake of the losing side
print("-- how the losing side's last snake went out (decisive games; head-on = it took a surviving partner's team-mate-not-involved with it)")
res["loss_mode"] = {}
for p, nm in enumerate(POLICY):
    lost = (d["share"][:, p] == 0)
    cats_l = {}
    for g in np.where(lost)[0]:
        # last eliminated snake(s) of policy p within contest
        mine = [i for i in range(N) if pol[g, i] == p]
        dmv = {i: dm[g, i] for i in mine}
        last_m = max(dmv.values())
        last_i = [i for i in mine if dmv[i] == last_m]
        c = cls[g, last_i[0]]
        cats_l.setdefault(c, []).append(g)
    res["loss_mode"][nm] = {}
    print(f"  {nm} lost {int(lost.sum())} games; last {nm} snake died by:")
    for c, gl in sorted(cats_l.items(), key=lambda kv: -len(kv[1])):
        num = np.zeros(Gm); num[gl] = 1
        ci = boot.ratio(num, lost.astype(float))
        res["loss_mode"][nm][c] = {"n": len(gl), "share_of_losses": ci, "per_game": boot.ratio(num)}
        print(f"      {c:26s} n={len(gl):3d}  share of its losses {fmt(ci, pct=True)}  per game {fmt(boot.ratio(num), pct=True)}")
cmp_cats = sorted({c for nm in POLICY for c in res["loss_mode"][nm]})
print("   difference per game, G-lost minus S-lost, by mode of loss:")
for c in cmp_cats:
    ng = np.zeros(Gm); ns = np.zeros(Gm)
    for g in range(Gm):
        pass
    # per game indicator that G lost this way, S lost this way
    pass
for c in cmp_cats:
    ng = np.zeros(Gm); ns = np.zeros(Gm)
    for nm, arr in (("G", ng), ("S", ns)):
        p = 0 if nm == "G" else 1
        for g in np.where(d["share"][:, p] == 0)[0]:
            mine = [i for i in range(N) if pol[g, i] == p]
            last_m = max(dm[g, i] for i in mine)
            last_i = [i for i in mine if dm[g, i] == last_m][0]
            if cls[g, last_i] == c:
                arr[g] = 1
    df = boot.diff(ng, np.ones(Gm), ns, np.ones(Gm))
    res["loss_mode"].setdefault("diff", {})[c] = df
    print(f"      {c:26s} G {int(ng.sum()):3d} S {int(ns.sum()):3d}  per-game diff {fmt(df, 3)} {'*' if sig(df) else ''}")

# --- draws
print("-- draws (contest ended with nobody alive and >= 2 snakes eliminated on the last move)")
dr_idx = np.where(d["draw"])[0]
res["draw_detail"] = {}
comp_cnt = {}
ho_draw = 0
for g in dr_idx:
    K = int(d["K_end"][g]); last = K  # elim_turn == K for last
    members = [i for i in range(N) if d["final_elim_turn"][g, i] == K]
    comp = "".join(sorted(POLICY[pol[g, i]] for i in members))
    cs = tuple(sorted(CAUSES[int(cause[g, i])] for i in members))
    comp_cnt.setdefault((comp, cs), 0)
    comp_cnt[(comp, cs)] += 1
for (comp, cs), n_ in sorted(comp_cnt.items(), key=lambda kv: -kv[1]):
    print(f"  {n_:3d} draws: final group {comp:5s} causes {cs}")
res["draw_detail"] = {f"{k[0]}|{','.join(k[1])}": v for k, v in comp_cnt.items()}
# draw as share of games by whether head-on was the cause
hd = np.array([any(cause[g, i] == 5 for i in range(N) if d["final_elim_turn"][g, i] == d["K_end"][g]) for g in dr_idx])
print(f"  draws involving a head-on: {int(hd.sum())} of {len(dr_idx)}")
dr_ho = np.zeros(Gm); dr_ho[dr_idx[hd]] = 1
dr_nho = np.zeros(Gm); dr_nho[dr_idx[~hd]] = 1
res["draw_by_headon"] = boot.ratio(dr_ho); res["draw_other"] = boot.ratio(dr_nho)
print("  per game: draws with head-on", fmt(res["draw_by_headon"], pct=True), "; draws without", fmt(res["draw_other"], pct=True))
# draw rate when first G-S head-on happens
# --- the decided-games view: what ends decisive games for each winner
print("-- decisive games: winner and how the winner's last opponent went out")
res["decisive"] = {}
for p, nm in enumerate(POLICY):
    won = d["share"][:, p] == 1
    cats_w = {}
    q = 1 - p
    for g in np.where(won)[0]:
        theirs = [i for i in range(N) if pol[g, i] == q]
        last_m = max(dm[g, i] for i in theirs)
        last_i = [i for i in theirs if dm[g, i] == last_m][0]
        cats_w.setdefault(cls[g, last_i], []).append(g)
    print(f"  {nm} won {int(won.sum())}; the loser's last snake died by: " + ", ".join(f"{c} {len(v)}" for c, v in sorted(cats_w.items(), key=lambda kv: -len(kv[1]))))
    res["decisive"][nm] = {c: len(v) for c, v in cats_w.items()}

json.dump(res, open(out_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
np.savez_compressed(out_path.replace(".json", "_tables.npz"), cls=cls.astype(str), cause=cause, dm=dm, dead=dead, contested_death=contested_death, K_end=d["K_end"], pol=pol, share=d["share"], draw=d["draw"])
