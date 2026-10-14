"""Loading and per-snake / per-game tables from simulate.py's npz."""
from __future__ import annotations

import json
import numpy as np

CAUSES = {0: "none", 1: "out-of-health", 2: "wall", 3: "self", 4: "other-body", 5: "head-on", 6: "hazard"}
LINEUP = np.array([0, 1, 0, 1])
POLICY = ("G", "S")


def load(path, dedupe=None):
    """Load simulate.py's npz. dedupe=True keeps one game per distinct (start positions, seat parity)
    (constrictor games are deterministic given those, so repeats are copies); d['cluster'] is the
    placement id (both seat parities of a placement share it) and d['mult'] the multiplicity."""
    import os
    if dedupe is None:
        dedupe = os.environ.get("SLINKY_DEDUPE", "0") == "1"
    z = np.load(path) if isinstance(path, str) else path
    d = {k: z[k] for k in (z.files if hasattr(z, "files") else z.keys())}
    if dedupe:
        G0 = d["alive"].shape[0]
        start = d["head"][:, 0].reshape(G0, -1)
        par = (np.arange(G0) % 2)
        keys = [tuple(start[g].tolist()) + (int(par[g]),) for g in range(G0)]
        first, mult, seen = [], [], {}
        for g, k in enumerate(keys):
            if k not in seen:
                seen[k] = len(first)
                first.append(g)
                mult.append(1)
            else:
                mult[seen[k]] += 1
                # determinism check: a repeat must replay the same game
                f = first[seen[k]]
                assert (d["act"][g] == d["act"][f]).all() and (d["alive"][g] == d["alive"][f]).all(), "repeat differs"
        first = np.array(first)
        place_ids = {}
        cluster = np.array([place_ids.setdefault(tuple(start[g].tolist()), len(place_ids)) for g in first])
        d = {k: (v[first] if isinstance(v, np.ndarray) and v.shape[:1] == (G0,) else v) for k, v in d.items()}
        d["cluster"], d["mult"], d["n_raw"] = cluster, np.array(mult), G0
        # the parity of the kept game is that of its original id
        d["game_id"] = first
    else:
        d["game_id"] = np.arange(d["alive"].shape[0])
    G, T, N = d["alive"].shape
    d["G"], d["T"], d["N"] = G, T, N
    g = d["game_id"][:, None]
    s = np.arange(N)[None, :]
    d["pol"] = LINEUP[(s - g) % N]  # [G, N]: 0 = G, 1 = S, policy of seat s in game id g
    # alive sequence: alive_seq[g, k] = alive after k moves, k = 0..T
    seq = np.concatenate([d["alive"][:, :1], d["alive_after"]], axis=1)
    d["alive_seq"] = seq
    pol_alive = np.stack([(seq & (d["pol"][:, None, :] == p)).any(-1) for p in (0, 1)], axis=-1)  # [G,T+1,2]
    contested = pol_alive.all(-1)  # [G, T+1] both policies alive after k moves
    # contest ends at first k where not contested (always exists: game ends)
    d["K_end"] = np.argmax(~contested, axis=1)
    assert (~contested).any(1).all(), "game not finished within tmax"
    d["move_dead"] = d["final_elim_turn"] - 1  # 0-based move index of the fatal move; -1 if alive
    d["move_dead"] = np.where(d["final_alive"], 10**6, d["move_dead"])
    return d


def contest_outcome(d):
    """The benchmark's early-stop outcome: per-game share (G, S) in {0, 1/12, ...}, placement per policy."""
    G, N = d["G"], d["N"]
    K = d["K_end"]
    pol = d["pol"]
    elim = d["final_elim_turn"]
    alive_end = d["alive_seq"][np.arange(G), K]  # [G, N] alive when the contest ended
    share = np.zeros((G, 2))
    draw = np.zeros(G, bool)
    place = np.zeros((G, N))  # per-seat placement at contest end (benchmark definition)
    for g in range(G):
        a = alive_end[g]
        # eliminations that happened up to the contest end only
        e = np.where(elim[g] <= K[g], elim[g], 10**9)
        e = np.where(a, 10**9, e)  # alive at contest end = infinity
        if a.any():
            group = a
        else:
            last = e.max()
            group = e == last
            draw[g] = group.sum() >= 2
        for p in (0, 1):
            share[g, p] = (group & (pol[g] == p)).sum() / group.sum()
        outl = (e[None, :] > e[:, None]).sum(1)
        tied = (e[None, :] == e[:, None]).sum(1) - 1
        place[g] = 1 + outl + 0.5 * tied
    d["share"], d["draw"], d["place"], d["alive_end"] = share, draw, place, alive_end
    return d


def bench_pergame(mode, n):
    rows = [json.loads(l) for l in open("/home/user/slinkydotcpp/benchmarks/results/modes.jsonl")]
    for r in rows:
        if r["game_mode"] == mode and r["lineup"] == "G,S,G,S" and r["games"] >= n and r["seed"] == 0:
            pg = r["per_game"]
            w, sc = pg["width"], pg["share_scale"]
            def dec(t):
                return np.array([int(t[i : i + w], 36) for i in range(0, len(t), w)])
            return {x: dec(pg["share"][x])[:n] / sc for x in pg["share"]}, {x: dec(pg["place2"][x])[:n] / 4.0 for x in pg["place2"]}
    return None


# ---------------------------------------------------------------------------- bootstrap
class Boot:
    """Cluster bootstrap over games: resample games with replacement, recompute ratio-of-sums."""

    def __init__(self, G, B=4000, seed=12345, cluster=None):
        """cluster: optional int array [G] (e.g. the start placement); games of a cluster are resampled together."""
        rng = np.random.default_rng(seed)
        if cluster is None:
            self.M = None
            self.W = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)  # [B, G]
        else:
            C = int(cluster.max()) + 1
            self.M = np.zeros((C, G)); self.M[cluster, np.arange(G)] = 1.0
            self.W = rng.multinomial(C, np.full(C, 1.0 / C), size=B).astype(np.float64)  # [B, C]
        self.G, self.B = G, B

    def _agg(self, x):
        x = np.asarray(x, float)
        return x if self.M is None else np.tensordot(self.M, x, axes=(1, 0))

    def _ci(self, boots, pt):
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        return pt, lo, hi

    def ratio(self, num, den=None):
        """num, den: [G] or [G, K] per-game sums. Returns (point, lo, hi)."""
        num = np.asarray(num, float)
        den = np.ones_like(num) if den is None else np.asarray(den, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = num.sum(0) / den.sum(0)
            numc, denc = self._agg(num), self._agg(den)
            boots = (self.W @ numc) / (self.W @ denc)
        return self._ci(boots, pt)

    def diff(self, num_a, den_a, num_b, den_b):
        """ratio(a) - ratio(b) with the same resamples (paired by game)."""
        num_a, den_a, num_b, den_b = (np.asarray(x, float) for x in (num_a, den_a, num_b, den_b))
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = num_a.sum(0) / den_a.sum(0) - num_b.sum(0) / den_b.sum(0)
            na, da, nb, db = (self._agg(x) for x in (num_a, den_a, num_b, den_b))
            boots = (self.W @ na) / (self.W @ da) - (self.W @ nb) / (self.W @ db)
        return self._ci(boots, pt)


def fmt(ci, digits=3, pct=False):
    pt, lo, hi = ci
    if pct:
        return f"{100*pt:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
    return f"{pt:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def sig(ci):
    """True when the 95% interval excludes 0."""
    _, lo, hi = ci
    return bool((lo > 0) | (hi < 0))


def by_pol(x, pol, p):
    """Sum over the seats of policy p. x: [G, ..., N] with the seat axis last; pol: [G, N]."""
    sh = (pol.shape[0],) + (1,) * (x.ndim - 2) + (pol.shape[1],)
    return (x * (pol.reshape(sh) == p)).sum(-1)
