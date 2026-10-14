"""Tests for the plateau watcher's stop rules (``benchmarks/plateau.py``): the trend rule, the
first (pairwise) rule, the stop decision and a watcher run on fake cells (no games are played).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


P = _load("benchmarks_plateau", "plateau.py")

M = 1_048_576  # the snapshots' spacing in env steps per mode

# DQN-M and PPO-M share against the heuristic (mean over the 4 modes) from the 3rd snapshot on
# (env steps 3 * M, 4 * M, ...), as the rule was designed on them.
DQN_M = [
    0.128, 0.143, 0.142, 0.145, 0.157, 0.152, 0.148, 0.186, 0.173, 0.158,
    0.175, 0.183, 0.167, 0.152, 0.150, 0.182, 0.174, 0.166, 0.146, 0.137,
    0.164, 0.151, 0.188, 0.126, 0.184, 0.161, 0.148, 0.139, 0.153,
]  # fmt: skip
PPO_M = [0.170, 0.204, 0.203, 0.215, 0.248, 0.245, 0.250, 0.256]
DQN_STOP = 13_631_488  # 13 * M
# Snapshots 1 and 2 (the watcher plays them against the heuristic too): the run is still climbing.
EARLY = [0.100, 0.118]


def points(shares: list[float], first: int = 3) -> list[tuple[int, float]]:
    """(env steps, share) of ``shares`` from snapshot ``first`` on."""
    return [((first + i) * M, s) for i, s in enumerate(shares)]


def verdicts(
    series: list[tuple[int, float]], window: int = 6, threshold: float = 0.02
) -> list[tuple[int, dict]]:
    """The trend check after each point of ``series``: (env steps, result)."""
    return [
        (series[i][0], P.trend_check(series[: i + 1], window, threshold))
        for i in range(len(series))
    ]


def stop_step(
    series: list[tuple[int, float]],
    window: int = 6,
    threshold: float = 0.02,
    confirm: int = 2,
    min_steps: int = 5_242_880,
) -> int | None:
    """The env steps at which the watcher would stop on ``series`` (None: it does not)."""
    flats = []
    for steps, r in verdicts(series, window, threshold):
        flats.append(r["flat"])
        if P.plateaued(flats, steps, confirm, min_steps):
            return steps
    return None


# --- The trend rule ---------------------------------------------------------------------


def test_dqn_m_stops_at_13_6m():
    # The flat checks up to the stop: 9.44M, then 12.58M and 13.63M.
    flat_steps = [s for s, r in verdicts(points(DQN_M)) if r["flat"] and s <= DQN_STOP]
    assert flat_steps == [9 * M, 12 * M, 13 * M]
    assert stop_step(points(DQN_M)) == DQN_STOP == 13_631_488
    # The first window is the 6th point (8.39M, with 3.15M..8.39M in it): not flat yet.
    first = P.trend_check(points(DQN_M)[:6], 6, 0.02)
    assert first["trend_gain"] == pytest.approx(0.0235714, abs=1e-6)
    assert first["trend_gain_ci95"] == pytest.approx(0.0169675, abs=1e-6)
    assert not first["flat"]


def test_dqn_m_with_the_first_two_snapshots_stops_at_the_same_step():
    series = points(EARLY, first=1) + points(DQN_M)
    flat_steps = [s for s, r in verdicts(series) if r["flat"] and s <= DQN_STOP]
    assert flat_steps == [9 * M, 12 * M, 13 * M]
    assert stop_step(series) == DQN_STOP


def test_ppo_m_never_flat():
    series = points(PPO_M)
    assert not any(r["flat"] for _, r in verdicts(series))
    assert stop_step(series) is None
    assert stop_step(points(EARLY, first=1) + series) is None


def test_t_table_is_the_scipy_quantile_and_decides_the_same(monkeypatch):
    scipy_stats = pytest.importorskip("scipy.stats")
    for dof in range(1, 301):
        assert P.t_table(dof) == pytest.approx(float(scipy_stats.t.ppf(0.975, dof)), abs=1e-3)
    assert P.t_crit(4) == pytest.approx(2.7764451, abs=1e-6)
    monkeypatch.setattr(P, "t_crit", P.t_table)
    assert stop_step(points(DQN_M)) == DQN_STOP
    assert stop_step(points(PPO_M)) is None


def test_t_table_values():
    assert P.t_table(1) == 12.7062
    assert P.t_table(4) == 2.7764
    assert P.t_table(30) == 2.0423
    assert P.t_table(60) == pytest.approx(2.0003, abs=1e-3)
    assert P.t_table(120) == pytest.approx(1.9799, abs=1e-3)
    with pytest.raises(ValueError):
        P.t_table(0)


def test_trend_matches_a_polyfit():
    series = points(DQN_M)[:12]
    window = 6
    r = P.trend_check(series, window, 0.02)
    last = series[-window:]
    x = np.array([s for s, _ in last]) / 1e6
    y = np.array([v for _, v in last])
    (slope, _), cov = np.polyfit(x, y, 1, cov=True)  # cov is scaled by SSR / (window - 2)
    span = x[-1] - x[0]
    assert span == pytest.approx(5 * 1.048576)
    assert r["trend_slope_per_M"] == pytest.approx(slope)
    assert r["trend_gain"] == pytest.approx(slope * span)
    assert r["trend_gain_ci95"] == pytest.approx(2.7764451 * math.sqrt(cov[0, 0]) * span, rel=1e-6)
    assert r["trend_points"] == window
    assert (r["rule"], r["window"]) == ("trend", window)


def test_only_the_last_window_points_count():
    series = points(DQN_M)[:15]
    assert P.trend_check(series, 6, 0.02) == P.trend_check(series[-6:], 6, 0.02)
    # A different window is a different fit.
    assert (
        P.trend_check(series, 8, 0.02)["trend_gain"] != P.trend_check(series, 6, 0.02)["trend_gain"]
    )
    assert P.trend_check(series, 8, 0.02)["trend_points"] == 8


def test_clean_linear_rise_is_never_flat():
    for step in (0.01, 0.002):  # +0.05 and +0.01 over a window (the threshold is 0.02)
        series = points([0.10 + step * i for i in range(14)])
        found = verdicts(series)
        assert not any(r["flat"] for _, r in found)
        assert found[-1][1]["trend_gain"] == pytest.approx(step * 5)
    assert stop_step(points([0.10 + 0.002 * i for i in range(14)])) is None


def test_flat_noisy_series_is_flat():
    noisy = [
        0.150, 0.168, 0.141, 0.159, 0.146, 0.163, 0.150, 0.138, 0.165, 0.147,
        0.158, 0.144, 0.160, 0.152, 0.140, 0.161, 0.149, 0.155,
    ]  # fmt: skip
    series = points(noisy, first=4)
    found = verdicts(series)
    assert all(r["flat"] for _, r in found[5:])
    assert all(abs(r["trend_gain"]) < 0.02 for _, r in found[5:])
    assert stop_step(series) == series[6][0]  # two flat windows, at 6 and 7 points


def test_significant_decline_is_flat():
    series = points([0.30 - 0.012 * i + (0.002 if i % 2 else -0.002) for i in range(10)])
    found = verdicts(series)
    for _, r in found[5:]:
        assert r["flat"]
        assert r["trend_gain"] + r["trend_gain_ci95"] < 0  # the decline is significant
    assert stop_step(series) == series[6][0]


def test_rise_below_the_threshold_flat_only_when_not_significant():
    # g = 0.01 < 0.02 with a clean line: significant, so rising ...
    clean = P.trend_check(points([0.10 + 0.002 * i for i in range(6)]), 6, 0.02)
    assert 0 < clean["trend_gain"] < 0.02 and not clean["flat"]
    # ... the same slope under noise that swamps it: not significant, so flat.
    noise = [0.012, -0.015, 0.009, -0.011, 0.014, -0.010]
    muddy = P.trend_check(points([0.10 + 0.002 * i + n for i, n in enumerate(noise)]), 6, 0.02)
    assert 0 < muddy["trend_gain"] < 0.02 and muddy["trend_gain"] < muddy["trend_gain_ci95"]
    assert muddy["flat"]


def test_gain_at_or_above_the_threshold_is_not_flat_even_if_insignificant():
    # DQN-M at 10.49M: g = 0.032 with a half-width of 0.040.
    r = P.trend_check(points(DQN_M)[:8], 6, 0.02)
    assert r["trend_gain"] >= 0.02 and r["trend_gain"] - r["trend_gain_ci95"] <= 0
    assert not r["flat"]
    assert P.trend_check(points(DQN_M)[:8], 6, 0.04)["flat"]  # a looser threshold


def test_fewer_than_window_points_is_not_flat_and_has_no_trend():
    for n in range(0, 6):
        r = P.trend_check(points(DQN_M)[:n], 6, 0.02)
        assert r == {
            "rule": "trend",
            "window": 6,
            "flat": False,
            "trend_gain": None,
            "trend_gain_ci95": None,
            "trend_slope_per_M": None,
            "trend_points": None,
        }
    assert P.trend_check(points(DQN_M)[:6], 6, 0.02)["trend_points"] == 6


def test_a_window_below_three_is_refused():
    with pytest.raises(ValueError):
        P.trend_check(points(DQN_M), 2, 0.02)
    with pytest.raises(SystemExit):
        P.parse_args(["--run", "r", "--kind", "dqn_modes", "--window", "2"])


def test_a_perfectly_flat_series_is_flat():
    r = P.trend_check(points([0.2] * 6), 6, 0.02)
    assert r["flat"] and r["trend_gain"] == pytest.approx(0.0, abs=1e-12)
    # Whatever the level: rounding in the fit must not make a constant series "rising".
    for level in (0.1, 0.15, 0.3, 1 / 3, 0.57):
        assert P.trend_check(points([level] * 6), 6, 0.02)["flat"]


def test_a_non_finite_share_counts_as_missing():
    nan = float("nan")
    series = points([0.10, 0.12, nan, 0.13, 0.12, 0.11])
    # 5 usable points are fewer than the window: not flat, no trend (and no NaN in the line).
    r = P.trend_check(series, 6, 0.02)
    assert r["trend_points"] is None and r["trend_gain"] is None and not r["flat"]
    # With one more point the window is the last 6 usable ones.
    longer = series + points([0.12], first=9)
    r = P.trend_check(longer, 6, 0.02)
    assert r == P.trend_check([p for p in longer if p[1] == p[1]], 6, 0.02)
    assert r["trend_points"] == 6
    json.dumps(r, allow_nan=False)
    inf = P.trend_check(points([0.1, 0.12, float("inf"), 0.13, 0.12, 0.11]), 6, 0.02)
    assert inf["trend_gain"] is None and not inf["flat"]


def test_no_line_through_a_single_snapshot():
    r = P.trend_check([(3 * M, 0.1)] * 6, 6, 0.02)  # the same env steps six times
    assert r["trend_gain"] is None and r["flat"] is False


# --- The stop decision ------------------------------------------------------------------


def test_plateaued_needs_confirm_flat_checks_and_min_steps():
    assert not P.plateaued([True], 6 * M, 2, 5 * M)  # one is not two
    assert not P.plateaued([False, True], 6 * M, 2, 5 * M)
    assert P.plateaued([False, True, True], 6 * M, 2, 5 * M)
    assert not P.plateaued([True, True, False], 6 * M, 2, 5 * M)
    assert not P.plateaued([True, True], 4 * M, 2, 5 * M)  # before the minimum steps
    assert P.plateaued([True, True], 5 * M, 2, 5 * M)
    assert P.plateaued([True, True, True], 9 * M, 3, 5 * M)
    assert not P.plateaued([], 9 * M, 2, 5 * M)


def test_verdict_counts_only_the_active_rule():
    legacy = {"env_steps": 6 * M, "flat": True}  # a line of the first rule: no "rule" field
    pair = {**legacy, "rule": "pair"}
    trend = {**legacy, "rule": "trend", "flat": True, "pair_flat": False}
    assert P.verdict(pair, "pair") and P.verdict(legacy, "pair")
    assert P.verdict(trend, "trend")
    assert not P.verdict(legacy, "trend") and not P.verdict(pair, "trend")
    # A trend line keeps the first rule's verdict in pair_flat.
    assert not P.verdict(trend, "pair")
    assert P.verdict({**trend, "pair_flat": True, "flat": False}, "pair")
    assert not P.verdict({**trend, "flat": False}, "trend")


def test_snapshots_lists_each_step_once(tmp_path):
    for name in ("params-1048576.npz", "params-01048576.npz", "params-2097152.npz", "params-x.npz"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "runner.npz").write_bytes(b"")
    assert P.snapshots(tmp_path) == [M, 2 * M]


def test_defaults():
    args = P.parse_args(["--run", "r", "--kind", "dqn_modes"])
    assert (args.rule, args.window, args.threshold, args.confirm) == ("trend", 6, 0.02, 2)
    assert args.min_steps == 5_242_880
    args = P.parse_args(["--run", "r", "--kind", "dqn_modes", "--rule", "pair", "--window", "4"])
    assert (args.rule, args.window) == ("pair", 4)
    with pytest.raises(SystemExit):
        P.parse_args(["--run", "r", "--kind", "dqn_modes", "--rule", "nope"])


# --- decide() on fake cells -------------------------------------------------------------

MODE_NAMES = ["m1", "m2"]
HALF_STEPS = 12  # the per-game shares are multiples of 1 / 12 (see x_record)


def x_record(share: float, per_game: list[int] | None = None, games: int = 12) -> dict:
    """A cell's result line as :func:`plateau.gain` reads it: the per-game shares of X in
    twelfths (by default around ``share``, an alternating pattern, so the games differ)."""
    if per_game is None:
        base = round(share * HALF_STEPS)
        per_game = [min(max(base + (1 if i % 2 else -1), 0), HALF_STEPS) for i in range(games)]
    zeros = np.zeros(len(per_game), np.int64)
    return {
        "results": {"X": {"share": share, "seats": 1}},
        "per_game": {
            "encoding": "base36",
            "width": 2,
            "share_scale": HALF_STEPS,
            "share": {"X": P.bench.base36(np.array(per_game), 2)},
            "place2": {"X": P.bench.base36(zeros, 2)},
        },
    }


def h2h_record(share: float = 0.5, ci: float = 0.04) -> dict:
    return {"results": {"A": {"share": share, "share_ci95": ci}}}


def make_args(argv: list[str] | None = None, **over) -> argparse.Namespace:
    args = P.parse_args(["--run", "r", "--kind", "dqn_modes", *(argv or [])])
    args.modes = MODE_NAMES
    for key, value in over.items():
        setattr(args, key, value)
    return args


def cells_for(shares: list[float], first: int = 1, h2h_share: float = 0.5):
    """steps, records and h2h of snapshots ``first``, ``first + 1``, ... with these shares."""
    steps = [(first + i) * M for i in range(len(shares))]
    records = {(x, m): x_record(s) for x, s in zip(steps, shares, strict=True) for m in MODE_NAMES}
    h2h = {(x, m): h2h_record(h2h_share) for x in steps for m in MODE_NAMES}
    return steps, records, h2h


OLD_KEYS = {
    "type", "env_steps", "against", "gain", "gain_ci95", "h2h_share", "flat",
    "share_vs_heuristic", "game_modes", "date",
}  # fmt: skip


def reference_pair(records, h2h, steps, k, args) -> tuple[bool, float, float, float]:
    """The first rule as it was before the trend rule: (flat, D, half, H2H)."""
    a, b = steps[k], steps[k - args.gap]
    gains, halves, shares = [], [], []
    for mode in args.modes:
        new = P.bench.decode_per_game(records[(a, mode)])["X"]["share"]
        old = P.bench.decode_per_game(records[(b, mode)])["X"]["share"]
        d = new - old
        gains.append(float(d.mean()))
        halves.append(P.bench.Z95 * float(d.std(ddof=1)) / math.sqrt(len(d)))
        shares.append(h2h[(a, mode)]["results"]["A"]["share"])
    d = float(np.mean(gains))
    half = math.sqrt(sum(x * x for x in halves)) / len(halves)
    h2h_share = float(np.mean(shares))
    lo, hi = args.h2h_band
    flat = d < args.threshold and d - half <= 0.0 <= d + half and lo <= h2h_share <= hi
    return flat, d, half, h2h_share


def pair_cells(ups: int, h2h_share: float):
    """4 snapshots over 48 games of 6/12 share: the last one is 1/12 better in ``ups`` games."""
    steps = [(i + 1) * M for i in range(4)]
    games = {x: [6] * 48 for x in steps}
    games[steps[3]] = [7] * ups + [6] * (48 - ups)
    records = {(x, m): x_record(0.5, games[x]) for x in steps for m in MODE_NAMES}
    return steps, records, {(x, m): h2h_record(h2h_share) for x in steps for m in MODE_NAMES}


@pytest.mark.parametrize(
    ("ups", "h2h_share", "expected"),
    [
        (1, 0.50, True),  # D = 0.002, its interval covers 0, parity
        (1, 0.60, False),  # no gain, but the head to head is lopsided
        (8, 0.50, False),  # D = 0.014 < 0.02 but significant
        (48, 0.50, False),  # D = 0.083 >= 0.02
    ],
)
def test_pair_rule_reproduces_the_first_decision(ups, h2h_share, expected):
    steps, records, h2h = pair_cells(ups, h2h_share)
    args = make_args(["--rule", "pair"])
    rec = P.decide(records, h2h, steps, 3, args)
    flat, d, half, share = reference_pair(records, h2h, steps, 3, args)
    assert flat is expected
    assert rec["flat"] is expected
    assert rec["gain"] == pytest.approx(d) and rec["gain"] == pytest.approx(ups / 12 / 48)
    assert rec["gain_ci95"] == pytest.approx(half)
    assert rec["h2h_share"] == pytest.approx(share)
    assert rec["env_steps"] == steps[3] and rec["against"] == steps[1]
    assert rec["share_vs_heuristic"] == pytest.approx(0.5)
    assert set(rec) == OLD_KEYS | {"rule"} and rec["rule"] == "pair"
    assert set(rec["game_modes"]) == set(MODE_NAMES)
    assert set(rec["game_modes"]["m1"]) == {
        "share_vs_heuristic", "gain", "gain_ci95", "h2h_share", "h2h_ci95",
    }  # fmt: skip
    # The same numbers, and the first rule's verdict next to the trend rule's.
    trend = P.decide(records, h2h, steps, 3, make_args())
    assert trend["pair_flat"] is expected and trend["gain"] == rec["gain"]


def test_pair_rule_ignores_the_trend():
    # A decline of 0.04 per snapshot: significant for the pairwise test (not flat by it),
    # while the trend rule calls a significant decline flat.
    shares = [0.30, 0.26, 0.22, 0.18, 0.14, 0.10, 0.06, 0.02]
    steps, records, h2h = cells_for(shares)
    pair = P.decide(records, h2h, steps, 7, make_args(["--rule", "pair"]))
    trend = P.decide(records, h2h, steps, 7, make_args())
    assert trend["pair_flat"] == pair["flat"] is False
    assert trend["flat"] is True
    assert not any(key.startswith("trend_") for key in pair)


def test_trend_rule_record():
    steps, records, h2h = cells_for(EARLY + DQN_M[:8])
    args = make_args()
    k = len(steps) - 1  # 10.49M: not flat
    rec = P.decide(records, h2h, steps, k, args)
    expected = P.trend_check(points(EARLY + DQN_M[:8], first=1), 6, 0.02)
    assert set(rec) == OLD_KEYS | {
        "pair_flat", "rule", "window", "trend_gain", "trend_gain_ci95", "trend_slope_per_M",
        "trend_points",
    }  # fmt: skip
    for key, value in expected.items():
        assert rec[key] == pytest.approx(value) if isinstance(value, float) else rec[key] == value
    assert rec["flat"] is False
    assert rec["trend_points"] == 6
    # The pairwise numbers are there as before.
    flat, d, half, share = reference_pair(records, h2h, steps, k, args)
    assert rec["gain"] == pytest.approx(d) and rec["pair_flat"] is flat
    json.dumps(rec)  # a JSON line


def test_trend_rule_before_window_points():
    steps, records, h2h = cells_for(EARLY + DQN_M[:3])  # 5 snapshots < 6
    rec = P.decide(records, h2h, steps, 4, make_args())
    assert rec["flat"] is False
    assert rec["rule"] == "trend" and rec["window"] == 6
    assert rec["trend_gain"] is rec["trend_gain_ci95"] is rec["trend_slope_per_M"] is None
    assert rec["trend_points"] is None
    assert "gain" in rec  # the pairwise numbers are still computed


def test_trend_uses_the_stored_shares_of_earlier_snapshots(tmp_path):
    """A restarted watcher has the earlier shares only in plateau.jsonl (decision lines of the
    first rule, with no trend fields): the trend must be computed from them."""
    shares = EARLY + DQN_M[:6]  # 8 snapshots; the window of the 8th is snapshots 3..8
    steps, records, h2h = cells_for(shares)
    args = make_args()
    full = P.decide(records, h2h, steps, 7, args)

    # The stored lines: decisions of the first rule for snapshots 3..7 (k >= gap).
    lines = []
    for k in range(2, 7):
        old = P.decide(records, h2h, steps, k, make_args(["--rule", "pair"]))
        del old["rule"]  # written by a version before the trend rule: no "rule" at all
        lines.append(json.dumps(old))
    stored = tmp_path / "plateau.jsonl"
    stored.write_text("\n".join(lines) + "\n")
    done = P.read_decisions(stored)
    assert sorted(done) == steps[2:7]
    assert all("trend_gain" not in rec for rec in done.values())

    # This process has played only snapshot 8 and its gap partner 6.
    fresh = {key: r for key, r in records.items() if key[0] in (steps[7], steps[5])}
    found = P.share_points(fresh, done, steps, 7, MODE_NAMES)
    assert [x for x, _ in found] == steps[2:]
    assert [s for _, s in found] == pytest.approx(shares[2:])
    resumed = P.decide(fresh, h2h, steps, 7, args, done)
    assert resumed["trend_points"] == 6
    for key in ("trend_gain", "trend_gain_ci95", "trend_slope_per_M"):
        assert resumed[key] == pytest.approx(full[key])
    assert resumed["flat"] == full["flat"]

    # Without the stored lines only 2 snapshots are known: no trend yet.
    alone = P.decide(fresh, h2h, steps, 7, args, {})
    assert alone["trend_points"] is None and alone["flat"] is False


def test_share_points_use_each_snapshot_once_and_skip_unusable_shares():
    steps, records, _ = cells_for([0.10, 0.20, 0.30, 0.40])
    straight = P.share_points(records, {}, steps, 3, MODE_NAMES)
    assert [s for _, s in straight] == pytest.approx([0.10, 0.20, 0.30, 0.40])
    dup = [steps[0], steps[1], steps[1], steps[2], steps[3]]  # a snapshot listed twice
    assert P.share_points(records, {}, dup, 4, MODE_NAMES) == straight

    records[(steps[1], "m1")]["results"]["X"]["share"] = float("nan")  # no usable share
    got = P.share_points(records, {}, steps, 3, MODE_NAMES)
    assert [x for x, _ in got] == [steps[0], steps[2], steps[3]]
    # The stored decision line stands in for it (if its own share is a number).
    stored = {steps[1]: {"share_vs_heuristic": 0.21}}
    got = P.share_points(records, stored, steps, 3, MODE_NAMES)
    assert [x for x, _ in got] == steps and got[1][1] == 0.21
    nan_stored = {steps[1]: {"share_vs_heuristic": float("nan")}}
    assert len(P.share_points(records, nan_stored, steps, 3, MODE_NAMES)) == 3

    del records[(steps[2], "m2")]["results"]["X"]  # a cell line without X
    got = P.share_points(records, {}, steps, 3, MODE_NAMES)
    assert [x for x, _ in got] == [steps[0], steps[3]]
    assert [
        x for x, _ in P.share_points({}, {steps[0]: {"share_vs_heuristic": None}}, steps, 0, ["m1"])
    ] == []


def test_a_cell_without_a_share_does_not_break_decide():
    steps, records, h2h = cells_for(EARLY + DQN_M[:4])
    records[(steps[1], "m1")]["results"]["X"]["share"] = float("nan")  # an earlier snapshot
    rec = P.decide(records, h2h, steps, 5, make_args())  # 6 snapshots, one unusable
    assert rec["trend_points"] is None and rec["flat"] is False
    json.dumps(rec, allow_nan=False)


def test_log_line():
    steps, records, h2h = cells_for(EARLY + DQN_M[:3])
    warm = P.log_line(P.decide(records, h2h, steps, 4, make_args()))
    assert "trend waiting for 6 points" in warm and warm.endswith("-> not flat")
    steps, records, h2h = cells_for(EARLY + DQN_M[:7])
    rec = P.decide(records, h2h, steps, 8, make_args())
    line = P.log_line(rec)
    assert f"{rec['env_steps']:,}: trend gain {rec['trend_gain']:+.3f} +/- " in line
    assert f"{rec['trend_gain_ci95']:.3f}" in line and "6 points" in line
    assert line.endswith("-> flat" if rec["flat"] else "-> rising")
    pair = P.log_line(P.decide(records, h2h, steps, 8, make_args(["--rule", "pair"])))
    assert pair.startswith(f"{rec['env_steps']:,}: gain ") and "trend" not in pair


# --- The watcher on fake cells ----------------------------------------------------------


class FakeRun:
    """Patches the watcher's snapshot listing and cell playing: the cells are made up from
    ``shares`` (by snapshot index), nothing is played. ``lost``: the steps whose strength
    cells the watcher no longer finds (as if plateau-cells.jsonl were gone)."""

    def __init__(self, monkeypatch, shares: list[float]):
        self.shares = shares
        self.available = len(shares)
        self.lost: set[int] = set()
        self.played = 0
        monkeypatch.setattr(P, "snapshots", self.snapshots)
        monkeypatch.setattr(P, "run_path", Path)
        monkeypatch.setattr(P, "strength_cell", lambda kind, run, x, mode, games: ("H", x))
        monkeypatch.setattr(P, "h2h_cell", lambda kind, run, a, b, mode, games: ("A", a))
        monkeypatch.setattr(P, "play_all", self.play_all)

    def snapshots(self, run):
        return [(i + 1) * M for i in range(self.available)]

    def play_all(self, cells, s, out, pool):
        done = {}
        for key, (cell, _) in cells.items():
            what, x = cell
            if what == "A":
                done[key] = h2h_record()
            elif x not in self.lost:
                done[key] = x_record(self.shares[x // M - 1])
            self.played += 1
        return done


def watch(run: Path, *argv: str) -> int:
    return P.watch(P.parse_args(["--run", str(run), "--kind", "dqn_modes", "--once", *argv]))


def lines_of(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "plateau.jsonl").read_text().splitlines() if x.strip()]


def test_watcher_stops_dqn_m_at_13_6m(tmp_path, monkeypatch):
    FakeRun(monkeypatch, EARLY + DQN_M)
    assert watch(tmp_path) == 0
    assert json.loads((tmp_path / "STOP").read_text())["plateau_at"] == DQN_STOP
    found = lines_of(tmp_path)
    assert [r["env_steps"] for r in found] == [x * M for x in range(3, 14)]  # gap .. stop
    assert [r["env_steps"] for r in found if r["flat"]] == [9 * M, 12 * M, 13 * M]
    assert all(r["rule"] == "trend" and r["window"] == 6 for r in found)
    warm = [r for r in found if r["env_steps"] < 6 * M]  # fewer than 6 snapshots played
    assert len(warm) == 3 and all(r["trend_gain"] is None and not r["flat"] for r in warm)
    assert all(r["trend_points"] == 6 for r in found[3:])


def test_watcher_never_stops_ppo_m(tmp_path, monkeypatch):
    FakeRun(monkeypatch, EARLY + PPO_M)
    assert watch(tmp_path) == 0
    assert not (tmp_path / "STOP").exists()
    assert len(lines_of(tmp_path)) == len(EARLY + PPO_M) - 2
    assert not any(r["flat"] for r in lines_of(tmp_path))


def test_watcher_pair_rule_still_selectable(tmp_path, monkeypatch):
    FakeRun(monkeypatch, [0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    assert watch(tmp_path, "--rule", "pair", "--min-steps", str(4 * M)) == 0
    assert json.loads((tmp_path / "STOP").read_text())["plateau_at"] == 4 * M
    found = lines_of(tmp_path)
    assert [r["rule"] for r in found] == ["pair", "pair"]  # 3rd and 4th snapshot
    assert all(r["flat"] for r in found)


def test_watcher_restart_reads_the_stored_shares(tmp_path, monkeypatch):
    fake = FakeRun(monkeypatch, EARLY + DQN_M)
    fake.available = 8  # the trainer has written 8 snapshots so far (8.39M)
    assert watch(tmp_path) == 0
    assert not (tmp_path / "STOP").exists()
    before = lines_of(tmp_path)
    assert [r["env_steps"] for r in before] == [x * M for x in range(3, 9)]

    # The machine restarts with the trainer further on; the watcher no longer finds the cells
    # of the first 6 snapshots (only plateau.jsonl is left of them; the 7th and 8th are still
    # needed as the partners of the 9th and 10th).
    fake.available = len(fake.shares)
    fake.lost = {x * M for x in range(1, 7)}
    assert watch(tmp_path) == 0
    assert json.loads((tmp_path / "STOP").read_text())["plateau_at"] == DQN_STOP
    after = lines_of(tmp_path)
    assert after[: len(before)] == before  # the stored decisions are not redone
    assert [r["env_steps"] for r in after] == [x * M for x in range(3, 14)]
    assert [r["env_steps"] for r in after if r["flat"]] == [9 * M, 12 * M, 13 * M]
    # snapshot 9 is decided from the stored shares of 4..6, the cells of 7 and 8 and its own
    assert after[6]["env_steps"] == 9 * M and after[6]["trend_points"] == 6

    # The same decisions as a watcher that never restarted.
    other = tmp_path / "other"
    other.mkdir()
    fake.lost = set()
    assert watch(other) == 0
    for resumed, straight in zip(after[6:], lines_of(other)[6:], strict=True):
        for key in ("trend_gain", "trend_gain_ci95", "trend_slope_per_M"):
            assert resumed[key] == pytest.approx(straight[key])
        assert resumed["flat"] == straight["flat"]


def test_watcher_restarted_on_first_rule_lines_needs_its_own_flat_checks(tmp_path, monkeypatch):
    """plateau.jsonl written by the first rule, every line flat: a watcher restarted with the
    trend rule must not stop on the strength of those (they say nothing about the trend)."""
    shares = EARLY + DQN_M
    steps, records, h2h = cells_for(shares)
    lines = []
    for k in range(2, 12):  # snapshots 3..12
        old = P.decide(records, h2h, steps, k, make_args(["--rule", "pair"]))
        del old["rule"]  # a version before the trend rule wrote no "rule"
        old["flat"] = True
        lines.append(json.dumps(old))
    (tmp_path / "plateau.jsonl").write_text("\n".join(lines) + "\n")

    FakeRun(monkeypatch, shares)
    assert watch(tmp_path) == 0
    # Snapshot 13 is trend-flat (the stop of a file with no old lines) but 12 is an old line;
    # snapshot 14 is the second flat trend check.
    assert json.loads((tmp_path / "STOP").read_text())["plateau_at"] == 14 * M
    found = lines_of(tmp_path)
    assert [r["env_steps"] for r in found] == [x * M for x in range(3, 15)]
    assert all("rule" not in r for r in found[:10])  # the stored lines are not redone
    assert [r["rule"] for r in found[10:]] == ["trend", "trend"]
    assert [r["flat"] for r in found[10:]] == [True, True]
    # The trend of snapshot 13 used the shares of 8..13, all from this process's cells.
    assert found[10]["trend_points"] == 6

    # The same file under --rule pair: the old lines count (flat at 4M and 5M, the first
    # snapshot at the minimum steps), and a trend line counts by its pair_flat.
    (tmp_path / "STOP").unlink()
    assert watch(tmp_path, "--rule", "pair") == 0
    assert json.loads((tmp_path / "STOP").read_text())["plateau_at"] == 5 * M


def test_watcher_rerun_after_stop_stops_again_without_new_decisions(tmp_path, monkeypatch):
    FakeRun(monkeypatch, EARLY + DQN_M)
    assert watch(tmp_path) == 0
    before = (tmp_path / "plateau.jsonl").read_text()
    (tmp_path / "STOP").unlink()
    assert watch(tmp_path) == 0
    assert json.loads((tmp_path / "STOP").read_text())["plateau_at"] == DQN_STOP
    assert (tmp_path / "plateau.jsonl").read_text() == before
