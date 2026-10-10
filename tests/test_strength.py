"""Tests for the strength benchmark (``benchmarks/strength.py``): fast, few or no games."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import jax
import pytest

from slinky import evaluate
from slinky.types import GameConfig

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))  # also seen by --workers' spawned processes
import strength as S  # noqa: E402

DUEL = GameConfig()


def settings(**kw) -> S.Settings:
    base = dict(
        game=DUEL, max_turns=500, seed=0, target_ci=None, slots=32, round_games=64, mem_mb=1024.0
    )
    return S.Settings(**{**base, **kw})


def matchup(a: str = "mcts-16", b: str = "heuristic", games: int = 256) -> S.Matchup:
    return S.Matchup(S.load_agent(a), S.load_agent(b), games)


def fake_record(m: S.Matchup, s: S.Settings, wins: int, draws: int, losses: int, **kw) -> dict:
    """A result line as ``run_matchup`` writes it, without playing."""
    n = wins + draws + losses
    _, score, ci = S.outcome_stats(wins, draws, losses)
    record = {
        "schema": S.SCHEMA, "matchup_id": S.matchup_id(m, s), "config_id": S.config_id(m, s),
        "a": m.a.name, "b": m.b.name, "a_config": m.a.config, "b_config": m.b.config,
        "game": S.game_dict(s.game), "max_turns": s.max_turns, "games_requested": m.games,
        "games": n, "wins": wins, "draws": draws, "losses": losses, "truncated": 0,
        "score": score, "ci95": ci, "mean_turns": 100.0, "wall_seconds": 1.0,
        "compile_seconds": 0.5, "play_seconds": 0.5, "seconds_per_game_turn": 1e-4,
        "slot_utilization": 0.9, "peak_rss_mb": 1.0, "seed": s.seed,
        "git_commit": "abc", "git_dirty": False, "code_fingerprint": "f" * 16,
        "stopped_early": False, "date": "2026-10-10T00:00:00+00:00",
    }  # fmt: skip
    return {**record, **kw}


# --- Agents and identity -------------------------------------------------------------


def test_agent_names_come_from_the_registry():
    agents = S.parse_agents("mcts-64-rollout, mcts-64:rollout_steps=10, heuristic, random")
    assert [a.name for a in agents] == ["mcts-64-rollout10", "heuristic", "random"]
    assert agents[0].config["rollout_steps"] == 10 and agents[0].config["type"] == "mcts"
    assert agents[1].config["weights"]["territory"] > 0
    # Table filters only parse names: no checkpoint is read.
    assert S.names_of("dqn:/nonexistent/run,mcts-8-rm") == {"dqn:/nonexistent/run", "mcts-8-rm"}
    assert S.names_of(None) is None
    with pytest.raises(S.SpecError, match="has no effect"):
        S.parse_agents("mcts-64-gamma0.3")
    with pytest.raises(S.SpecError, match="dqn:<run dir>"):
        S.parse_agents("dqn@dqn-duel-seed0")
    with pytest.raises(S.SpecError, match="no DQN checkpoint"):
        S.parse_agents("dqn:/nonexistent/run")


def test_identity_hashing():
    m, s = matchup(), settings()
    mid, cid = S.matchup_id(m, s), S.config_id(m, s)
    # How the games are scheduled doesn't change which games are played.
    for same in (settings(slots=8), settings(mem_mb=10.0), settings(round_games=16)):
        assert S.matchup_id(m, same) == mid
    # What is played does.
    others = [
        S.matchup_id(m, settings(seed=1)),
        S.matchup_id(m, settings(max_turns=400)),
        S.matchup_id(m, settings(target_ci=0.03)),
        S.matchup_id(m, settings(game=GameConfig(num_snakes=3))),
        S.matchup_id(matchup(games=128), s),
        S.matchup_id(matchup("mcts-16-rm"), s),
        S.matchup_id(matchup(b="random_legal"), s),
        S.matchup_id(matchup("heuristic", "mcts-16"), s),
    ]
    assert len({mid, *others}) == len(others) + 1
    # With --target-ci the round size sets the stopping points.
    t = settings(target_ci=0.03)
    assert S.matchup_id(m, t) != S.matchup_id(m, dataclasses.replace(t, round_games=32))
    # The config id ignores how many games and which seed (lines of one config pool).
    assert S.config_id(matchup(games=64), settings(seed=5)) == cid
    # Keys depend on the seed and the names only (pinned: results must reproduce).
    key = S.matchup_key(0, "mcts-64", "heuristic")
    assert jax.random.key_data(key).tolist() == [3246473066, 3519825672]
    assert jax.random.key_data(S.matchup_key(3, "a", "b")).tolist() == [1647082670, 2037490400]


def test_provenance_is_captured_once():
    first = S.provenance()
    assert S.provenance() is first
    assert len(first.code_fingerprint) == 16 and first.code_files > 5
    fingerprint, files = S.code_fingerprint()
    assert files >= first.code_files  # more modules may have been imported since


# --- Planning ------------------------------------------------------------------------


def test_plan_slots_and_rounds():
    for games in (1, 7, 16, 100, 256, 1000):
        for snakes in (2, 3, 4):
            game = GameConfig(num_snakes=snakes)
            for target_ci in (None, 0.03):
                s = settings(game=game, target_ci=target_ci, round_games=60)
                plan = S.plan_matchup(matchup(b="random_legal", games=games), s)
                assert plan.games == -(-games // snakes) * snakes  # whole seat rotations
                assert all(r % snakes == 0 for r in plan.rounds[:-1]) and plan.rounds[-1] > 0
                assert 1 <= plan.slots <= min(32, min(plan.rounds))
                if target_ci is None:
                    assert plan.rounds == (plan.games,)
                else:
                    assert max(plan.rounds) <= max(60, snakes) + snakes
    plan = S.plan_matchup(matchup(games=256), settings(target_ci=0.03))
    assert plan.rounds == (64, 64, 64, 64) and plan.slots == 32 and not plan.memory_limited
    # Big trees: the memory budget cuts the slots (1 KB per node and simulation).
    big = S.plan_matchup(matchup("mcts-50000"), settings())
    assert big.memory_limited and big.slots < 32 and big.tree_mb <= 1024 / S.TREE_SAFETY
    assert not S.plan_matchup(matchup("mcts-50000"), settings(mem_mb=1e6)).memory_limited
    assert S.tree_bytes_per_game(S.load_agent("heuristic"), DUEL) == 0
    rm = S.tree_bytes_per_game(S.load_agent("mcts-100-rm"), DUEL)
    assert 1.0e3 < S.tree_bytes_per_game(S.load_agent("mcts-100"), DUEL) / 101 < 1.1e3 < rm / 101


def test_rough_costs_and_schedule():
    cheap, dear = matchup("mcts-16", "random_legal"), matchup("mcts-1024", "dqn")
    assert cheap.rough_seconds() < matchup("mcts-16").rough_seconds() < dear.rough_seconds()
    assert S.lpt_makespan([4, 3, 3, 2, 2, 2], 2) == 8
    assert S.lpt_makespan([5, 1], 4) == 5


# --- Statistics ----------------------------------------------------------------------


def test_outcome_stats_match_evaluate():
    for w, d, lo in ((60, 20, 20), (0, 0, 7), (3, 1, 0), (100, 0, 0), (1, 1, 1)):
        n, score, ci = S.outcome_stats(w, d, lo)
        ref = evaluate._summarize(w, d, lo, 0, n)
        assert (n, score) == (ref.num_games, ref.score) and ci == pytest.approx(ref.score_ci95)


def test_score_interval():
    # Wilson's closed forms at the boundary: a shutout is not "exact".
    score, low, high = S.score_interval(256, 0, 0)
    assert score == 1.0 and high == 1.0 and low == pytest.approx(256 / (256 + S.Z95**2))
    assert S.score_interval(0, 0, 256)[2] == pytest.approx(S.Z95**2 / (256 + S.Z95**2))
    # Close to the normal interval in the middle.
    score, low, high = S.score_interval(120, 30, 106)
    assert (high - low) / 2 == pytest.approx(S.outcome_stats(120, 30, 106)[2], rel=0.1)
    # Monotonic: turning a loss into a draw or a draw into a win moves both ends up,
    # and the interval always has width and contains the score.
    for n in (1, 2, 5, 16, 40):
        for w in range(n + 1):
            for d in range(n + 1 - w):
                lo_ = n - w - d
                score, low, high = S.score_interval(w, d, lo_)
                assert 0 <= low < score + 1e-12 and score - 1e-12 < high <= 1 and high > low
                if lo_ > 0:
                    up = S.score_interval(w, d + 1, lo_ - 1)
                    assert up[1] > low and up[2] > high - 1e-12
                if d > 0:
                    up = S.score_interval(w + 1, d - 1, lo_)
                    assert up[1] > low and up[2] > high - 1e-12
    assert S.interval_text(3, 1, 0) == "0.875 [0.396, 0.987]"


# --- The results file and resume -----------------------------------------------------


def test_torn_lines_and_old_schema(tmp_path):
    out = tmp_path / "r.jsonl"
    s = settings()
    first, second = matchup(), matchup(b="random_legal")
    S.append_record(out, fake_record(first, s, 1, 2, 3))
    with open(out, "a") as f:
        f.write('{"schema": 2, "a": "torn')  # a writer killed mid-line
    S.append_record(out, fake_record(second, s, 3, 2, 1))
    with open(out, "a") as f:
        f.write(json.dumps({"schema": 1, "a": "x"}) + "\n")
    records, bad, old = S.read_records(out)
    assert [r["b"] for r in records] == ["heuristic", "random_legal"] and (bad, old) == (1, 1)
    assert out.read_text().count("\n") == 4  # the torn line was terminated before appending
    assert S.done_ids(out) == {S.matchup_id(first, s), S.matchup_id(second, s)}


def test_sweep_resumes_reruns_and_survives_failures(tmp_path, monkeypatch, capsys):
    out, s = tmp_path / "r.jsonl", settings(max_turns=50)
    played = []

    def fake_run(m, s_, plan, tag):
        played.append((m.a.name, m.b.name))
        if m.b.name == "random":
            raise RuntimeError("boom")
        return fake_record(m, s_, 2, 0, 0, date=f"2026-10-10T00:00:{len(played):02d}+00:00")

    monkeypatch.setattr(S, "run_matchup", fake_run)
    ms = [matchup(b="heuristic", games=4), matchup(b="random_legal", games=4)]
    assert S.sweep(s, ms, out, workers=1, rerun=False) == 0
    assert S.sweep(s, ms, out, workers=1, rerun=False) == 0
    assert len(played) == 2  # the second sweep skipped both
    assert "2 already in the results file, 0 to run" in capsys.readouterr().err
    # --rerun plays them again; --table then uses the new lines.
    assert S.sweep(s, ms, out, workers=1, rerun=True) == 0
    assert len(played) == 4 and len(S.read_records(out)[0]) == 4
    cells = S.pool_records(S.read_records(out)[0])
    assert cells[("mcts-16", "heuristic")].games == 2
    # A failing matchup is reported, not written, and retried on the next run.
    bad = [matchup(b="random", games=4)]
    assert S.sweep(s, bad, out, workers=1, rerun=False) == 1
    assert S.sweep(s, bad, out, workers=1, rerun=False) == 1
    assert played[-2:] == [("mcts-16", "random")] * 2 and len(S.read_records(out)[0]) == 4
    # Lines made by other code are reported when skipped.
    monkeypatch.setattr(S, "provenance", lambda: S.Provenance("def", True, "0" * 16, 1))
    S.sweep(s, ms, out, workers=1, rerun=False)
    assert "produced by other code" in capsys.readouterr().err


def test_another_process_finishing_first_is_skipped(tmp_path, monkeypatch):
    out, s = tmp_path / "r.jsonl", settings()
    m = matchup(games=4)
    monkeypatch.setattr(S, "run_matchup", lambda *a: pytest.fail("must not play"))
    S.append_record(out, fake_record(m, s, 1, 0, 0))  # written after the sweep started
    assert S.execute(m, s, out, known=set(), tag="t") is None
    # With --rerun, lines that were there at the start don't count as finished.
    monkeypatch.setattr(S, "run_matchup", lambda m_, s_, p, t: fake_record(m_, s_, 0, 1, 0))
    assert S.execute(m, s, out, known={S.matchup_id(m, s)}, tag="t")["draws"] == 1


# --- Tables ----------------------------------------------------------------------------


def test_pooling_across_seeds_and_runs():
    m, s0, s1 = matchup(), settings(seed=0), settings(seed=1)
    records = [
        fake_record(m, s0, 10, 0, 6, date="2026-10-01T00:00:00+00:00"),
        fake_record(m, s0, 20, 4, 8, date="2026-10-02T00:00:00+00:00"),  # contains the above
        fake_record(m, s1, 5, 1, 2, date="2026-10-03T00:00:00+00:00"),
        fake_record(m, settings(max_turns=100), 1, 0, 0, date="2026-09-01T00:00:00+00:00"),
    ]
    cell = S.pool_records(records)[("mcts-16", "heuristic")]
    assert (cell.wins, cell.draws, cell.losses) == (25, 5, 10)
    assert sorted(cell.seeds) == [0, 1] and cell.other_configs == 1
    assert S.pool_records(records, seed=1)[("mcts-16", "heuristic")].games == 8


def test_table_flags_mixed_configs_code_and_early_stops(tmp_path, capsys):
    out = tmp_path / "r.jsonl"
    s = settings()
    old = S.load_agent("mcts-16")
    old = S.Agent(old.spec, {**old.config, "exploration": 0.5})  # an older default
    lines = [
        fake_record(S.Matchup(old, S.load_agent("random_legal"), 8), s, 8, 0, 0),
        fake_record(matchup(games=8), s, 3, 2, 3, date="2026-10-11T00:00:00+00:00"),
        fake_record(matchup("mcts-32", games=8), s, 5, 0, 3, code_fingerprint="e" * 16,
                    stopped_early=True),
    ]  # fmt: skip
    for line in lines:
        S.append_record(out, line)
    assert S.table_main(out, None, None, None) == 0
    text, err = capsys.readouterr()
    assert "1.000 [0.676, 1.000]‡ (8)" in text  # older mcts-16 config than in the other cell
    assert "0.500 [0.215, 0.785] (8)" in text and "§ (8)" in text
    assert "mcts-16 appear with different configs" in err and "2 code versions" in err
    assert S.table_main(out, {"mcts-32"}, None, None) == 0
    assert "‡" not in capsys.readouterr().out
    assert S.table_main(tmp_path / "missing.jsonl", None, None, None) == 1


# --- Command line ----------------------------------------------------------------------


def test_cli_errors_and_dry_run(tmp_path, capsys):
    out = str(tmp_path / "r.jsonl")
    assert S.main(["--a", "heuristic", "--b", "random", "--size", "2", "--out", out]) == 2
    assert "--size 2" in capsys.readouterr().err
    assert S.main(["--a", "mcts-8", "--b", "random", "--snakes", "5", "--out", out]) == 2
    assert "at most 4 snakes" in capsys.readouterr().err
    assert S.main(["--a", "mcts-8-bogus", "--b", "random", "--out", out]) == 2
    args = ["--a", "mcts-16,mcts-64", "--b", "random_legal,heuristic", "--out", out]
    assert S.main([*args, "--games-for", "mcts-999=8,mcts-64@heuristic=64", "--dry-run",
                   "--workers", "2"]) == 0  # fmt: skip
    text, err = capsys.readouterr()
    assert "--games-for mcts-999 matches no matchup" in err
    assert "mcts-64 vs heuristic: 64 games requested, 64 played, 32 slots" in text
    assert "4 matchups to run" in text and "with --workers 2" in text


def test_sweep_in_process_end_to_end(tmp_path, capsys):
    out = tmp_path / "r.jsonl"
    args = ["--a", "random_legal", "--b", "random", "--games", "6", "--max-turns", "10",
            "--slots", "4", "--out", str(out)]  # fmt: skip
    assert S.main(args) == 0
    (rec,) = S.read_records(out)[0]
    assert rec["games"] == 6 and rec["slots"] == 4 and rec["rounds"] == 1
    assert 0 < rec["slot_utilization"] <= 1 and rec["loop_iterations"] >= 10
    assert rec["ci95_low"] <= rec["score"] <= rec["ci95_high"]
    assert rec["code_fingerprint"] == S.provenance().code_fingerprint
    # Slots and rounds don't change the games: the same matchup with other slots and
    # --target-ci rounds (that never stop) reproduces the counts.
    other = tmp_path / "other.jsonl"
    assert S.main([*args[:-1], str(other), "--slots", "1", "--target-ci", "0", "--round-games",
                   "2"]) == 0  # fmt: skip
    (again,) = S.read_records(other)[0]
    assert again["rounds"] == 3 and again["slots"] == 1
    for field in ("wins", "draws", "losses", "truncated", "mean_turns"):
        assert again[field] == rec[field]
    assert S.main(args) == 0  # resumed: nothing to run
    assert "1 already in the results file, 0 to run" in capsys.readouterr().err


@pytest.mark.skipif(not hasattr(S.os, "sched_setaffinity"), reason="needs CPU affinity")
def test_workers_run_matchups_in_pinned_processes(tmp_path, capsys):
    out = tmp_path / "r.jsonl"
    args = ["--a", "random_legal", "--b", "random,random_legal", "--games", "4", "--max-turns",
            "8", "--slots", "2", "--workers", "2", "--out", str(out)]  # fmt: skip
    assert S.main(args) == 0
    records = S.read_records(out)[0]
    assert sorted(r["b"] for r in records) == ["random", "random_legal"]
    assert all(r["worker_core"] is not None and r["cpus_usable"] == 1 for r in records)
    assert S.main(args) == 0
    assert "2 already in the results file, 0 to run" in capsys.readouterr().err
