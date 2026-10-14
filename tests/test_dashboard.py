"""Tests for the training dashboard server (``baselines/dashboard.py``)."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "baselines_dashboard", ROOT / "baselines" / "dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dash = _load()


def write_run(root: Path, name: str, config: dict | None, lines: list[str]) -> Path:
    run = root / name
    run.mkdir(parents=True)
    if config is not None:
        (run / "config.json").write_text(json.dumps(config))
    (run / "metrics.jsonl").write_text("".join(line + "\n" for line in lines))
    return run


def train(steps: int, **extra) -> str:
    return json.dumps({"type": "train", "env_steps": steps, "loss": 0.1, **extra})


@pytest.fixture
def runs(tmp_path):
    config = {"algorithm": "rainbow", "total_env_steps": 1000, "log_every": 300, "num_envs": 4}
    write_run(tmp_path, "rainbow-a", config, [train(300), train(600), '{"type": "train", "env_st'])
    dqn = {"total_env_steps": 600, "log_every": 300, "num_envs": 4}  # no "algorithm": dqn.py
    evals = json.dumps({"type": "eval", "env_steps": 600, "score": 0.9, "score_ci95": 0.01})
    write_run(tmp_path, "group/dqn-b", dqn, [train(300), train(600, loss=float("nan")), evals])
    ppo = {"total_env_steps": 40_000, "log_every": 32_768, "num_envs": 64, "num_steps": 128}
    ppo_evals = [
        {"opponent": "random_legal", "mode": "greedy", "score": 0.99},
        {"opponent": "random_legal", "mode": "sample", "score": 0.98},
        {"opponent": "heuristic", "mode": "greedy", "score": 0.3},
    ]
    ppo_lines = [
        json.dumps({"type": "eval", "env_steps": 32_768, "score_ci95": 0.01, **e})
        for e in ppo_evals
    ]
    write_run(tmp_path, "ppo-c", {**ppo, "gae_lambda": 0.95}, [train(32_768), *ppo_lines])
    # REINFORCE: whole games per update, so its last chunk overshoots the budget.
    reinforce = {"algorithm": "reinforce", "total_env_steps": 10_000, "num_envs": 32}
    write_run(tmp_path, "reinforce-d", reinforce, [train(6_123), train(10_480)])
    (tmp_path / "not-a-run").mkdir()
    return tmp_path


def write_replay(path: Path, agents: list[str], winners: list[int | None], **extra) -> None:
    games = [
        {
            "id": i,
            "seats": agents if i % 2 == 0 else agents[::-1],
            "result": {"winner": w, "draw": w is None, "truncated": False, "turns": 10 * (i + 1),
                       "elims": []},
            "frames": [],
        }
        for i, w in enumerate(winners)
    ]  # fmt: skip
    replay = {
        "format": "slinky-replay/1", "board": {"width": 11, "height": 11},
        "ruleset": "standard", "map": "standard", "agents": agents, "max_turns": 500,
        "games": games, **extra,
    }  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(replay))


@pytest.fixture
def replays(tmp_path):
    root = tmp_path / "replays"
    # Seats alternate: game 0 seats (a, b), game 1 (b, a). Winner seats 0, 0, None:
    # rainbow wins game 0, ppo game 1, game 2 is a draw.
    write_replay(root / "rainbow" / "r-vs-p.json", ["rainbow", "ppo"], [0, 0, None])
    write_replay(root / "evil.json", ["</script><script>alert(1)</script>", "x"], [1])
    (root / "notes.json").write_text('{"format": "something-else"}')
    (root / "broken.json").write_text("{not json")
    return root


def test_replay_listing_and_summary(replays):
    found = dash.find_replays([replays])
    assert sorted(p.name for p in found.values()) == [
        "broken.json", "evil.json", "notes.json", "r-vs-p.json",
    ]  # fmt: skip
    ok = {p.name: dash.read_replay(p) for p in found.values()}
    assert ok["notes.json"] is None and ok["broken.json"] is None
    rid, path = next((i, p) for i, p in found.items() if p.name == "r-vs-p.json")
    s = dash.replay_summary(rid, path, ok["r-vs-p.json"])
    assert s["wins"] == {"rainbow": 1, "ppo": 1} and s["draws"] == 1 and s["games"] == 3
    assert s["mean_turns"] == 20 and s["board"] == "11×11" and s["name"] == "r-vs-p"
    # A file root is a replay too.
    assert list(dash.find_replays([path]).values()) == [path]


def test_total_steps_rounds_up_to_whole_chunks():
    # Chunks of 300 // 4 * 4 = 300 steps: 1000 -> 1200, as dqn.py and rainbow.py round.
    assert dash.total_steps({"total_env_steps": 1000, "log_every": 300, "num_envs": 4}) == 1200
    assert dash.total_steps(
        {"total_env_steps": 1_280_000, "log_every": 16_000, "num_envs": 32}
    ) == (1_280_000)
    # PPO: chunks of whole iterations (64 envs x 128 steps = 8192 env steps).
    ppo = {"total_env_steps": 40_000, "log_every": 32_768, "num_envs": 64, "num_steps": 128}
    assert dash.total_steps(ppo) == 65_536
    assert dash.total_steps({**ppo, "log_every": 1000}) == 40_960
    assert dash.total_steps({}) is None
    # REINFORCE: the budget itself (chunks are whole batches of games, of any length).
    assert dash.total_steps({"algorithm": "reinforce", "total_env_steps": 5000}) == 5000
    assert dash.total_steps({"algorithm": "reinforce"}) is None


def test_alphazero_headline_eval_is_the_search_agent(tmp_path):
    config = {
        "algorithm": "az",
        "total_env_steps": 3000,
        "log_every": 2048,
        "num_envs": 64,
        "num_steps": 32,
    }
    evals = [
        {"opponent": "random_legal", "mode": "raw", "score": 0.95},
        {"opponent": "random_legal", "mode": "search", "score": 0.99},
        {"opponent": "heuristic", "mode": "search", "score": 0.4},
    ]
    lines = [train(2048), *(json.dumps({"type": "eval", "env_steps": 2048, **e}) for e in evals)]
    run = write_run(tmp_path, "az-d", config, lines)
    summary = dash.run_summary("az-d", run)
    assert summary["algorithm"] == "az" and summary["total_steps"] == 4096
    assert summary["last_eval"]["score"] == 0.99


def test_find_runs_and_summaries(runs):
    found = dash.find_runs([runs])
    assert sorted(p.name for p in found.values()) == ["dqn-b", "ppo-c", "rainbow-a", "reinforce-d"]
    by_name = {p.name: dash.run_summary(i, p) for i, p in found.items()}
    a, b = by_name["rainbow-a"], by_name["dqn-b"]
    assert a["algorithm"] == "rainbow" and b["algorithm"] == "dqn"
    # The half-written last line of a live run is skipped.
    assert a["env_steps"] == 600 and a["total_steps"] == 1200 and not a["finished"]
    assert a["live"] and a["last_eval"] is None
    assert b["finished"] and b["last_eval"]["score"] == 0.9
    # PPO: recognised by its config; the run card shows the greedy eval vs random_legal.
    c = by_name["ppo-c"]
    assert c["algorithm"] == "ppo" and c["total_steps"] == 65_536
    assert c["last_eval"]["score"] == 0.99
    d = by_name["reinforce-d"]
    assert d["algorithm"] == "reinforce" and d["total_steps"] == 10_000 and d["finished"]

    # The version changes with every append (the page refetches on it).
    run = found[a["id"]]
    with open(run / "metrics.jsonl", "a") as f:
        f.write(train(900) + "\n")
    assert dash.run_summary(a["id"], run)["version"] != a["version"]
    # A run directory given as a root is a run too.
    assert list(dash.find_runs([run]).values()) == [run]


def test_live_window_follows_slow_runs(runs, monkeypatch):
    run = runs / "rainbow-a"
    later = (run / "metrics.jsonl").stat().st_mtime + 600  # 10 minutes after the last write
    monkeypatch.setattr(dash.time, "time", lambda: later)
    assert not dash.run_summary("x", run)["live"]
    # A run whose chunks take 5 minutes is still live 10 minutes after its last line.
    with open(run / "metrics.jsonl", "a") as f:
        f.write(train(900, seconds=300.0) + "\n")
    later = (run / "metrics.jsonl").stat().st_mtime + 600
    assert dash.run_summary("x", run)["live"]


def test_http_api(runs, replays):
    dash.Handler.roots = [runs]
    dash.Handler.replay_roots = [replays]
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.headers["Content-Type"], r.read()

    try:
        status, kind, body = get("/")
        assert status == 200 and kind.startswith("text/html") and b"Training dashboard" in body
        _, _, body = get("/api/runs")
        listed = {r["name"]: r for r in json.loads(body)}
        assert set(listed) == {"rainbow-a", "dqn-b", "ppo-c", "reinforce-d"}
        run_id = listed["dqn-b"]["id"]
        _, kind, body = get(f"/api/metrics?run={urllib.request.quote(run_id)}")
        data = json.loads(body)  # strict JSON: the NaN loss is sent as null
        assert kind == "application/json" and data["config"]["total_env_steps"] == 600
        assert [r["loss"] for r in data["records"] if r["type"] == "train"] == [0.1, None]
        for bad in ("/api/metrics?run=../../etc", "/api/metrics", "/nope"):
            with pytest.raises(urllib.error.HTTPError) as err:
                get(bad)
            assert err.value.code == 404
        # Games tab: only the replay files, newest first; each one in the viewer.
        _, _, body = get("/api/replays")
        listed = {r["name"]: r for r in json.loads(body)}
        assert set(listed) == {"r-vs-p", "evil"}
        status, kind, page = get(f"/viewer?replay={urllib.request.quote(listed['r-vs-p']['id'])}")
        assert status == 200 and kind.startswith("text/html")
        page = page.decode()
        start = page.index('id="replay-data">') + len('id="replay-data">')
        embedded = json.loads(page[start : page.index("</script>", start)])
        assert embedded["agents"] == ["rainbow", "ppo"] and len(embedded["games"]) == 3
        # A name with "</script>" can't close the data tag early.
        _, _, page = get(f"/viewer?replay={urllib.request.quote(listed['evil']['id'])}")
        assert b"</script><script>alert" not in page
        for bad in ("/viewer?replay=../../etc/passwd", "/viewer", "/viewer?replay=notes.json"):
            with pytest.raises(urllib.error.HTTPError) as err:
                get(bad)
            assert err.value.code == 404
        # DNS rebinding: a page on another name that resolves to 127.0.0.1 is refused.
        request = urllib.request.Request(base + "/api/runs", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(request, timeout=10)
        assert err.value.code == 421
        request = urllib.request.Request(base + "/", headers={"Host": "localhost:8050"})
        with urllib.request.urlopen(request, timeout=10) as r:
            assert r.status == 200
    finally:
        server.shutdown()
        server.server_close()


# --- ppo_modes: runs that play every game mode -----------------------------------------


def share_eval(steps, mode, opponent, score, play="greedy", **extra):
    """An eval record as ``baselines/ppo_modes.py`` writes it (the fields the dashboard reads)."""
    return json.dumps(
        {
            "type": "eval", "env_steps": steps, "game_mode": mode, "opponent": opponent,
            "mode": play, "score": score, "score_ci95": 0.04, "num_snakes": 4, **extra,
        }
    )  # fmt: skip


MODES_CONFIG = {
    "algorithm": "ppo_modes",
    "game_modes": ["royale", "duel", "standard", "constrictor"],
    "total_env_steps": 40_000,
    "log_every": 32_768,
    "num_envs": 64,
    "num_steps": 128,
}


def modes_records():
    """Train and eval records of a four-mode run at two evaluation points."""
    lines = []
    for steps in (32_768, 65_536):
        lines.append(
            train(
                steps,
                v_loss=0.05,
                entropy=1.0,
                value_mean=0.01,
                games=10,
                draws=2,
                mean_game_len=60.0,
            )
        )
        for i, mode in enumerate(MODES_CONFIG["game_modes"]):
            for opponent, base in (("random_legal", 0.6), ("heuristic", 0.1)):
                score = base + 0.1 * i + steps / 1e6
                lines.append(share_eval(steps, mode, opponent, score))
                lines.append(share_eval(steps, mode, opponent, 0.001, play="sample"))
    return lines


def test_ppo_modes_run_is_recognised_and_its_headline_is_the_first_game_mode(tmp_path):
    run = write_run(tmp_path, "modes-a", MODES_CONFIG, modes_records())
    summary = dash.run_summary("modes-a", run)
    assert summary["algorithm"] == "ppo_modes"
    # Steps and progress are per game mode: 40,000 rounds up to whole 32,768-step chunks.
    assert summary["total_steps"] == 65_536 and summary["env_steps"] == 65_536
    assert summary["finished"]
    # The headline is the first mode of the config (royale, not "standard" nor the last record
    # written), against random_legal, greedy, at the latest point.
    head = summary["last_eval"]
    assert (head["game_mode"], head["opponent"], head["mode"]) == (
        "royale",
        "random_legal",
        "greedy",
    )
    assert head["env_steps"] == 65_536 and head["score"] == pytest.approx(0.6 + 65_536 / 1e6)
    # Another first mode makes another headline; the run's other modes never are.
    duel_first = {**MODES_CONFIG, "game_modes": ["duel", "royale"]}
    run = write_run(tmp_path, "modes-b", duel_first, modes_records())
    assert dash.run_summary("modes-b", run)["last_eval"]["game_mode"] == "duel"


def test_is_headline_eval_with_and_without_a_game_mode():
    head = {"type": "eval", "opponent": "random_legal", "mode": "greedy", "score": 0.5}
    # Records without a game mode behave as they did (no config needed).
    assert dash.is_headline_eval(head) and dash.is_headline_eval(head, {})
    assert dash.is_headline_eval({"score": 0.5}) and dash.is_headline_eval({"mode": "search"})
    assert dash.is_headline_eval(head, {"game_modes": ["duel"]})  # (a legacy record)
    for other in ({"opponent": "heuristic"}, {"mode": "sample"}, {"mode": "raw"}):
        assert not dash.is_headline_eval({**head, **other})
    # A record that names a game mode is the headline only for the run's first one ...
    royale = {**head, "game_mode": "royale"}
    config = {"game_modes": ["royale", "duel"]}
    assert dash.is_headline_eval(royale, config)
    assert not dash.is_headline_eval({**head, "game_mode": "duel"}, config)
    assert not dash.is_headline_eval({**royale, "opponent": "heuristic"}, config)
    assert not dash.is_headline_eval({**royale, "mode": "sample"}, config)
    # ... and, without a config that names the modes, for none.
    for no_config in (None, {}, {"game_modes": []}, {"game_modes": "royale"}):
        assert not dash.is_headline_eval(royale, no_config)
    assert dash.headline_game_mode(config) == "royale" and dash.headline_game_mode({}) is None


def test_ppo_modes_http_api(tmp_path):
    write_run(tmp_path, "modes-a", MODES_CONFIG, modes_records())
    dash.Handler.roots = [tmp_path]
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/runs", timeout=10) as r:
            (listed,) = json.loads(r.read())
        assert listed["algorithm"] == "ppo_modes" and listed["last_eval"]["game_mode"] == "royale"
        run_id = urllib.request.quote(listed["id"])
        with urllib.request.urlopen(f"{base}/api/metrics?run={run_id}", timeout=10) as r:
            data = json.loads(r.read())
        # The page gets the config (to name the headline mode) and every record.
        assert data["config"]["game_modes"][0] == "royale"
        assert sum(r["type"] == "eval" for r in data["records"]) == 2 * 4 * 2 * 2
    finally:
        server.shutdown()
        server.server_close()


# --- The page's chart definitions (pure helpers of dashboard.html, run under node) ----------


def page_js(fn: str, data):
    """``fn(input)`` evaluated after dashboard.html's ``@pure-begin``/``@pure-end`` block."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    page = (ROOT / "baselines" / "dashboard.html").read_text(encoding="utf-8")
    m = re.search(r"// @pure-begin[^\n]*\n(.*?)// @pure-end", page, re.S)
    assert m is not None
    code = (
        m.group(1)
        + "\nconst input = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        + f"\nprocess.stdout.write(JSON.stringify(({fn})(input)));"
    )
    out = subprocess.run(
        [node, "-e", code], input=json.dumps(data), capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


# Per chart: whether it is shown for the selected runs, and each run's points (x, y, lo, hi).
CHART_POINTS = """({runs}) => {
  const datas = runs.map((r) => ({ ...r, evals: r.records.filter((x) => x.type === "eval"),
                                   train: r.records.filter((x) => x.type === "train") }));
  const out = {};
  for (const def of CHARTS) {
    if (def.visible && !def.visible(datas)) continue;
    const per = {};
    datas.forEach((d) => {
      const pts = pointsOf(def, d);
      if (pts) per[d.name] = pts.map((p) => [p.x, p.y]);
    });
    if (Object.keys(per).length) out[def.id] = { title: def.title, sub: def.sub, runs: per };
  }
  return out;
}"""


def page_run(name, config, lines):
    return {"name": name, "algorithm": dash.algorithm(config), "config": config,
            "records": [json.loads(line) for line in lines]}  # fmt: skip


def legacy_duel_run():
    """A ppo.py-style duel run: evals carry no game_mode."""
    lines = [train(32_768, v_loss=0.05, entropy=1.0)]
    for opponent, score in (("random_legal", 0.97), ("heuristic", 0.12)):
        for play in ("greedy", "sample"):
            lines.append(
                json.dumps(
                    {
                        "type": "eval",
                        "env_steps": 32_768,
                        "opponent": opponent,
                        "mode": play,
                        "score": score if play == "greedy" else 0.5,
                        "score_ci95": 0.02,
                    }
                )  # fmt: skip
            )
    return page_run("ppo-duel", {"gae_lambda": 0.95, "algorithm": "ppo"}, lines)


def test_page_treats_ppo_modes_as_ppo():
    algorithms = ["ppo", "ppo_modes", "dqn", "rainbow", "az", "unknown", "pp"]
    is_ppo = page_js("(names) => names.map((algorithm) => isPPO({ algorithm }))", algorithms)
    assert dict(zip(algorithms, is_ppo, strict=True)) == {
        "ppo": True, "ppo_modes": True, "dqn": False, "rainbow": False, "az": False,
        "unknown": False, "pp": False,
    }  # fmt: skip
    assert page_js("(x) => isAZ({ algorithm: x })", "az") is True
    assert page_js("(x) => isAZ({ algorithm: x })", "ppo_modes") is False


def test_page_charts_per_game_mode():
    modes_run = page_run("modes-a", MODES_CONFIG, modes_records())
    legacy = legacy_duel_run()
    shown = page_js(CHART_POINTS, {"runs": [modes_run]})
    # One chart per game mode and opponent, for the greedy records of that mode only ...
    for i, mode in enumerate(MODES_CONFIG["game_modes"]):
        for opponent, base in (("random_legal", 0.6), ("heuristic", 0.1)):
            chart = shown[f"eval-{opponent}-{mode}"]
            assert chart["title"] == f"Share vs {opponent}, {mode}"
            assert chart["runs"] == {
                "modes-a": [
                    [32_768, base + 0.1 * i + 0.032768],
                    [65_536, base + 0.1 * i + 0.065536],
                ]
            }
    # ... and none of the charts for runs without a game mode shows its records (those
    # charts match game_mode == null), and its loss chart is PPO's: absent.
    assert not [k for k in shown if k in ("eval-random_legal", "eval-heuristic", "loss")]
    for k in ("vloss", "q", "entropy", "len", "draws"):
        assert shown[k]["runs"]["modes-a"], k
    # The share subtitle explains ties (a draw is not just 1/2 in a 4-snake game).
    sub = shown["eval-random_legal-standard"]["sub"]
    assert "1/k" in sub and "Parity is 1/N" in sub

    # A mode nobody logged has no chart (a specialist shows its own mode only).
    specialist = page_run(
        "modes-royale",
        {**MODES_CONFIG, "game_modes": ["royale"]},
        [
            share_eval(32_768, "royale", "random_legal", 0.7),
            share_eval(32_768, "royale", "heuristic", 0.2),
        ],
    )
    shown = page_js(CHART_POINTS, {"runs": [specialist]})
    assert {k for k in shown if k.startswith("eval-")} == {
        "eval-random_legal-royale",
        "eval-heuristic-royale",
    }

    # A legacy duel run is a duel: it joins the duel charts as a reference, once a selected run
    # logs game_mode, and keeps its own charts. Its sampled evals are not plotted.
    shown = page_js(CHART_POINTS, {"runs": [modes_run, legacy]})
    assert set(shown["eval-random_legal-duel"]["runs"]) == {"modes-a", "ppo-duel"}
    assert shown["eval-random_legal-duel"]["runs"]["ppo-duel"] == [[32_768, 0.97]]
    assert shown["eval-heuristic-duel"]["runs"]["ppo-duel"] == [[32_768, 0.12]]
    for mode in ("standard", "royale", "constrictor"):  # other modes: the modes run alone
        assert set(shown[f"eval-heuristic-{mode}"]["runs"]) == {"modes-a"}
    assert set(shown["eval-random_legal"]["runs"]) == {"ppo-duel"}
    assert shown["eval-random_legal"]["runs"]["ppo-duel"] == [[32_768, 0.97]]
    assert shown["eval-heuristic"]["runs"]["ppo-duel"] == [[32_768, 0.12]]

    # With no run that logs game_mode selected, the page is what it was: no per-mode charts.
    shown = page_js(CHART_POINTS, {"runs": [legacy]})
    assert not [k for k in shown if k.endswith(tuple(MODES_CONFIG["game_modes"]))]
    assert "loss" not in shown  # PPO's loss is not comparable: it has the value loss chart


def test_page_markers_and_headline_fact():
    page = (ROOT / "baselines" / "dashboard.html").read_text(encoding="utf-8")
    assert page.count("// @pure-begin") == 1 and page.count("// @pure-end") == 1
    assert 'String(d.algorithm).startsWith("ppo")' in page  # ppo and ppo_modes
    # The run card names the game mode of the headline eval it shows.
    assert "ev.game_mode" in page
