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
    if (Object.keys(per).length) {
      out[def.id] = { title: def.title, sub: def.sub, section: def.section ?? null, runs: per };
    }
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


# --- The multi-mode trainers: reinforce_modes, dqn_modes, rainbow_modes, az_modes -----------

CAMPAIGN_MODES = ["standard", "duel", "royale", "constrictor"]
SNAKES = {"standard": 4, "duel": 2, "royale": 4, "constrictor": 4}

# What each trainer's config.json adds to the shared fields (src/slinky/unified.py's
# ModesSettings), as the campaign runs them: the budget is the cap of the main phase.
CAMPAIGN_CONFIGS = {
    "reinforce_modes": {"total_env_steps": 33_554_432, "mode_envs": ["duel=10"],
                        "updates_per_log": 32},
    "dqn_modes": {"total_env_steps": 33_554_432, "num_envs": 32, "log_every": 32_768,
                  "eps_start": 1.0, "eps_end": 0.05},
    "rainbow_modes": {"total_env_steps": 33_554_432, "num_envs": 32, "log_every": 32_768,
                      "noisy": True, "eps_start": 1.0, "eps_end": 0.05},
    "az_modes": {"total_env_steps": 16_777_216, "num_envs": 256, "num_steps": 32,
                 "log_every": 32_768},
}  # fmt: skip


def campaign_config(algorithm: str, **extra) -> dict:
    base = {"algorithm": algorithm, "game_modes": CAMPAIGN_MODES, "cooldown_from": -1}
    return {**base, **CAMPAIGN_CONFIGS[algorithm], **extra}


# The metrics each trainer's chunk_record puts in a game mode's block (value of mode number i).
MODE_METRICS = {
    "reinforce_modes": lambda i: {"pg_loss": 0.001, "v_loss": 0.05 + 0.01 * i,
                                  "entropy": 1.0 - 0.1 * i, "value_mean": 0.01 * i},
    "dqn_modes": lambda i: {"loss": 0.02 * (i + 1), "mean_q": 0.1 * i},
    "rainbow_modes": lambda i: {"loss": 0.5 * (i + 1), "mean_q": 0.1 * i},
    "az_modes": lambda i: {"policy_loss": 1.2, "value_loss": 0.3 + 0.01 * i,
                           "net_entropy": 1.1 - 0.05 * i, "target_entropy": 1.0},
}  # fmt: skip
# ... and at the top of the record, besides the first mode's game statistics.
TOP_METRICS = {
    "reinforce_modes": {"loss": 0.02, "pg_loss": 0.001, "v_loss": 0.06, "entropy": 0.9,
                        "value_mean": 0.0},
    "dqn_modes": {"loss": 0.03, "mean_q": 0.0, "eps": 0.05},
    "rainbow_modes": {"loss": 0.6, "mean_q": 0.0, "eps": 0.05, "noise_sigma": 0.4},
    "az_modes": {"loss": 1.5, "policy_loss": 1.2, "value_loss": 0.3, "net_entropy": 1.1,
                 "target_entropy": 1.0},
}  # fmt: skip


def modes_train(algorithm: str, steps: int, seconds: float = 10.0) -> str:
    """A train record as ``<algorithm>.chunk_record`` writes it (the fields the dashboard reads)."""
    per_mode = {
        mode: {
            "num_snakes": SNAKES[mode], "games": 20, "mean_game_len": 100.0 + 10 * i,
            "draws": i, "truncated": 0,
            "deaths": {"wall": 1, "self": 2 + i, "body": 3, "h2h": 4, "starve": 5,
                       "hazard": 6 if mode == "royale" else 0},
            **MODE_METRICS[algorithm](i),
        }
        for i, mode in enumerate(CAMPAIGN_MODES)
    }  # fmt: skip
    head = per_mode[CAMPAIGN_MODES[0]]
    return json.dumps(
        {
            "type": "train", "env_steps": steps, "lr": 1e-3, "seconds": seconds,
            "env_steps_per_s": 3000.0, "elapsed": steps / 3000.0,
            **{k: head[k] for k in ("games", "mean_game_len", "draws", "truncated", "deaths")},
            **TOP_METRICS[algorithm], "game_modes": per_mode,
        }
    )  # fmt: skip


def test_total_steps_of_the_multi_mode_trainers():
    # The campaign's real configs: a cap that is whole chunks already ...
    for algorithm in ("dqn_modes", "rainbow_modes", "az_modes"):
        config = campaign_config(algorithm)
        assert dash.total_steps(config) == config["total_env_steps"], algorithm
    # ... and a cooldown's end (config.json is rewritten with cooldown_from and the new total).
    ppo = {"algorithm": "ppo_modes", "num_envs": 64, "num_steps": 128, "log_every": 32_768,
           "total_env_steps": 12_845_056, "cooldown_from": 10_747_904}  # fmt: skip
    assert dash.total_steps(ppo) == 12_845_056
    # REINFORCE's chunks are whole games of varying length: the budget itself, which the
    # last chunk overshoots a little (the real run ended at 9,631,980 of 9,628,083).
    cooled = campaign_config("reinforce_modes", total_env_steps=9_628_083, cooldown_from=7_530_931)
    assert dash.total_steps(cooled) == 9_628_083
    assert dash.total_steps({"algorithm": "reinforce_modes"}) is None
    # Otherwise rounded up to whole chunks: DQN and Rainbow take num_envs steps per iteration,
    # log_every // num_envs of them per chunk (9 x 32 = 288) ...
    for algorithm in ("dqn_modes", "rainbow_modes"):
        small = campaign_config(algorithm, total_env_steps=1000, log_every=300)
        assert dash.total_steps(small) == 1152, algorithm
        assert dash.total_steps({**small, "log_every": 1}) == 1024  # at least one iteration
    # ... AlphaZero whole iterations of num_envs * num_steps = 2048 steps.
    az = campaign_config("az_modes", total_env_steps=21_000, num_envs=64, log_every=4096)
    assert dash.total_steps(az) == 24_576  # chunks of 4096
    assert dash.total_steps({**az, "log_every": 1000}) == 22_528  # chunks of 2048


def test_phase_of_a_multi_mode_run():
    assert dash.phase(campaign_config("dqn_modes")) == "main"
    assert dash.phase(campaign_config("reinforce_modes", cooldown_from=7_530_931)) == "cooldown"
    # A run that anneals its learning rate over total_env_steps runs to its end: no phases.
    assert dash.phase({**MODES_CONFIG, "anneal_lr": True, "cooldown_from": -1}) is None
    assert dash.phase({**MODES_CONFIG, "anneal_lr": False, "cooldown_from": -1}) == "main"
    for config in ({}, {"algorithm": "reinforce"}, {"algorithm": "az"}, {"gae_lambda": 0.9}):
        assert dash.phase(config) is None and not dash.is_modes(config)
    assert dash.phase({"algorithm": "az_modes", "cooldown_from": "x"}) == "main"


def test_multi_mode_run_summaries(tmp_path):
    steps = (32_768, 65_536)
    for algorithm in CAMPAIGN_CONFIGS:
        lines = [modes_train(algorithm, s) for s in steps]
        write_run(tmp_path, f"{algorithm}-M-s0", campaign_config(algorithm), lines)
    found = dash.find_runs([tmp_path])
    by_name = {p.name: dash.run_summary(i, p) for i, p in found.items()}
    assert set(by_name) == {f"{a}-M-s0" for a in CAMPAIGN_CONFIGS}
    for algorithm, config in CAMPAIGN_CONFIGS.items():
        s = by_name[f"{algorithm}-M-s0"]
        assert s["algorithm"] == algorithm and s["phase"] == "main"
        # A cap, 65,536 steps in: not finished, no eval (the campaign evaluates snapshots).
        assert s["total_steps"] == config["total_env_steps"] and s["env_steps"] == 65_536
        assert not s["finished"] and s["last_eval"] is None
        assert s["train_seconds"] == 20.0 and s["last_train"]["env_steps"] == 65_536

    # The cooldown rewrites config.json: the run ends at the new total, and is finished there.
    run = write_run(
        tmp_path, "cooled",
        campaign_config("reinforce_modes", total_env_steps=9_628_083, cooldown_from=7_530_931),
        [modes_train("reinforce_modes", 7_530_931), modes_train("reinforce_modes", 9_631_980)],
    )  # fmt: skip
    s = dash.run_summary("cooled", run)
    assert s["phase"] == "cooldown" and s["total_steps"] == 9_628_083 and s["finished"]
    # DONE marks a finished run whatever its steps; a stopped main phase is not finished.
    run = tmp_path / "az_modes-M-s0"
    (run / "STOPPED").write_text("")
    assert not dash.run_summary("x", run)["finished"]
    (run / "DONE").write_text("")
    assert dash.run_summary("x", run)["finished"]
    # The runs that were there before have no phase.
    for name in ("rainbow-a", "reinforce-d"):
        config = {"algorithm": name.split("-")[0], "total_env_steps": 10_000, "num_envs": 32}
        s = dash.run_summary(name, write_run(tmp_path, name, config, [train(6_123)]))
        assert s["phase"] is None


def test_run_list_rereads_only_the_runs_that_changed(runs, monkeypatch):
    # The run list is polled every few seconds: a finished run's metrics are parsed once.
    run, other = runs / "rainbow-a", runs / "ppo-c"
    first = dash.run_summary("a", run)
    reads = []
    real = dash.read_records
    monkeypatch.setattr(dash, "read_records", lambda r: reads.append(r) or real(r))
    again = dash.run_summary("a", run)
    assert not reads and again == first
    with open(run / "metrics.jsonl", "a") as f:
        f.write(train(900, seconds=5.0) + "\n")
    changed = dash.run_summary("a", run)
    assert reads == [run] and changed["env_steps"] == 900 and changed["train_seconds"] == 5.0
    assert changed["version"] != first["version"]
    # Another run is read once, then served from the cache; /api/metrics always reads.
    dash.run_summary("b", other)
    dash.run_summary("b", other)
    dash.read_run(run)
    assert reads == [run, other, run]


def test_multi_mode_http_api(tmp_path):
    for algorithm in CAMPAIGN_CONFIGS:
        lines = [modes_train(algorithm, s) for s in (32_768, 65_536)]
        name = algorithm.replace("_", "-")
        write_run(tmp_path, f"{name}-M-s0", campaign_config(algorithm), lines)
    dash.Handler.roots = [tmp_path]
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/runs", timeout=10) as r:
            listed = {s["algorithm"]: s for s in json.loads(r.read())}
        assert set(listed) == set(CAMPAIGN_CONFIGS)
        for algorithm, config in CAMPAIGN_CONFIGS.items():
            s = listed[algorithm]
            assert s["total_steps"] == config["total_env_steps"] and s["env_steps"] == 65_536
            assert s["phase"] == "main" and not s["finished"] and s["name"].endswith("-M-s0")
        run_id = urllib.request.quote(listed["az_modes"]["id"])
        with urllib.request.urlopen(f"{base}/api/metrics?run={run_id}", timeout=10) as r:
            data = json.loads(r.read())
        # The page gets the per-mode blocks of every train record (and the config's mode order).
        assert data["config"]["game_modes"] == CAMPAIGN_MODES and data["phase"] == "main"
        assert list(data["records"][-1]["game_modes"]) == CAMPAIGN_MODES
    finally:
        server.shutdown()
        server.server_close()


def campaign_page_run(algorithm: str, name: str | None = None, steps=(32_768, 65_536)):
    lines = [modes_train(algorithm, s) for s in steps]
    return page_run(name or f"{algorithm}-M", campaign_config(algorithm), lines)


def test_page_families_of_the_multi_mode_trainers():
    algorithms = [
        "ppo", "ppo_modes", "reinforce", "reinforce_modes", "dqn", "dqn_modes", "rainbow",
        "rainbow_modes", "az", "az_modes", "unknown", "modes",
    ]  # fmt: skip
    code = """(names) => names.map((algorithm) => {
      const d = { algorithm, evals: [], train: [] };
      return [family(d), isModes(d), isPG(d), isAZ(d), logsGameMode(d)];
    })"""
    got = dict(zip(algorithms, page_js(code, algorithms), strict=True))
    # family, multi-mode, policy gradient (no "loss"), AlphaZero, plays every game mode
    assert got == {
        "ppo": ["ppo", False, True, False, False],
        "ppo_modes": ["ppo", True, True, False, True],
        "reinforce": ["reinforce", False, True, False, False],
        "reinforce_modes": ["reinforce", True, True, False, True],
        "dqn": ["dqn", False, False, False, False],
        "dqn_modes": ["dqn", True, False, False, True],
        "rainbow": ["rainbow", False, False, False, False],
        "rainbow_modes": ["rainbow", True, False, False, True],
        "az": ["az", False, False, True, False],
        "az_modes": ["az", True, False, True, True],
        "unknown": ["unknown", False, False, False, False],
        "modes": ["modes", False, False, False, False],
    }  # fmt: skip


def test_page_charts_of_the_multi_mode_trainers():
    top = {  # the charts of each trainer, as its single-mode namesake has them
        "reinforce_modes": {"vloss", "q", "entropy", "len", "draws", "speed"},
        "dqn_modes": {"loss", "q", "len", "draws", "speed", "eps"},
        "rainbow_modes": {"loss", "q", "len", "draws", "speed", "sigma"},  # noisy: no epsilon
        "az_modes": {"loss", "vloss", "entropy", "len", "draws", "speed"},  # no value estimate
    }
    per_mode = {  # ... and the per-mode twin of each of those that the records have per mode
        "reinforce_modes": {"vloss", "q", "entropy", "len", "draws"},
        "dqn_modes": {"loss", "q", "len", "draws"},
        "rainbow_modes": {"loss", "q", "len", "draws"},
        "az_modes": {"vloss", "entropy", "len", "draws"},  # AlphaZero logs no loss per mode
    }
    modes = tuple(f"-{m}" for m in CAMPAIGN_MODES)
    for algorithm in top:
        shown = page_js(CHART_POINTS, {"runs": [campaign_page_run(algorithm)]})
        by_mode = {k for k in shown if k.endswith(modes)}
        assert set(shown) - by_mode == top[algorithm], algorithm
        assert by_mode == {f"{c}-{m}" for c in per_mode[algorithm] for m in CAMPAIGN_MODES}
        # No eval charts: these runs log no evals (the campaign evaluates snapshots elsewhere),
        # and the duel-style "Score vs random_legal" chart is not theirs.
        assert not [k for k in shown if k.startswith("eval-")], algorithm
        for k in by_mode:
            assert shown[k]["runs"] and shown[k]["section"] == "modes", k
        assert all(shown[k]["section"] is None for k in set(shown) - by_mode)

    # Each per-mode chart plots its own mode's block; the top-level chart the first mode's.
    run = campaign_page_run("dqn_modes", "dqn-M")
    shown = page_js(CHART_POINTS, {"runs": [run]})
    steps = (32_768, 65_536)
    assert shown["len"]["runs"] == {"dqn-M": [[s, 100.0] for s in steps]}
    assert shown["len-duel"]["runs"] == {"dqn-M": [[s, 110.0] for s in steps]}
    assert shown["len-constrictor"]["title"] == "Self-play game length, constrictor"
    assert shown["draws-duel"]["runs"] == {"dqn-M": [[s, 1 / 20] for s in steps]}
    assert shown["q-royale"]["runs"]["dqn-M"][0][1] == pytest.approx(0.2)
    assert shown["loss-constrictor"]["runs"]["dqn-M"][0][1] == pytest.approx(0.08)
    # REINFORCE's per-mode value loss, AlphaZero's per-mode policy entropy.
    run = campaign_page_run("reinforce_modes", "pg-M")
    shown = page_js(CHART_POINTS, {"runs": [run]})
    assert shown["vloss-royale"]["runs"]["pg-M"][0][1] == pytest.approx(0.07)
    run = campaign_page_run("az_modes", "az-M")
    shown = page_js(CHART_POINTS, {"runs": [run]})
    assert shown["entropy-duel"]["runs"]["az-M"][0][1] == pytest.approx(1.05)

    # Runs of several trainers share the per-mode charts that apply to each; a single-mode
    # run (no game_modes blocks) is not in any of them.
    runs = [campaign_page_run("dqn_modes"), campaign_page_run("az_modes"), legacy_duel_run()]
    shown = page_js(CHART_POINTS, {"runs": runs})
    assert set(shown["len-duel"]["runs"]) == {"dqn_modes-M", "az_modes-M"}
    assert set(shown["q-duel"]["runs"]) == {"dqn_modes-M"}
    assert set(shown["entropy-duel"]["runs"]) == {"az_modes-M"}
    # With no multi-mode run selected there are no per-mode charts at all.
    shown = page_js(CHART_POINTS, {"runs": [legacy_duel_run()]})
    assert not [k for k in shown if k.endswith(modes)]


def test_page_logged_modes_and_run_links():
    run = campaign_page_run("dqn_modes")
    code = "(run) => loggedModes({ ...run, train: run.records.filter((r) => r.type === 'train') })"
    assert page_js(code, run) == CAMPAIGN_MODES
    # In the order of the config's game_modes, then any others; none without per-mode blocks.
    config = {**run["config"], "game_modes": ["royale", "duel"]}
    assert page_js(code, {**run, "config": config}) == ["royale", "duel", "standard", "constrictor"]
    plain = {**run, "records": [{"type": "train", "env_steps": 1}]}
    assert page_js(code, plain) == [] and page_js(code, legacy_duel_run()) == []

    # #run=<key>: a run's id, else the name of its directory (the newest of that name first).
    runs = [
        {"id": "/root/runs/dqn-modes-M-s0", "name": "dqn-modes-M-s0"},
        {"id": "/old/runs/dqn-modes-M-s0", "name": "dqn-modes-M-s0"},
        {"id": "runs/az", "name": "az"},
    ]  # fmt: skip
    keys = ["/old/runs/dqn-modes-M-s0", "dqn-modes-M-s0", "az", "runs/az", "nope", ""]
    code = "({ runs, keys }) => keys.map((k) => { const r = findRun(runs, k); return r && r.id; })"
    assert page_js(code, {"runs": runs, "keys": keys}) == [
        "/old/runs/dqn-modes-M-s0", "/root/runs/dqn-modes-M-s0", "runs/az", "runs/az", None, None,
    ]  # fmt: skip
    assert page_js(code, {"runs": [], "keys": ["az"]}) == [None]


def test_page_links_to_a_run_and_labels_a_cap():
    page = (ROOT / "baselines" / "dashboard.html").read_text(encoding="utf-8")
    # The hash that selects a run (documented in dashboard.py's docstring), read at load and
    # on hashchange, and kept up to date while one run is plotted.
    assert "#run=" in page and "/^#run=(.+)$/" in page
    assert "link.pending" in page and "applyRunLink" in page and "commitSelection" in page
    assert "#run=" in dash.__doc__ and "directory" in dash.__doc__
    # The run card says "cap" for a multi-mode run's main phase, and a cooldown for its end.
    assert 'r.phase === "main"' in page and 'r.phase === "cooldown"' in page


# --- Strength vs search (simsweep.jsonl) -----------------------------------------------------


def sweep_line(mode: str, sims: int, share: float, **extra) -> str:
    row = {
        "type": "simsweep", "schema": 1, "point": f"{mode}-{sims}", "label": "final",
        "env_steps": 65_536, "game_mode": mode, "simulations": sims, "play": "greedy",
        "chart": True, "share": share,
        "ci95": 0.01, "games": 4096, "games_target": 4096, "complete": True,
    }  # fmt: skip
    return json.dumps({**row, **extra})


def serve(root: Path):
    dash.Handler.roots = [root]
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def test_simsweep_http_api(tmp_path):
    lines = [modes_train("az_modes", s) for s in (32_768, 65_536)]
    run = write_run(tmp_path, "az-modes-M-s0", campaign_config("az_modes"), lines)
    write_run(tmp_path, "az-modes-M-s1", campaign_config("az_modes"), lines)  # no sweep file
    sweep = run / "simsweep.jsonl"
    sweep.write_text(
        sweep_line("duel", 64, 0.5) + "\n" + sweep_line("duel", 4, 0.3) + "\n"
        + sweep_line("duel", 64, 0.8) + "\n" + '{"type": "other"}\n' + '{"type": "simsweep", "poi'
    )  # fmt: skip
    server, base = serve(tmp_path)
    try:
        before = {s["name"]: s for s in get_json(base + "/api/runs")}
        metrics = f"{base}/api/metrics?run="
        data = get_json(metrics + urllib.request.quote(before["az-modes-M-s0"]["id"]))
        assert [(r["simulations"], r["share"]) for r in data["simsweep"]] == [(64, 0.8), (4, 0.3)]
        other = get_json(metrics + urllib.request.quote(before["az-modes-M-s1"]["id"]))
        assert other["simsweep"] == []
        with sweep.open("a") as f:
            f.write(sweep_line("duel", 256, 0.9) + "\n")
        after = {s["name"]: s for s in get_json(base + "/api/runs")}
        a, b = before["az-modes-M-s0"], after["az-modes-M-s0"]
        assert a["version"] != b["version"] and a["updated"] == b["updated"]
        assert before["az-modes-M-s1"]["version"] == after["az-modes-M-s1"]["version"]
    finally:
        server.shutdown()
        server.server_close()


def sweep_rows():
    rows = []
    for s, shares in ((0, [0.1, 0.2, 0.3, 0.4]), (4, [0.5, 0.6, 0.7, 0.8])):
        for m, y in zip(["standard", "duel", "royale", "constrictor"], shares, strict=True):
            rows.append(json.loads(sweep_line(m, s, y, ci95=0.03 if m == "duel" else 0.04)))
    rows.append(json.loads(sweep_line("duel", 16, 0.9)))  # only one mode at 16: no mean there
    rows.append(json.loads(sweep_line("duel", 4, 0.65, ci95=0.03)))  # the last row of a point wins
    rows.append(json.loads(sweep_line("constrictor", 0, 0.99, chart=False)))
    rows.append(json.loads(sweep_line("duel", 0, 0.5, label="latest", env_steps=70_000)))
    rows.append(json.loads(sweep_line("duel", 0, 0.5, label="best", env_steps=1)))
    rows.append(json.loads(sweep_line("duel", 0, 0.5, label="snapshot", env_steps=2)))
    return rows


def test_page_sweep_x():
    got = page_js("(l) => l.map((s) => sweepX(s, l))", [0, 4, 16, 64, 256, 1024])
    assert got == pytest.approx([0, 1, 2, 3, 4, 5], abs=1e-12) or got == [-0, 1, 2, 3, 4, 5]
    assert page_js("(l) => l.map((s) => sweepX(s, [16, 64]))", [0, 16, 64]) == [1, 2, 3]
    assert page_js("(l) => sweepX(0, [0])", None) == -1


def test_page_sweep_cards():
    cards = page_js("(rows) => sweepCards({ simsweep: rows })", sweep_rows())
    assert [(c["label"], c["env_steps"]) for c in cards] == [
        ("final", 65_536), ("best", 1), ("latest", 70_000), ("snapshot", 2),
    ]  # fmt: skip
    final = cards[0]
    assert final["levels"] == [0, 4, 16]
    assert [p["sims"] for p in final["mean"]] == [0, 4]  # all four modes only there
    assert final["mean"][0]["y"] == pytest.approx(0.25)
    assert final["mean"][0]["ci"] == pytest.approx((0.04**2 * 3 + 0.03**2) ** 0.5 / 4)
    duel4 = [p for p in final["perMode"]["duel"] if p["sims"] == 4]
    assert [p["y"] for p in duel4] == [0.65]  # last wins
    assert [r["game_mode"] for r in final["reference"]] == ["constrictor"]
    assert [p["y"] for p in final["perMode"]["constrictor"]] == [0.4, 0.8]  # chart:false excluded
