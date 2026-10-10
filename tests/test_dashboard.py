"""Tests for the training dashboard server (``baselines/dashboard.py``)."""

from __future__ import annotations

import importlib.util
import json
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
    (tmp_path / "not-a-run").mkdir()
    return tmp_path


def test_total_steps_rounds_up_to_whole_chunks():
    # Chunks of 300 // 4 * 4 = 300 steps: 1000 -> 1200, as dqn.py and rainbow.py round.
    assert dash.total_steps({"total_env_steps": 1000, "log_every": 300, "num_envs": 4}) == 1200
    assert dash.total_steps(
        {"total_env_steps": 1_280_000, "log_every": 16_000, "num_envs": 32}
    ) == (1_280_000)
    assert dash.total_steps({}) is None


def test_find_runs_and_summaries(runs):
    found = dash.find_runs([runs])
    assert sorted(p.name for p in found.values()) == ["dqn-b", "rainbow-a"]
    by_name = {p.name: dash.run_summary(i, p) for i, p in found.items()}
    a, b = by_name["rainbow-a"], by_name["dqn-b"]
    assert a["algorithm"] == "rainbow" and b["algorithm"] == "dqn"
    # The half-written last line of a live run is skipped.
    assert a["env_steps"] == 600 and a["total_steps"] == 1200 and not a["finished"]
    assert a["live"] and a["last_eval"] is None
    assert b["finished"] and b["last_eval"]["score"] == 0.9

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


def test_http_api(runs):
    dash.Handler.roots = [runs]
    server = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.headers["Content-Type"], r.read()

    try:
        status, kind, body = get("/")
        assert status == 200 and kind.startswith("text/html") and b"Training runs" in body
        _, _, body = get("/api/runs")
        listed = {r["name"]: r for r in json.loads(body)}
        assert set(listed) == {"rainbow-a", "dqn-b"}
        run_id = listed["dqn-b"]["id"]
        _, kind, body = get(f"/api/metrics?run={urllib.request.quote(run_id)}")
        data = json.loads(body)  # strict JSON: the NaN loss is sent as null
        assert kind == "application/json" and data["config"]["total_env_steps"] == 600
        assert [r["loss"] for r in data["records"] if r["type"] == "train"] == [0.1, None]
        for bad in ("/api/metrics?run=../../etc", "/api/metrics", "/nope"):
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
