"""Tests for ``puffer/train.py``: training the Battlesnake policy with PufferLib's PuffeRL.

A real (tiny) training run on the C environment on the CPU, a few epochs long, then the
properties the script promises:

* the run directory holds a network that slinky's ``ppo:<run dir>`` agent loads and plays,
  and that is the trained one (the export equals ``checkpoint.pt``'s model);
* ``metrics.jsonl`` has the PuffeRL logs (steps, SPS, losses, the C environment's
  statistics), the evaluations and a summary, as valid JSON;
* ``--resume`` continues the same run (steps, epochs, learning-rate schedule);
* the options (PufferLib's ``default.ini`` first, then the Battlesnake defaults, then the
  command line), the argument checks, the 4-snake evaluation, and the ``Multiprocessing``
  backend and the graceful Ctrl-C, which run the script in a subprocess (JAX is loaded in
  this process, and forking after that is not safe).

Needs the built binding (``.venv-puffer/bin/python puffer/setup.py build_ext --inplace``);
run with ``.venv-puffer/bin/python -m pytest tests/test_puffer_train.py``.
"""

from __future__ import annotations

import configparser
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np
import pytest

from slinky.agents import make_agent, make_env
from slinky.evaluate import play_match
from slinky.types import GameConfig

torch = pytest.importorskip("torch")
pytest.importorskip("pufferlib")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `puffer` is a directory of the repository, not installed
    sys.path.append(str(ROOT))
pytest.importorskip(
    "puffer.binding", reason="build it: .venv-puffer/bin/python puffer/setup.py build_ext --inplace"
)

from puffer import policy as policy_lib  # noqa: E402
from puffer import train  # noqa: E402
from puffer.battlesnake import Battlesnake  # noqa: E402

SCRIPT = ROOT / "puffer" / "train.py"
# A run of seconds: 8 games, 16-step segments, a small network, games cut at 40 turns.
# An epoch is agents * 16 steps: 256 for a duel.
SMALL = [
    *("--num-envs", "8"),
    *("--bptt-horizon", "16"),
    *("--minibatch-size", "64"),
    *("--conv-channels", "4,4,4"),
    *("--conv-strides", "2,2,1"),
    *("--hidden", "16"),
    *("--max-turns", "40"),
    *("--print-every", "1e9"),
]
TINY = [*SMALL, *("--eval-games", "0")]
DUEL_EPOCH = 256
METRIC_ENV_KEYS = {
    "episode_length", "draws", "truncated", "deaths_wall", "deaths_self", "deaths_body",
    "deaths_head", "deaths_starve", "deaths_hazard", "n",
}  # fmt: skip
LOSS_KEYS = {"policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "explained_variance"}


@pytest.fixture(autouse=True)
def restore_global_rngs():
    """``train`` seeds python's, numpy's and torch's generators; keep other tests unaffected."""
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def run_args(run_dir: Path, *extra: str):
    return train.parse_args([*TINY, "--run-dir", str(run_dir), *extra])


def records(run_dir: Path, kind: str | None = None) -> list[dict]:
    with open(run_dir / "metrics.jsonl") as f:
        lines = [json.loads(line) for line in f]  # every line is strict JSON
    return [r for r in lines if kind is None or r["type"] == kind]


def wait_for_records(run_dir: Path, kind: str, proc: subprocess.Popen, timeout: float = 90.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert proc.poll() is None, f"the run ended early: {proc.communicate()}"
        try:
            if (run_dir / "metrics.jsonl").exists() and records(run_dir, kind):
                return
        except json.JSONDecodeError:  # caught mid-write
            pass
        time.sleep(0.2)
    raise AssertionError(f"no {kind!r} record after {timeout} s")


# --- Options -----------------------------------------------------------------------------


def test_defaults_are_pufferlibs_then_battlesnakes_then_the_command_line():
    ini = configparser.ConfigParser()
    ini.read(train.pufferlib_ini())
    args = train.parse_args([])
    # Every [train] setting of PufferLib's default.ini reaches PuffeRL, except the labels
    # and the directory that the run directory replaces.
    assert set(args.train) == set(ini["train"]) - set(train.NOT_OPTIONS)
    # Untouched settings keep PufferLib's value; the overridden ones are ours.
    assert args.train["vtrace_rho_clip"] == 1.0 and args.train["max_minibatch_size"] == 32768
    assert args.train["optimizer"] == "adam" and ini["train"]["optimizer"] == "muon"
    assert args.train["learning_rate"] == 1e-3 and args.train["bptt_horizon"] == 128
    assert args.train["prio_alpha"] == 0.0 and args.train["device"] == "cpu"
    assert set(train.BATTLESNAKE_TRAIN) <= set(args.train)
    # The command line beats both, in the type of the setting.
    args = train.parse_args(
        ["--learning-rate", "3e-4", "--total-timesteps", "2e6", "--no-anneal-lr", "--gamma", ".9"]
    )
    assert args.train["learning_rate"] == 3e-4 and args.train["total_timesteps"] == 2_000_000
    assert args.train["anneal_lr"] is False and args.train["gamma"] == 0.9
    assert args.train["batch_size"] == "auto"
    assert train.parse_args(["--batch-size", "65536"]).train["batch_size"] == 65536
    assert args.max_turns == 1000 and args.reward == "win_loss" and args.num_envs == 128


def test_every_puffer_setting_is_a_flag():
    parser, own, defaults = train.build_parser()
    flags = {opt for action in parser._actions for opt in action.option_strings}
    for key in defaults:
        assert "--" + key.replace("_", "-") in flags
    for key in own:
        assert "--" + key.replace("_", "-") in flags
    assert not set(own) & set(defaults)


def test_game_options_follow_the_environment():
    assert train.game_kwargs(train.parse_args([]))["mode"] == "duel"
    four = train.parse_args(["--mode", "royale", "--num-snakes", "3"])
    assert (four.num_snakes, train.game_kwargs(four)["mode"]) == (3, "royale")
    custom = train.parse_args(["--ruleset", "wrapped", "--width", "10", "--height", "8"])
    assert train.game_kwargs(custom)["mode"] is None and not train.evaluable(custom)
    assert train.evaluable(train.parse_args(["--mode", "constrictor"]))
    assert not train.evaluable(train.parse_args(["--ruleset", "solo", "--num-snakes", "1"]))
    assert Path(train.parse_args(["--mode", "royale"]).run_dir).parts[0] == "runs"
    assert Path(train.parse_args(["--mode", "royale"]).run_dir).name.startswith("puffer-royale-")
    for bad in (
        ["--mode", "royale", "--ruleset", "wrapped"],  # a mode sets the ruleset
        ["--mode", "duel", "--num-snakes", "3"],
        ["--mode", "chess"],
        ["--width", "40"],
        ["--ruleset", "standard", "--num-snakes", "9", "--width", "5", "--height", "5"],
        ["--reward", "points"],
        ["--vec-backend", "Ray"],
        ["--conv-channels", "8,8", "--conv-strides", "2"],
        ["--eval-plays", "argmax"],
        ["--eval-every", "10", "--width", "9", "--height", "9"],  # slinky's ppo plays 11x11 only
        ["--optimizer", "sgd"],
    ):
        with pytest.raises(ValueError):
            train.parse_args(bad)


def test_batch_and_device_checks(tmp_path):
    with pytest.raises(ValueError, match="multiple of agents"):
        train.train(run_args(tmp_path / "a", "--batch-size", "1000", "--total-timesteps", "1e5"))
    with pytest.raises(ValueError, match="below one batch"):
        train.train(run_args(tmp_path / "b", "--total-timesteps", "100"))
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="torch.cuda is not available"):
            train.train(run_args(tmp_path / "c", "--device", "cuda"))


def test_metrics_are_strict_json():
    assert train.jsonable({"a": (1, np.float32(2.5)), "b": [float("nan"), np.inf], "c": "x"}) == {
        "a": [1, 2.5],
        "b": [None, None],
        "c": "x",
    }


# --- A run -------------------------------------------------------------------------------


def test_training_run_exports_a_network_slinky_plays(tmp_path):
    run = tmp_path / "run"
    # PuffeRL logs about every quarter second (and logs the losses of the epoch before),
    # so a run of a second and a half has several logs with losses in them.
    summary = train.train(run_args(run, "--total-timesteps", "1e9", "--max-seconds", "1.5"))
    steps_run, epochs_run = summary["agent_steps"], summary["epochs"]
    assert steps_run == epochs_run * DUEL_EPOCH and epochs_run >= 5
    assert not summary["interrupted"] and summary["sps"] > 0 and summary["parameters"] > 0
    assert {"params.npz", "config.json", "checkpoint.pt", "metrics.jsonl"} <= set(os.listdir(run))

    log = records(run)
    assert [r["type"] for r in log][-1] == "summary"
    trains = records(run, "train")
    steps = [r["agent_steps"] for r in trains]
    assert len(steps) >= 3 and steps == sorted(steps) and steps[-1] <= steps_run
    for r in trains:
        assert r["sps"] > 0 and r["uptime"] > 0 and r["epoch"] >= 1 and r["wall_time"] > 0
        assert 0 <= r["learning_rate"] <= 1e-3
    last = trains[-1]
    assert set(last["losses"]) >= LOSS_KEYS and math.isfinite(last["losses"]["entropy"])
    env_stats = [r["env"] for r in trains if "env" in r]
    assert env_stats and all(set(e) == METRIC_ENV_KEYS for e in env_stats)
    assert all(0 < e["episode_length"] <= 40 and e["n"] > 0 for e in env_stats)
    # Games are cut at 40 turns, so every duel is over by then.
    assert all(0 <= e["draws"] <= 1 and 0 <= e["truncated"] <= 1 for e in env_stats)

    # The exported network is the trained one, in ppo.py's format with the run's settings.
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert saved["global_step"] == steps_run and saved["epoch"] == epochs_run
    env = Battlesnake(num_envs=1, mode="duel")
    exported = policy_lib.load_ppo_checkpoint(run, env)
    for name, tensor in saved["model"].items():
        torch.testing.assert_close(exported.state_dict()[name], tensor, rtol=0, atol=0)
    config = json.loads((run / "config.json").read_text())
    assert (config["conv_channels"], config["hidden"], config["max_turns"]) == ([4, 4, 4], 16, 40)
    assert config["puffer_agent_steps"] == steps_run
    assert config["puffer_args"]["train"]["learning_rate"] == 1e-3

    # A resumed run inherits the run's settings, but not the time limit of this session.
    resumed = train.parse_args(["--resume", str(run)])
    assert (resumed.max_seconds, resumed.run_dir) == (0.0, str(run))
    assert resumed.conv_channels == (4, 4, 4) and resumed.train["total_timesteps"] == 1e9

    # slinky's registry loads the run directory and the agent plays complete games.
    game = GameConfig(max_turns=40)
    agent = make_agent(f"ppo:{run}", game)
    assert agent.needs_obs and "greedy" in agent.description
    other = make_agent("random_legal", game)
    result = play_match(make_env(game, True), agent.policy, other.policy, jax.random.key(0), 6, 40)
    assert result.wins + result.draws + result.losses == 6 == result.num_games


def test_resume_continues_the_run(tmp_path):
    run = tmp_path / "run"
    train.train(run_args(run, "--total-timesteps", str(3 * DUEL_EPOCH)))
    assert records(run, "train")[-1]["learning_rate"] == 0  # the schedule ended at 3 epochs

    # Resuming with a longer run restores the schedule at epoch 3 of 5 (a cosine) and the
    # optimizer, before any step; here no step is taken.
    longer = ["--resume", str(run), "--total-timesteps", str(5 * DUEL_EPOCH)]
    idle = train.train(train.parse_args([*longer, "--max-seconds", "1e-9"]))
    assert (idle["agent_steps"], idle["session_steps"], idle["epochs"]) == (3 * DUEL_EPOCH, 0, 3)
    assert idle["learning_rate"] == pytest.approx(1e-3 * (1 + math.cos(3 * math.pi / 5)) / 2)
    state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["optimizer"]["state"], "Adam's moments are in the checkpoint"

    final = train.train(train.parse_args([*longer, "--max-seconds", "0"]))
    assert (final["agent_steps"], final["session_steps"], final["epochs"]) == (
        5 * DUEL_EPOCH,
        2 * DUEL_EPOCH,
        5,
    )
    log = records(run)
    kinds = [r["type"] for r in log]
    assert kinds.count("resume") == 2 and kinds.count("summary") == 3
    trains = records(run, "train")
    epochs = [r["epoch"] for r in trains]
    assert epochs == sorted(epochs) and epochs[-1] == 5
    assert [r["agent_steps"] for r in trains][-1] == 5 * DUEL_EPOCH
    resumed = next(i for i, r in enumerate(log) if r["type"] == "resume")
    assert log[resumed]["agent_steps"] == 3 * DUEL_EPOCH
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert saved["global_step"] == 5 * DUEL_EPOCH
    # The saved options still describe the run: network and game come from them.
    with pytest.raises(ValueError, match="not a run of puffer/train.py"):
        train.parse_args(["--resume", str(tmp_path / "nowhere")])


def test_four_snake_run_is_evaluated_with_a_table(tmp_path):
    run = tmp_path / "run"
    args = run_args(
        run,
        *("--mode", "standard"),
        *("--total-timesteps", "1024"),  # 2 epochs of 32 agents * 16 steps
        *("--eval-games", "8"),
        *("--eval-opponents", "random_legal"),
    )
    summary = train.train(args)
    assert summary["agents"] == 32 and summary["batch_size"] == 512
    (evaluation,) = summary["evals"]
    assert evaluation["agent_steps"] == 1024 and evaluation["opponent"] == "random_legal"
    assert evaluation["games"] == 8 and evaluation["parity"] == 0.25
    assert 0 <= evaluation["share"] <= 1 and 1 <= evaluation["placement"] <= 4
    assert evaluation["wins"] + evaluation["draws"] + evaluation["losses"] == 8
    (logged,) = records(run, "eval")
    assert {k: logged[k] for k in evaluation} == evaluation  # the record in the log
    snapshot = Path(evaluation["snapshot"])
    assert snapshot.parent == (run / "snapshots").resolve()
    assert (snapshot / "params.npz").exists() and (snapshot / "config.json").exists()
    assert "greedy vs random_legal" in train.format_eval(evaluation)


def test_other_boards_train_but_are_not_evaluated(tmp_path, capsys):
    run = tmp_path / "run"
    args = train.parse_args(
        [
            *SMALL,  # the default evaluation settings: it must be skipped, not fail
            *("--ruleset", "wrapped", "--width", "10", "--height", "8", "--num-snakes", "3"),
            *("--run-dir", str(run), "--total-timesteps", "768"),
        ]
    )
    summary = train.train(args)
    assert summary["evals"] == [] and "evaluation skipped" in capsys.readouterr().out
    assert summary["agents"] == 24 and summary["agent_steps"] == 768
    env = Battlesnake(num_envs=1, width=10, height=8, num_snakes=3, ruleset="wrapped")
    assert policy_lib.load_ppo_checkpoint(run, env).board_shape == (8, 10, 13)


# --- The script, in a subprocess ----------------------------------------------------------


def script(run_dir: Path, *extra: str) -> list[str]:
    return [sys.executable, "-W", "ignore", str(SCRIPT), *TINY, "--run-dir", str(run_dir), *extra]


def test_multiprocessing_backend_runs_from_the_command_line(tmp_path):
    run = tmp_path / "run"
    cmd = script(
        run,
        *("--vec-backend", "Multiprocessing", "--vec-num-envs", "2", "--vec-overwork"),
        *("--torch-threads", "1", "--total-timesteps", "2048"),
    )
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "32 agents (2 per game)" in done.stdout
    (summary,) = records(run, "summary")
    assert summary["agent_steps"] >= 2048 and summary["agents"] == 32
    assert policy_lib.load_ppo_checkpoint(run, Battlesnake(num_envs=1, mode="duel"))


def test_interrupt_checkpoints_and_exits(tmp_path):
    run = tmp_path / "run"
    cmd = script(run, "--total-timesteps", "1e9")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path
    )
    try:
        wait_for_records(run, "train", proc)
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=120)
    finally:
        proc.kill()
    assert proc.returncode == 130, err
    assert "interrupted" in out
    (summary,) = records(run, "summary")
    assert summary["interrupted"] and summary["agent_steps"] > 0
    state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["global_step"] == summary["agent_steps"]
    # ... and the run can be resumed.
    more = train.parse_args(
        ["--resume", str(run), "--total-timesteps", "1e9", "--max-seconds", "1"]
    )
    assert train.train(more)["agent_steps"] > summary["agent_steps"]
