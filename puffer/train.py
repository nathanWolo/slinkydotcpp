"""Train the Battlesnake policy with PufferLib 3.0's trainer (``PuffeRL``) on the C environment.

::

    .venv-puffer/bin/python puffer/train.py                          # duel, CPU, 10M agent steps
    .venv-puffer/bin/python puffer/train.py --mode standard --num-envs 256 --device cuda
    .venv-puffer/bin/python puffer/train.py --vec-backend Multiprocessing --vec-num-envs 4
    .venv-puffer/bin/python puffer/train.py --resume runs/puffer-duel-20261015-120000

A *step* here is an *agent step* (one agent slot advancing one turn), PufferLib's unit; a
duel game turn is two. In games of 3 or more snakes the slots of snakes that have died and
wait for the game to end count too (and are averaged into the losses, entropy and KL: PuffeRL
cannot drop rows): under random legal play that is a fifth to a third of the rows with 4
snakes, and more with more snakes.
``--device cuda`` runs on NVIDIA and, with a ROCm build of PyTorch (which exposes AMD GPUs
as ``cuda``), on AMD. The environment is pure C on the CPU either way.

What lands in the run directory (``runs/puffer-<mode>-<timestamp>`` by default):

* ``params.npz`` + ``config.json``: the network in ``baselines/ppo.py``'s checkpoint format,
  rewritten at every checkpoint and at the end, so ``slinky.agents.make_agent("ppo:<run
  dir>")`` plays the latest policy against every other agent (``puffer.policy``). The
  extra ``puffer_*`` keys in ``config.json`` are ignored by ppo.py. Boards other than the
  non-wrapped 11x11 one that agent plays get ``puffer-params.npz`` instead, which ``ppo:``
  does not read (it would play the network on the wrong board).
* ``metrics.jsonl``: one JSON line per PuffeRL log (``"type": "train"``: agent steps, SPS,
  wall time, learning rate, ``env`` = the C environment's per-game statistics, ``losses``,
  ``perf``), one per evaluation (``"type": "eval"``) and a final ``"type": "summary"``.
* ``checkpoint.pt``: model, optimizer, schedule position and step counters. ``--resume
  <run dir>`` continues from it with the saved options (explicit flags override them, the
  optimizer's settings included), in the same directory. Games in flight are not saved: a
  resumed run starts new games. Without ``--resume``, a run in a directory that already
  holds one replaces its ``checkpoint.pt`` and ``metrics.jsonl``, as ``baselines/ppo.py``
  does.
* ``snapshots/step-<N>/``: the exported network at each evaluation, which is also how each
  evaluation gets a fresh ``ppo:<dir>`` name (slinky's agent registry caches by name).

Evaluation (``--eval-every`` agent steps, and always at the end unless ``--eval-games 0``)
exports a snapshot and plays it, greedily, against ``random_legal`` and ``heuristic``
with slinky's own tools: ``play_match`` for two snakes (seats alternate) and ``play_table``
(one policy against copies of the opponent) for more. slinky's ppo agent plays 11x11 boards
that are not wrapped only, so other boards train and export but are not evaluated. JAX runs
on the CPU here (``JAX_PLATFORMS`` is defaulted), so it never competes with torch for a GPU.

Defaults. Every PuffeRL ``[train]`` setting is a flag (``--learning-rate``, ``--gamma`` ...).
They start from PufferLib's ``config/default.ini`` (read from the installed package), then
this module's ``BATTLESNAKE_TRAIN`` overrides, then the command line. The overrides are a
conservative starting point close to ``baselines/ppo.py``, **not tuned**:

* Adam, learning rate 1e-3 with PuffeRL's cosine decay to 0 (ppo.py: linear), eps 1e-5,
  gradient norm clipped at 0.5, gamma 0.99, GAE lambda 0.95, clip 0.2, value coefficient
  0.5, entropy coefficient 0.01: ppo.py's values. (PufferLib's default optimizer is muon,
  which was tuned on other environments; ppo.py uses Adam.)
* PuffeRL does not loop epochs over shuffled minibatches. Each update draws
  ``update_epochs * batch / minibatch`` minibatches of whole trajectory *segments*, and
  recomputes the advantages before each one from the current values and the importance
  ratios of the segments already trained on (V-trace style, ratios clipped at 1: at ratio
  1 this is plain GAE). We use ``update_epochs=2`` and ``minibatch_size=4096``: for a duel
  with the default 128 games, a batch of 32768 agent steps is learned from in 16 gradient
  steps, so each sample is used twice on average (ppo.py: 32 steps over 16384 samples, four
  times each, with half the minibatch). That is gentler, because the advantages track the
  policy as it moves, and on a CPU the gradient steps dominate the cost, so fewer is faster.
* ``bptt_horizon=128`` (ppo.py's ``num_steps``) is the segment length, the unit of
  sampling: GAE cannot cross a segment's end and the kernel leaves the last step of every
  segment with advantage 0, so 1 sample in 128 has no policy gradient. The batch is
  ``agents * horizon`` (``batch_size=auto``), one segment per agent slot, so 128 duel games
  give 256 agents and 32768 steps per update.
* ``prio_alpha=0``: PuffeRL's prioritized segment sampling is off (uniform sampling, no
  importance weights), as in plain PPO.
* ``--max-turns 1000`` as ppo.py. A game cut off by it is terminal for the trainer
  (PufferLib 3.0 has no truncation signal), unlike ppo.py, which bootstraps through it.
  Self-play with parameter sharing, as ppo.py: all snakes are played by the one network.

Vectorization. ``Serial`` (the default) steps every game in the trainer's process. The C
environment makes about a million agent steps a second with observations (``puffer/bench.py``),
so on a CPU it is a few percent of the time and the policy (forward and gradient steps) is the
rest; ``Multiprocessing`` then only adds processes that compete for the cores (measured on 4
vCPUs, standard mode: 11.5k steps/s Serial, 7.0k with 2 workers). It is for a GPU machine with
cores to spare, where the policy is fast and the environments are not.

PuffeRL installs a SIGINT handler that exits without saving, and prints a terminal
dashboard; this module replaces the first with a graceful stop (the run is checkpointed
and can be resumed) and keeps the second off unless ``--dashboard`` is given.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import contextlib
import functools
import importlib.util
import json
import math
import os
import random
import signal
import sys
import threading
import time
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

if __package__ in (None, ""):  # run as a script: make `puffer` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil
import pufferlib
import pufferlib.vector
import torch

from puffer import policy as policy_lib
from puffer.battlesnake import (
    MAPS,
    MODES,
    REWARDS,
    RULESETS,
    Battlesnake,
    check_placement,
    resolve_game,
)

DEFAULT_MODE = "duel"
BACKENDS = ("Serial", "Multiprocessing")
EVAL_OPPONENTS = ("random_legal", "heuristic")
EVAL_PLAYS = ("greedy", "sample")
# What slinky's ppo agent can play (PPOConfig's game).
EVAL_BOARD = (11, 11)

# PuffeRL settings that are not training hyperparameters: wandb/neptune labels and the
# directory of PuffeRL's own checkpoints, which this module replaces by the run directory.
NOT_OPTIONS = ("name", "project", "data_dir")

# Over config/default.ini's [train] (see the module docstring for the reasons).
BATTLESNAKE_TRAIN: dict[str, Any] = dict(
    seed=0,
    device="cpu",
    total_timesteps=10_000_000,
    optimizer="adam",
    learning_rate=1e-3,
    anneal_lr=True,
    gamma=0.99,
    gae_lambda=0.95,
    update_epochs=2,
    clip_coef=0.2,
    vf_clip_coef=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-5,
    batch_size="auto",
    bptt_horizon=128,
    minibatch_size=4096,
    prio_alpha=0.0,
    # PuffeRL's own epoch-based interval is left off; --checkpoint-every is in agent steps.
    checkpoint_interval=2**31 - 1,
)

# Options of one session, which a resumed run does not inherit from the saved ones.
SESSION_OPTIONS = ("resume", "max_seconds", "print_every", "dashboard", "torch_threads")

# Help for the generated --<train setting> flags (the rest say where they come from).
TRAIN_HELP = {
    "seed": "seeds torch, python, numpy and the games",
    "device": "'cpu', or 'cuda' for an NVIDIA GPU or an AMD one with ROCm PyTorch",
    "total_timesteps": "agent steps to train for (also sets the learning-rate schedule)",
    "batch_size": "agent steps per update: 'auto' is agents * bptt_horizon, or a multiple",
    "bptt_horizon": "segment length, the unit of sampling (no recurrence is used)",
    "minibatch_size": "agent steps per gradient step, a multiple of bptt_horizon",
    "update_epochs": "minibatches per update = update_epochs * batch_size / minibatch_size",
    "prio_alpha": "priority exponent of segment sampling (0: uniform)",
    "checkpoint_interval": "checkpoint every this many epochs (see --checkpoint-every)",
}


def pufferlib_ini() -> Path:
    """PufferLib's ``config/default.ini``, located without importing the package.

    Importing pufferlib has side effects (it symlinks ``resources`` into the working
    directory), so ``--help`` and the option defaults avoid it.
    """
    spec = importlib.util.find_spec("pufferlib")
    if spec is None or spec.origin is None:
        raise SystemExit("pufferlib is not installed in this Python")
    return Path(spec.origin).parent / "config" / "default.ini"


def _literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def train_defaults() -> dict[str, Any]:
    """PuffeRL's settings: ``default.ini``'s ``[train]``, then :data:`BATTLESNAKE_TRAIN`."""
    ini = configparser.ConfigParser()
    ini.read(pufferlib_ini())
    values = {k: _literal(v) for k, v in ini["train"].items() if k not in NOT_OPTIONS}
    unknown = set(BATTLESNAKE_TRAIN) - set(values)
    assert not unknown, f"BATTLESNAKE_TRAIN has settings PuffeRL does not: {sorted(unknown)}"
    return {**values, **BATTLESNAKE_TRAIN}


# --- Command line ------------------------------------------------------------------------


def _bool(text: str) -> bool:
    if text.lower() in ("1", "true", "yes", "on"):
        return True
    if text.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def _count(text: str) -> int:
    """An integer, also as ``1e7`` or ``10_000_000``."""
    try:
        return int(text)
    except ValueError:
        value = float(text)
        if value != int(value):
            raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
        return int(value)


def _auto_or_count(text: str) -> int | str:
    return text if text == "auto" else _count(text)


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in text.split(",") if t.strip())


def _names(text: str) -> tuple[str, ...]:
    return tuple(t.strip() for t in text.split(",") if t.strip())


def _show(value: Any) -> str:
    return ",".join(map(str, value)) if isinstance(value, tuple) else str(value)


def _arg_type(default: Any) -> Any:
    if isinstance(default, bool):
        return _bool
    if isinstance(default, int):
        return _count
    if isinstance(default, float):
        return float
    return str


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, Any], dict[str, Any]]:
    """``(parser, own defaults, PuffeRL defaults)``: the options and what each defaults to.

    The parser itself suppresses every default, so :func:`parse_args` can tell what was
    given explicitly (which beats a resumed run's saved options, which beat the defaults).
    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        epilog="See the module docstring (python -c 'import puffer.train as t; "
        "help(t)') for the run directory, the evaluation and the choice of defaults.",
    )
    own: dict[str, Any] = {}

    def add(group: Any, name: str, default: Any, help: str, **kwargs: Any) -> None:
        own[name] = default
        flag = "--" + name.replace("_", "-")
        shown = "none" if default is None else _show(default)
        kwargs.setdefault("type", _arg_type(default) if default is not None else str)
        if kwargs["type"] is _bool:
            del kwargs["type"]
            kwargs["action"] = argparse.BooleanOptionalAction
        group.add_argument(
            flag, dest=name, default=argparse.SUPPRESS, help=f"{help} [{shown}]", **kwargs
        )

    run = parser.add_argument_group("run")
    add(run, "run_dir", None, "run directory (default runs/puffer-<mode>-<timestamp>)")
    add(run, "resume", None, "continue the run in this directory from its checkpoint.pt")
    add(run, "max_seconds", 0.0, "stop training after this many seconds (0: no limit)")
    add(run, "checkpoint_every", 1_000_000, "agent steps between checkpoints and exports")
    add(run, "print_every", 10.0, "seconds between progress lines on the terminal")
    add(run, "dashboard", False, "show PuffeRL's terminal dashboard instead of progress lines")
    add(run, "torch_threads", 0, "torch intra-op threads (0: torch's default)")

    game = parser.add_argument_group("game (the C environment; see puffer.battlesnake)")
    add(game, "mode", None, f"one of {_show(tuple(MODES))} (default duel unless --ruleset/--map)")
    add(game, "ruleset", None, f"one of {_show(RULESETS)} (excludes --mode)")
    add(game, "map", None, f"one of {_show(MAPS)} (excludes --mode)")
    add(game, "width", 11, "board width, 1 to 32")
    add(game, "height", 11, "board height, 1 to 32")
    add(game, "num_snakes", None, "snakes per game (default: the mode's, or 2)", type=_count)
    add(game, "food_spawn_chance", 15, "engine foodSpawnChance")
    add(game, "minimum_food", 1, "engine minimumFood")
    add(game, "hazard_damage_per_turn", 14, "engine hazardDamagePerTurn")
    add(game, "shrink_every_n_turns", 25, "royale shrink interval")
    add(game, "max_turns", 1000, "cut games off at this turn (0: no limit)")
    add(game, "reward", "win_loss", f"one of {_show(REWARDS)}")

    vec = parser.add_argument_group("vectorization (pufferlib.vector.make)")
    add(vec, "num_envs", 128, "games per environment object")
    add(vec, "vec_backend", "Serial", f"one of {_show(BACKENDS)}")
    add(vec, "vec_num_envs", None, "environment objects (1; Multiprocessing: one per worker)",
        type=_count)  # fmt: skip
    add(vec, "vec_num_workers", "auto", "worker processes (Multiprocessing; auto: one each)",
        type=_auto_or_count)  # fmt: skip
    add(vec, "vec_batch_size", "auto", "environment objects per batch (auto: half)",
        type=_auto_or_count)  # fmt: skip
    add(vec, "vec_overwork", False, "allow more workers than physical cores")
    add(vec, "report_interval", 128, "steps between the C environment's statistics")

    net = parser.add_argument_group("network (baselines/ppo.py's, see puffer.policy)")
    add(net, "conv_channels", policy_lib.DEFAULT_CONV_CHANNELS, "convolution channels", type=_ints)
    add(net, "conv_strides", policy_lib.DEFAULT_CONV_STRIDES, "convolution strides", type=_ints)
    add(net, "hidden", policy_lib.DEFAULT_HIDDEN, "width of the dense layer")

    ev = parser.add_argument_group("evaluation (11x11 boards, against slinky's agents)")
    add(ev, "eval_every", 0, "agent steps between evaluations (0: only at the end)")
    add(ev, "eval_games", 256, "games per opponent (0: never evaluate)")
    add(ev, "eval_opponents", EVAL_OPPONENTS, "slinky agent names, comma-separated", type=_names)
    add(ev, "eval_plays", ("greedy",), f"{_show(EVAL_PLAYS)}: how the network plays", type=_names)
    add(ev, "eval_seed", 0, "jax seed of the evaluation games")

    ppo = parser.add_argument_group(
        "PuffeRL training settings (PufferLib's default.ini, then BATTLESNAKE_TRAIN)"
    )
    defaults = train_defaults()
    for key, default in defaults.items():
        kwargs: dict[str, Any] = {}
        if key in ("batch_size", "bptt_horizon"):
            kwargs["type"] = _auto_or_count
        add(ppo, key, default, TRAIN_HELP.get(key, f"PuffeRL train.{key}"), **kwargs)
    return parser, {k: v for k, v in own.items() if k not in defaults}, defaults


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The options as a namespace: own options as attributes, PuffeRL's in ``.train``.

    Raises:
      ValueError: the options do not describe a runnable game or run.
    """
    parser, own, train = build_parser()
    explicit = vars(parser.parse_args(argv))
    values = {**own, **train}
    if (resume := explicit.get("resume")) is not None:
        saved = load_saved_args(resume)
        values.update({k: v for k, v in saved.items() if k != "train" and k not in SESSION_OPTIONS})
        values.update(saved["train"])
        values["run_dir"] = str(resume)
    values.update(explicit)
    args = argparse.Namespace(**values, train={key: values.pop(key) for key in train})
    for name in ("conv_channels", "conv_strides", "eval_opponents", "eval_plays"):
        setattr(args, name, tuple(getattr(args, name)))  # JSON gives lists
    validate(args)
    return args


def load_saved_args(run_dir: str | os.PathLike) -> dict[str, Any]:
    """The options a run was started with (``puffer_args`` in its ``config.json``)."""
    try:
        with open(Path(run_dir) / "config.json") as f:
            return json.load(f)["puffer_args"]
    except (OSError, KeyError) as e:
        raise ValueError(f"{run_dir} is not a run of puffer/train.py to resume: {e!r}") from None


def game_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The :class:`~puffer.battlesnake.Battlesnake` arguments that describe the game."""
    mode = args.mode
    if mode is None and args.ruleset is None and args.map is None:
        mode = DEFAULT_MODE
    return dict(
        mode=mode,
        width=args.width,
        height=args.height,
        num_snakes=args.num_snakes,
        ruleset=args.ruleset,
        map=args.map,
        food_spawn_chance=args.food_spawn_chance,
        minimum_food=args.minimum_food,
        hazard_damage_per_turn=args.hazard_damage_per_turn,
        shrink_every_n_turns=args.shrink_every_n_turns,
    )


def validate(args: argparse.Namespace) -> None:
    """Fail on options that cannot run, before anything is built."""
    game = game_kwargs(args)
    ruleset, map_name, snakes = resolve_game(
        game["mode"], game["num_snakes"], game["ruleset"], game["map"]
    )
    if ruleset not in RULESETS or map_name not in MAPS:
        raise ValueError(f"unknown ruleset {ruleset!r} or map {map_name!r}")
    if args.reward not in REWARDS:
        raise ValueError(f"unknown reward {args.reward!r}; expected one of {REWARDS}")
    if not (1 <= args.width <= 32 and 1 <= args.height <= 32):
        raise ValueError("width and height must be in [1, 32]")
    check_placement(args.width, args.height, snakes)
    if args.max_turns < 0 or args.num_envs < 1:
        raise ValueError("max_turns must be >= 0 and num_envs >= 1")
    if args.vec_backend not in BACKENDS:
        raise ValueError(f"unknown vec backend {args.vec_backend!r}; expected one of {BACKENDS}")
    if args.vec_num_envs is not None and args.vec_num_envs < 1:
        raise ValueError("vec_num_envs must be >= 1")
    if args.report_interval < 1:
        raise ValueError("report_interval must be >= 1")
    if bad := set(args.eval_plays) - set(EVAL_PLAYS):
        raise ValueError(f"unknown eval play {sorted(bad)}; expected {EVAL_PLAYS}")
    if args.train["optimizer"] not in ("adam", "muon"):
        raise ValueError(f"unknown optimizer {args.train['optimizer']!r}; expected adam or muon")
    if args.train["total_timesteps"] < 1:
        raise ValueError("total_timesteps must be >= 1")
    policy_lib._ints(args.conv_channels, "conv_channels")
    policy_lib._ints(args.conv_strides, "conv_strides")
    if len(args.conv_channels) != len(args.conv_strides):
        raise ValueError("conv_channels and conv_strides must have the same length")
    if args.eval_every < 0 or args.eval_games < 0 or args.checkpoint_every < 0:
        raise ValueError("eval_every, eval_games and checkpoint_every must be >= 0")
    if args.eval_every and args.eval_games and not evaluable(args, snakes):
        raise ValueError(
            "--eval-every needs a 2+ snake game on a non-wrapped 11x11 board "
            "(slinky's ppo agent plays only those)"
        )
    if args.run_dir is None:
        label = game["mode"] or f"{ruleset}-{map_name}"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        args.run_dir = str(Path("runs") / f"puffer-{label}-{stamp}")


def evaluable(args: argparse.Namespace, snakes: int | None = None) -> bool:
    """Whether slinky's ppo agent can play this game (so that it can be evaluated)."""
    game = game_kwargs(args)
    ruleset, _, resolved = resolve_game(
        game["mode"], game["num_snakes"], game["ruleset"], game["map"]
    )
    snakes = resolved if snakes is None else snakes
    wrapped = ruleset in ("wrapped", "wrapped_constrictor")
    return (args.width, args.height) == EVAL_BOARD and not wrapped and snakes >= 2


# --- PufferLib glue ----------------------------------------------------------------------


def load_pufferl() -> ModuleType:
    """``pufferlib.pufferl``, imported without its process-wide side effects.

    Importing it makes every ``RuntimeWarning`` an error and replaces the SIGINT handler
    by one that exits at once (``os._exit(0)``, unsaved). Neither is wanted in the process
    that also runs JAX or pytest, so both are put back.
    """
    previous = signal.getsignal(signal.SIGINT)
    try:
        with warnings.catch_warnings():
            import pufferlib.pufferl as pufferl
    finally:
        if previous is not None and threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, previous)
    return pufferl


class _NoUtilization:
    """Stands in for PuffeRL's CPU/GPU sampling thread, which only the dashboard reads.

    The real one polls NVML (or ROCm's SMI) from a background thread; a log-to-file run
    does not need that.
    """

    cpu_util = gpu_util = cpu_mem = gpu_mem = (0.0,)

    def stop(self) -> None:
        pass


@functools.cache
def trainer_class() -> type:
    """:class:`Trainer`, built on first use because it subclasses ``PuffeRL``."""
    pufferl = load_pufferl()

    class Trainer(pufferl.PuffeRL):
        """PuffeRL writing to a run directory instead of its own ``experiments/`` and console.

        * ``save_checkpoint`` (which PuffeRL calls every ``checkpoint_interval`` epochs and
          when training is done) writes ``checkpoint.pt`` and the ppo.py-format export.
        * ``print_dashboard`` is off unless asked for.
        * ``close`` only releases the environments: PuffeRL's copies the model to
          ``data_dir`` for its own loggers.
        """

        def __init__(
            self,
            config: dict[str, Any],
            vecenv: Any,
            policy: policy_lib.BattlesnakePolicy,
            logger: RunLogger,
            args: argparse.Namespace,
            run_dir: Path,
        ) -> None:
            self.args, self.run_dir = args, run_dir
            self.show_dashboard = bool(args.dashboard)
            self.saved_step = -1
            sampler = (
                contextlib.nullcontext()
                if self.show_dashboard
                else mock.patch.object(pufferl, "Utilization", _NoUtilization)
            )
            with sampler:
                super().__init__(config, vecenv, policy, logger)

        def print_dashboard(self, *args: Any, **kwargs: Any) -> None:
            if self.show_dashboard:
                super().print_dashboard(*args, **kwargs)

        def export(self, directory: str | os.PathLike) -> str:
            """The network in ppo.py's format in ``directory``; returns ``params.npz``."""
            args = self.args
            return policy_lib.export_ppo_checkpoint(
                self.uncompiled_policy,
                directory,
                max_turns=args.max_turns or policy_lib.DEFAULT_MAX_TURNS,
                puffer_args=jsonable(vars(args)),
                puffer_game=self.vecenv.driver_env.game_config,
                puffer_agent_steps=self.global_step,
                puffer_epoch=self.epoch,
                puffer_version=str(pufferlib.__version__),
            )

        def save_checkpoint(self) -> str:
            path = self.run_dir / "checkpoint.pt"
            if self.global_step == self.saved_step:
                return str(path)
            self.export(self.run_dir)
            state = dict(
                model=self.uncompiled_policy.state_dict(),
                optimizer=self.optimizer.state_dict(),
                global_step=self.global_step,
                epoch=self.epoch,
                batch_size=self.config["batch_size"],
            )
            tmp = path.with_suffix(".tmp")
            torch.save(state, tmp)
            os.replace(tmp, path)
            self.saved_step = self.global_step
            return str(path)

        def restore(self, state: dict[str, Any]) -> None:
            """Continue from :meth:`save_checkpoint`'s state (the model is already loaded).

            The optimizer keeps the saved moments but takes its settings from this session's
            options (the saved ones unless a flag overrides them): ``load_state_dict`` would
            otherwise bring back the saved learning rate, betas and eps, and ignore the flags.
            """
            self.optimizer.load_state_dict(state["optimizer"])
            self.global_step = self.last_log_step = self.saved_step = int(state["global_step"])
            self.epoch = int(state["epoch"])
            config = self.config
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] = config["learning_rate"]
                if "betas" in group:
                    group["betas"] = (config["adam_beta1"], config["adam_beta2"])
                if "eps" in group:
                    group["eps"] = config["adam_eps"]
            if config["anneal_lr"]:
                # The cosine schedule where this session's loop will leave `remaining` epochs
                # to run, which is the epoch count only if the batch and the total are as they
                # were; its recursive update then continues from here.
                batch = config["batch_size"]
                remaining = -(-max(config["total_timesteps"] - self.global_step, 0) // batch)
                done = min(max(self.total_epochs - remaining, 0), self.total_epochs)
                self.scheduler.last_epoch = done
                for group in self.optimizer.param_groups:
                    t = done / max(self.total_epochs, 1)
                    group["lr"] = group["initial_lr"] * (1 + math.cos(math.pi * t)) / 2

        def close(self) -> None:
            self.utilization.stop()
            self.vecenv.close()

    return Trainer


def jsonable(value: Any) -> Any:
    """``value`` as plain JSON data: numpy to Python, tuples to lists, NaN and inf to None."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.generic, np.ndarray)):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class RunLogger:
    """PuffeRL's logger interface (``run_id``, ``log``, ``close``), writing ``metrics.jsonl``.

    ``log`` receives the flat dict PuffeRL builds at each log (``SPS``, ``agent_steps``,
    ``environment/<stat>``, ``losses/<loss>`` ...) and stores it as one nested line. It
    also prints a progress line every ``print_every`` seconds.
    """

    def __init__(self, run_dir: Path, print_every: float, total_steps: int, quiet: bool) -> None:
        self.run_id = run_dir.name
        self.path = run_dir / "metrics.jsonl"
        self.print_every = print_every
        self.total_steps = total_steps
        self.quiet = quiet
        self.last_printed = float("-inf")
        self.last: dict[str, Any] = {}

    def write(self, record: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(jsonable({**record, "wall_time": time.time()})) + "\n")

    def log(self, logs: dict[str, Any], step: int) -> None:
        groups = {"environment": "env", "losses": "losses", "performance": "perf"}
        record: dict[str, Any] = {"type": "train", "agent_steps": int(step)}
        for key, value in logs.items():
            head, _, name = key.partition("/")
            if name and head in groups:
                record.setdefault(groups[head], {})[name] = value
            elif key != "agent_steps":
                record[{"SPS": "sps"}.get(key, key)] = value
        self.write(record)
        self.last = record
        now = time.monotonic()
        if not self.quiet and now - self.last_printed >= self.print_every:
            self.last_printed = now
            print(self.progress_line(record), flush=True)

    def progress_line(self, record: dict[str, Any]) -> str:
        env, losses = record.get("env", {}), record.get("losses", {})

        def get(group: dict[str, Any], name: str, fmt: str) -> str:
            value = group.get(name)
            return "-" if value is None else format(value, fmt)

        return (
            f"[{record['agent_steps']:>12,} / {self.total_steps:,}] "
            f"sps {record.get('sps', 0):>8,.0f}  ep_len {get(env, 'episode_length', '6.1f')}  "
            f"ent {get(losses, 'entropy', '5.3f')}  v_loss {get(losses, 'value_loss', '6.4f')}  "
            f"kl {get(losses, 'approx_kl', '6.4f')}  expl_var "
            f"{get(losses, 'explained_variance', '5.2f')}  lr {get(record, 'learning_rate', '.2e')}"
        )

    def close(self, model_path: str | None = None) -> None:
        pass


@contextlib.contextmanager
def forked_workers_ignore_sigint() -> Iterator[None]:
    """Worker processes forked inside inherit SIGINT ignored, so Ctrl-C reaches only us.

    The parent then checkpoints and terminates them (``vecenv.close``) instead of every
    worker printing a traceback.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@contextlib.contextmanager
def graceful_stop() -> Iterator[None]:
    """SIGINT and SIGTERM raise ``KeyboardInterrupt`` (to checkpoint) in the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(signal.Signals(signum).name)

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def make_vecenv(args: argparse.Namespace) -> Any:
    """The ``pufferlib.vector`` environment: ``vec_num_envs`` objects of ``num_envs`` games."""
    env_kwargs = dict(
        game_kwargs(args),
        num_envs=args.num_envs,
        max_turns=args.max_turns,
        reward=args.reward,
        report_interval=args.report_interval,
    )
    vec: dict[str, Any] = dict(backend=args.vec_backend)
    if args.vec_backend == "Multiprocessing":
        workers = args.vec_num_workers
        cores = psutil.cpu_count(logical=False) or 1
        vec_num_envs = args.vec_num_envs or (workers if isinstance(workers, int) else cores)
        vec.update(
            num_envs=vec_num_envs,
            num_workers=workers,
            batch_size=args.vec_batch_size,
            overwork=args.vec_overwork,
        )
    else:
        vec["num_envs"] = args.vec_num_envs or 1
    with forked_workers_ignore_sigint():
        return pufferlib.vector.make(Battlesnake, env_kwargs=env_kwargs, **vec)


def check_batch(config: dict[str, Any], agents: int) -> int | None:
    """The batch size PuffeRL will use, checked to be whole segments (``None``: both ``auto``).

    PuffeRL stores the rollout as ``agents`` rows per ``bptt_horizon`` steps; an explicit
    ``batch_size`` that is not a multiple of that would index past the buffers. With
    ``bptt_horizon`` ``auto`` it takes ``batch_size // agents`` as the horizon, so the same
    holds for that.
    """
    horizon, batch = config["bptt_horizon"], config["batch_size"]
    if batch == "auto":
        return None if horizon == "auto" else agents * horizon
    if horizon == "auto":
        horizon = batch // agents
        if horizon < 1:
            raise ValueError(f"batch_size {batch} is below the {agents} agents (one step each)")
    if batch % (agents * horizon):
        raise ValueError(
            f"batch_size {batch} must be a multiple of agents * bptt_horizon = "
            f"{agents} * {horizon} (or 'auto')"
        )
    return batch


# --- Evaluation --------------------------------------------------------------------------


def forget_agent(name: str, config: Any) -> None:
    """Drop an agent from slinky's registry cache, which never evicts.

    Every evaluation plays a snapshot under a new name, so keeping them all would hold each
    one's parameters and compiled policy (megabytes) for the life of the process.
    """
    from slinky import agents

    agents._CACHE.pop((agents.parse_agent(name).name, config), None)


def evaluate_snapshot(
    snapshot: str | os.PathLike,
    game_config: dict[str, Any],
    max_turns: int,
    games: int,
    opponents: Sequence[str],
    plays: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Play the exported network in ``snapshot`` against slinky's agents; one record each.

    Two snakes: ``play_match`` (the network is seated in alternating seats), with its
    wins, draws, losses and ``score = (wins + draws / 2) / games``. More snakes:
    ``play_table`` with one network seat per game rotating among the snakes and the other
    seats played by the opponent: the network's ``share`` of the wins (parity: 1 / snakes)
    and mean ``placement`` (parity: (snakes + 1) / 2).
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    from slinky.agents import make_agent, make_env
    from slinky.evaluate import play_match, play_table
    from slinky.types import GameConfig

    config = GameConfig(**{**game_config, "max_turns": max_turns})
    env = make_env(config, True)
    snakes = config.num_snakes
    key = jax.random.key(seed)
    records = []
    for play in plays:
        name = f"ppo{'' if play == 'greedy' else '-sample'}:{snapshot}"
        agent = make_agent(name, config)
        forget_agent(name, config)
        for opponent in opponents:
            other = make_agent(opponent, config)
            start = time.perf_counter()
            if snakes == 2:
                r = play_match(env, agent.policy, other.policy, key, games, max_turns)
                result = dict(
                    wins=r.wins, draws=r.draws, losses=r.losses, truncated=r.truncated,
                    mean_turns=r.mean_turns, score=r.score, score_ci95=r.score_ci95,
                )  # fmt: skip
            else:
                lineup = (0,) + (1,) * (snakes - 1)
                t = play_table(env, [agent.policy, other.policy], lineup, key, games, max_turns)
                result = dict(
                    wins=t.wins[0], draws=t.ties[0], losses=t.losses[0],
                    truncated=t.truncated, mean_turns=t.mean_turns,
                    share=t.share[0], share_ci95=t.share_ci95[0], parity=1 / snakes,
                    placement=t.placement[0], placement_ci95=t.placement_ci95[0],
                )  # fmt: skip
            records.append(
                dict(
                    type="eval",
                    opponent=opponent,
                    play=play,
                    games=games,
                    **result,
                    seconds=time.perf_counter() - start,
                    snapshot=str(snapshot),
                )  # fmt: skip
            )
    return records


def format_eval(record: dict[str, Any]) -> str:
    if "score" in record:
        outcome = (
            f"score {record['score']:.3f} +- {record['score_ci95']:.3f}  "
            f"W/D/L {record['wins']}/{record['draws']}/{record['losses']}"
        )
    else:
        outcome = (
            f"share {record['share']:.3f} +- {record['share_ci95']:.3f} "
            f"(parity {record['parity']:.3f})  placement {record['placement']:.2f}  "
            f"W/T/L {record['wins']}/{record['draws']}/{record['losses']}"
        )
    return f"eval {record['play']} vs {record['opponent']}: {outcome}  ({record['seconds']:.0f}s)"


# --- The training run --------------------------------------------------------------------


def crossed(every: int, before: int, now: int) -> bool:
    """Whether a multiple of ``every`` (> 0) lies in ``(before, now]``.

    Evaluations and checkpoints fall on multiples of their interval counted from step 0, so a
    resumed run keeps the schedule of the run (rather than restarting it at the resume step).
    """
    return every > 0 and now // every > before // every


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run ``args`` (from :func:`parse_args`); returns the summary also written to the log.

    The summary has ``agent_steps`` (in total), ``session_steps`` and ``training_seconds``
    (this call's; the time inside PuffeRL's ``evaluate`` and ``train``, excluding
    evaluation games), ``sps = session_steps / training_seconds``, the split
    ``collect_seconds`` / ``update_seconds``, ``evals`` (the evaluation records),
    ``interrupted``, ``run_dir``, the final ``learning_rate`` and what ran (``env``,
    ``device``, ``agents``, ``batch_size``, ``parameters``).
    """
    run_dir = Path(args.run_dir)
    resume = None
    if args.resume is not None:
        if not (run_dir / "checkpoint.pt").is_file():
            raise ValueError(f"{run_dir} has no checkpoint.pt to resume from")
        resume = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    train_cfg = dict(args.train)
    device = train_cfg["device"]
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"device {device!r} requested but torch.cuda is not available")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    seed = int(train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    vecenv = make_vecenv(args)
    start_step = 0
    try:
        env = vecenv.driver_env
        policy = policy_lib.BattlesnakePolicy(
            env, args.conv_channels, args.conv_strides, args.hidden
        ).to(device)
        if resume is not None:
            policy.load_state_dict(resume["model"])
        total = int(train_cfg["total_timesteps"])
        env_name = "battlesnake_" + (game_kwargs(args)["mode"] or "custom")
        config = dict(train_cfg, env=env_name, use_rnn=False, data_dir=str(run_dir))
        batch = check_batch(config, vecenv.num_agents)
        if batch is not None:
            if batch > total:
                raise ValueError(f"total_timesteps {total:,} is below one batch of {batch:,} steps")
            # PuffeRL's schedule has total // batch epochs, but the loop below runs until the
            # total is reached: a partial last batch would be an extra epoch at learning rate
            # 0 (the schedule's end), then a rising one.
            total = -(-total // batch) * batch
            config["total_timesteps"] = total
        logger = RunLogger(run_dir, args.print_every, total, quiet=bool(args.dashboard))
        with warnings.catch_warnings():
            # PuffeRL builds a CUDA autocast context whatever the device.
            warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'")
            trainer = trainer_class()(config, vecenv, policy, logger, args, run_dir)
        if resume is not None:
            trainer.restore(resume)
            start_step = trainer.global_step
            logger.write({"type": "resume", "agent_steps": start_step, "epoch": trainer.epoch})
        else:
            # A new run in an old run directory replaces it, as baselines/ppo.py does; its
            # checkpoint would otherwise survive until the first save and a --resume would
            # continue the old weights under the new run's options.
            for stale in ("metrics.jsonl", "checkpoint.pt"):
                (run_dir / stale).unlink(missing_ok=True)
            trainer.export(run_dir)  # a valid run directory from the start
    except BaseException:
        vecenv.close()
        raise
    header = (
        f"{env_name}: {vecenv.num_agents} agents ({env.num_snakes} per game), "
        f"batch {config['batch_size']:,} steps, device {device}"
        f"{' (ROCm)' if getattr(torch.version, 'hip', None) else ''}, "
        f"{sum(p.numel() for p in policy.parameters()):,} parameters, run {run_dir}"
    )
    if not args.dashboard:
        print(header, flush=True)

    can_eval = args.eval_games > 0 and evaluable(args)
    if args.eval_games > 0 and not can_eval:
        print("evaluation skipped: slinky's ppo agent plays 2+ snake 11x11 non-wrapped games")
    evals: list[dict[str, Any]] = []

    def run_eval() -> None:
        began = time.time()
        step = trainer.global_step
        snapshot = run_dir / "snapshots" / f"step-{step:012d}"
        trainer.export(snapshot)
        records = evaluate_snapshot(
            snapshot.resolve(),
            env.game_config,
            args.max_turns or policy_lib.DEFAULT_MAX_TURNS,
            args.eval_games,
            args.eval_opponents,
            args.eval_plays,
            args.eval_seed,
        )
        for record in records:
            record["agent_steps"] = step
            logger.write(record)
            print(format_eval(record), flush=True)
        evals.extend(records)
        # PuffeRL's sps divides the steps since its last log by the time since then, and its
        # uptime counts from its start: neither should include the evaluation.
        trainer.last_log_time = time.time()
        trainer.last_log_step = trainer.global_step
        trainer.start_time += trainer.last_log_time - began

    collect = update = evaluating = 0.0
    last_checkpoint = last_eval = trainer.global_step
    interrupted = False
    started = time.perf_counter()
    try:
        with graceful_stop():
            while trainer.global_step < total:
                if args.max_seconds and time.perf_counter() - started >= args.max_seconds:
                    break
                tick = time.perf_counter()
                trainer.evaluate()
                collect += time.perf_counter() - tick
                tick = time.perf_counter()
                trainer.train()
                update += time.perf_counter() - tick
                step = trainer.global_step
                if crossed(args.checkpoint_every, last_checkpoint, step):
                    trainer.save_checkpoint()
                    last_checkpoint = step
                if can_eval and crossed(args.eval_every, last_eval, step):
                    tick = time.perf_counter()
                    run_eval()
                    evaluating += time.perf_counter() - tick
                    last_eval = step
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted: saving a checkpoint", flush=True)
    finally:
        try:
            trainer.save_checkpoint()
        finally:
            trainer.close()  # also when the save fails: workers would outlive the process
    if can_eval and not interrupted and (trainer.global_step != last_eval or not evals):
        tick = time.perf_counter()
        run_eval()
        evaluating += time.perf_counter() - tick

    training = collect + update
    session = trainer.global_step - start_step
    summary = dict(
        type="summary",
        run_dir=str(run_dir),
        agent_steps=trainer.global_step,
        session_steps=session,
        epochs=trainer.epoch,
        training_seconds=training,
        collect_seconds=collect,
        update_seconds=update,
        eval_seconds=evaluating,
        sps=session / training if training > 0 else 0.0,
        interrupted=interrupted,
        learning_rate=trainer.optimizer.param_groups[0]["lr"],
        env=env_name,
        device=str(device),
        agents=vecenv.num_agents,
        batch_size=config["batch_size"],
        parameters=sum(p.numel() for p in policy.parameters()),
    )
    logger.write(summary)
    if not args.dashboard:
        print(
            f"done: {trainer.global_step:,} agent steps in {training:.0f}s of training "
            f"({summary['sps']:,.0f} steps/s: {collect:.0f}s collecting, {update:.0f}s updating)",
            flush=True,
        )
    return {**summary, "evals": evals, "last_log": logger.last}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        summary = train(args)
    except (ValueError, pufferlib.APIUsageError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 130 if summary["interrupted"] else 0


if __name__ == "__main__":
    sys.exit(main())
