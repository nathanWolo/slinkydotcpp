"""Throughput of the Battlesnake C environment, and optionally of PuffeRL training on it.

::

    .venv-puffer/bin/python puffer/bench.py                       # the four modes, markdown table
    .venv-puffer/bin/python puffer/bench.py --modes duel --num-envs 1024 --seconds 5
    .venv-puffer/bin/python puffer/bench.py --train-sps           # also ~60 s of training
    .venv-puffer/bin/python puffer/bench.py --train-sps --device cuda --num-envs 512

*Environment.* One process and one thread step the C games with uniformly random actions
(``puffer.battlesnake.benchmark``: 1024 pre-generated rows of actions, 16 warm-up steps),
so this is the raw cost of the rules, the autoreset and the statistics, with and without
the observations. An *agent step* is one snake acting; a *game step* is a whole game
advancing one turn (``num_snakes`` agent steps). Each cell is the best of ``--repeats``
timed runs, which keeps scheduler noise from lowering it.

Observations dominate: each agent's is 5737 bytes on an 11x11 board, so at about a million
agent steps a second the environment writes several GB/s. PuffeRL's ``Multiprocessing``
backend runs several of these processes at once.

*Training* (``--train-sps``). Runs ``puffer/train.py`` with its default policy and
settings for ``--train-seconds`` and reports agent steps per second, split into the time
spent collecting rollouts (the environment and the policy's forward pass) and updating
(the gradient steps). Further flags are passed on to ``puffer/train.py`` (``--device``,
``--num-envs``, ``--update-epochs`` ...); evaluation and checkpoints are off.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make `puffer` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puffer.battlesnake import MODES, Battlesnake, benchmark


def cpu_description() -> str:
    """``model name (N logical cores)``, as far as the platform tells."""
    model = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            names = [line.split(":", 1)[1].strip() for line in f if line.startswith("model name")]
        model = names[0] if names else model
    except OSError:
        pass
    return f"{model} ({os.cpu_count()} logical cores)"


def measure(mode: str, num_envs: int, observations: bool, seconds: float, repeats: int) -> float:
    """The best agent-steps per second of ``repeats`` runs of ``mode`` with ``num_envs`` games."""
    env = Battlesnake(num_envs=num_envs, mode=mode, compute_obs=observations)
    try:
        return max(benchmark(env, seconds) for _ in range(repeats))
    finally:
        env.close()


def environment_table(
    modes: Sequence[str],
    num_envs: Sequence[int],
    observations: Sequence[bool],
    seconds: float,
    repeats: int,
) -> str:
    """Markdown table of the environment's throughput, one row per configuration."""
    rows = [
        "| mode | snakes | games (num_envs) | observations | agent-steps/s | game-steps/s |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for mode in modes:
        snakes = MODES[mode][2]
        for n in num_envs:
            for obs in observations:
                agent_steps = measure(mode, n, obs, seconds, repeats)
                rows.append(
                    f"| {mode} | {snakes} | {n} | {'yes' if obs else 'no'} "
                    f"| {agent_steps:,.0f} | {agent_steps / snakes:,.0f} |"
                )
                print(".", end="", file=sys.stderr, flush=True)  # progress
    print(file=sys.stderr)
    return "\n".join(rows)


def training_table(argv: Sequence[str], seconds: float) -> str:
    """Train for ``seconds`` with ``puffer/train.py``'s defaults (+ ``argv``); a markdown table."""
    from puffer import train

    with tempfile.TemporaryDirectory() as run_dir:
        args = train.parse_args(
            [
                *("--run-dir", run_dir),
                *("--max-seconds", str(seconds)),
                *("--total-timesteps", "1e12"),
                *("--eval-games", "0"),
                *("--checkpoint-every", "0"),
                *("--print-every", "1e9"),
                *argv,
            ]
        )
        summary = train.train(args)
    rows = [
        "| mode | device | agents | batch | parameters | training seconds | agent-steps/s "
        "| collecting | updating |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['env']} | {summary['device']} | {summary['agents']:,} "
        f"| {summary['batch_size']:,} | {summary['parameters']:,} "
        f"| {summary['training_seconds']:.0f} | {summary['sps']:,.0f} "
        f"| {summary['collect_seconds']:.0f}s | {summary['update_seconds']:.0f}s |",
    ]
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--modes", default=",".join(MODES), help="comma-separated game modes")
    parser.add_argument("--num-envs", default="64,256,1024", help="comma-separated game counts")
    parser.add_argument("--observations", choices=("both", "yes", "no"), default="both")
    parser.add_argument("--seconds", type=float, default=2.0, help="timed seconds per run")
    parser.add_argument("--repeats", type=int, default=2, help="runs per cell (best is shown)")
    parser.add_argument("--train-sps", action="store_true", help="also measure PuffeRL training")
    parser.add_argument("--train-seconds", type=float, default=60.0)
    parser.add_argument("--no-env", action="store_true", help="skip the environment table")
    args, train_argv = parser.parse_known_args(argv)

    if not args.no_env:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        sizes = [int(n) for n in args.num_envs.split(",") if n.strip()]
        observations = {"both": (True, False), "yes": (True,), "no": (False,)}[args.observations]
        print(f"CPU: {cpu_description()}; single process, one thread; random actions\n")
        print(environment_table(modes, sizes, observations, args.seconds, args.repeats))
    if args.train_sps:
        print(f"\nPuffeRL training for {args.train_seconds:.0f} s on {cpu_description()}\n")
        print(training_table(train_argv, args.train_seconds))
    elif train_argv:
        parser.error(f"unrecognized arguments: {' '.join(train_argv)}")


if __name__ == "__main__":
    main()
