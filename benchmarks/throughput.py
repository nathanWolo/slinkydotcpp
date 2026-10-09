"""Throughput benchmark: environment steps per second for batches of games.

One "step" is one game advancing one turn (all snakes move). The measured
program is ``jax.jit(jax.vmap(rollout))`` where ``rollout`` plays one game for
``--turns`` turns with ``jax.lax.scan`` under ``random_legal_policy``; the
timing excludes compilation (a warmup call), blocks on the result and reports
the best of ``--repeats`` runs. ``agent-steps/s`` is ``steps/s * num_snakes``.

Variants (see ``VARIANTS``):

* ``sim``              ``obs=None``, ``step_autoreset``: pure simulation.
* ``sim+ego-obs``      egocentric observations computed (and consumed) every step.
* ``sim-no-autoreset`` plain ``env.step``; finished games stay done. Comparing it
                       with ``sim`` isolates the cost of autoreset.
* ``reset-only``       just ``env.reset`` per step (a "step" here is one reset),
                       i.e. the extra work ``step_autoreset`` does under vmap.

Usage::

    python benchmarks/throughput.py                    # full sweep
    python benchmarks/throughput.py --quick            # skip batch 8192
    python benchmarks/throughput.py --configs duel --batch-sizes 1024 --json out.json

Progress goes to stderr; stdout carries only the markdown report, so it can be
redirected to a file. The script runs unchanged on CPU, GPU and TPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from slinky.env import BattlesnakeEnv
from slinky.policies import random_legal_policy, random_policy
from slinky.types import GameConfig

CONFIGS: dict[str, GameConfig] = {
    "duel": GameConfig(),
    "standard-4p": GameConfig(num_snakes=4),
    "standard-19x19-4p": GameConfig(width=19, height=19, num_snakes=4),
    "wrapped-2p": GameConfig(ruleset="wrapped"),
    "constrictor-2p": GameConfig(ruleset="constrictor"),
}
POLICIES: dict[str, Callable] = {
    "random_legal": random_legal_policy,
    "random": random_policy,
}
DEFAULT_BATCH_SIZES = (1, 64, 1024, 8192)
QUICK_MAX_BATCH = 1024
DEFAULT_TURNS = 100
LENGTH_BATCH = 4096  # games used to measure the mean game length
LENGTH_TURNS = 1000  # turns each of them is played for (unfinished games are censored)


@dataclass(frozen=True)
class Variant:
    """What a benchmark row measures."""

    obs: str | None  # observation kind for the env, or None
    autoreset: bool  # step_autoreset vs plain step
    reset_only: bool = False  # time env.reset alone (one "step" is one reset)


VARIANTS: dict[str, Variant] = {
    "sim": Variant(obs=None, autoreset=True),
    "sim+ego-obs": Variant(obs="egocentric", autoreset=True),
    "sim-no-autoreset": Variant(obs=None, autoreset=False),
    "reset-only": Variant(obs=None, autoreset=True, reset_only=True),
}
# The cheap variants run for every config; the duel gets all of them.
OTHER_CONFIG_VARIANTS = ("sim", "sim-no-autoreset")


class Totals(NamedTuple):
    """Per-game scalars accumulated inside the scan.

    They are returned from the jitted function so that XLA cannot dead-code
    eliminate the work that produces them (rewards, observations, ...).
    """

    live: jax.Array  # int32[]  steps taken while the game was still running
    episodes: jax.Array  # int32[]  ``done`` flags seen (finished games, if autoreset)
    reward: jax.Array  # float32[] summed rewards of all snakes
    obs: jax.Array  # float32[] summed observation values


# --- Programs under test ---------------------------------------------------------


def _tree_sum(tree: Any) -> jax.Array:
    return sum((jnp.sum(x, dtype=jnp.float32) for x in jax.tree.leaves(tree)), jnp.float32(0))


def make_rollout(
    env: BattlesnakeEnv, policy: Callable, num_turns: int, autoreset: bool
) -> Callable[[jax.Array], tuple[Any, Totals]]:
    """``rollout(key) -> (final_state, Totals)`` playing one game for ``num_turns`` turns."""
    step = env.step_autoreset if autoreset else env.step

    def rollout(key: jax.Array):
        k_reset, k_play = jax.random.split(key)
        state, _ = env.reset(k_reset)
        # Every key is drawn up front, in one call, instead of splitting inside the scan.
        keys = jax.random.split(k_play, (num_turns, 2))

        def body(carry, ks):
            state, tot = carry
            actions = policy(ks[0], state, env)
            new_state, ts = step(ks[1], state, actions)
            tot = Totals(
                live=tot.live + (~state.done).astype(jnp.int32),
                episodes=tot.episodes + ts.done.astype(jnp.int32),
                reward=tot.reward + jnp.sum(ts.reward),
                obs=tot.obs + _tree_sum(ts.obs),
            )
            return (new_state, tot), None

        zero = Totals(jnp.int32(0), jnp.int32(0), jnp.float32(0), jnp.float32(0))
        (state, tot), _ = jax.lax.scan(body, (state, zero), keys)
        return state, tot

    return rollout


def make_reset_loop(env: BattlesnakeEnv, num_turns: int) -> Callable[[jax.Array], Any]:
    """``loop(key)`` running ``env.reset`` ``num_turns`` times (the autoreset overhead)."""

    def loop(key: jax.Array):
        def body(acc, k):
            state, _ = env.reset(k)
            # Consume the state so XLA must build all of it.
            return acc + _tree_sum(state), None

        acc, _ = jax.lax.scan(body, jnp.float32(0), jax.random.split(key, num_turns))
        return acc

    return loop


def build_program(
    env: BattlesnakeEnv, variant: Variant, policy: Callable, num_turns: int
) -> Callable[[jax.Array], Any]:
    """The jitted, batched program for a variant: ``keys[B, ...] -> outputs``."""
    if variant.reset_only:
        return jax.jit(jax.vmap(make_reset_loop(env, num_turns)))
    return jax.jit(jax.vmap(make_rollout(env, policy, num_turns, variant.autoreset)))


def measure_game_length(config: GameConfig, policy: Callable, seed: int) -> dict[str, float]:
    """Game-length statistics under ``policy``, from the per-turn ``done`` flags.

    Plays ``LENGTH_BATCH`` games without autoreset for ``LENGTH_TURNS`` turns;
    a game's length is the number of turns until its first ``done``.
    """
    env = BattlesnakeEnv(config, obs=None)

    def play(key):
        k_reset, k_play = jax.random.split(key)
        state, _ = env.reset(k_reset)

        def body(state, ks):
            state, ts = env.step(ks[1], state, policy(ks[0], state, env))
            return state, ts.done

        _, done = jax.lax.scan(body, state, jax.random.split(k_play, (LENGTH_TURNS, 2)))
        return done

    keys = jax.random.split(jax.random.key(seed), LENGTH_BATCH)
    done = np.asarray(jax.jit(jax.vmap(play))(keys))  # bool[B, T]
    finished = done.any(axis=1)
    lengths = done.argmax(axis=1)[finished] + 1  # first done at index t => t + 1 turns played
    return {
        "games": int(LENGTH_BATCH),
        "finished": int(finished.sum()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "max": int(lengths.max()),
    }


# --- Timing ---------------------------------------------------------------------


def time_program(
    fn: Callable[[jax.Array], Any], batch: int, repeats: int, seed: int
) -> tuple[list[float], float, Any]:
    """Run ``fn`` on ``batch`` fresh keys: (seconds per repeat, warmup seconds, last output)."""
    base = jax.random.key(seed)
    # Keys are made (and ready) before the clock starts.
    all_keys = [
        jax.block_until_ready(jax.random.split(jax.random.fold_in(base, r), batch))
        for r in range(repeats + 1)
    ]
    t0 = time.perf_counter()
    jax.block_until_ready(fn(all_keys[0]))  # compile + first run
    warmup = time.perf_counter() - t0
    times, out = [], None
    for keys in all_keys[1:]:
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(keys))
        times.append(time.perf_counter() - t0)
    return times, warmup, out


def run_cell(
    name: str, config: GameConfig, variant_name: str, batch: int, args: argparse.Namespace
) -> dict[str, Any]:
    variant = VARIANTS[variant_name]
    env = BattlesnakeEnv(config, obs=variant.obs)
    program = build_program(env, variant, POLICIES[args.policy], args.turns)
    times, warmup, out = time_program(program, batch, args.repeats, args.seed)
    best = min(times)
    steps = batch * args.turns
    row = {
        "config": name,
        "variant": variant_name,
        "batch": batch,
        "turns": args.turns,
        "num_snakes": config.num_snakes,
        "best_s": best,
        "times_s": times,
        "warmup_s": warmup,
        "steps_per_s": steps / best,
        "agent_steps_per_s": steps * config.num_snakes / best,
    }
    if not variant.reset_only:
        _, tot = out
        row["live_fraction"] = float(np.asarray(tot.live).sum() / steps)
        row["episodes"] = int(np.asarray(tot.episodes).sum())
    return row


# --- Reporting ------------------------------------------------------------------


def _fmt_rate(x: float) -> str:
    return f"{x:,.0f}"


def markdown_table(headers: list[str], rows: list[list[str]], right: set[int]) -> str:
    """GitHub-flavoured table; columns whose index is in ``right`` are right-aligned."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        padded = [
            c.rjust(w) if i in right else c.ljust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        ]
        return "| " + " | ".join(padded) + " |"

    sep = ["-" * (w - 1) + ":" if i in right else "-" * w for i, w in enumerate(widths)]
    return "\n".join([line(headers), "| " + " | ".join(sep) + " |", *map(line, rows)])


def results_table(results: list[dict[str, Any]]) -> str:
    headers = ["config", "variant", "batch", "steps/s", "agent-steps/s"]
    rows = [
        [
            r["config"],
            r["variant"],
            f"{r['batch']:,}",
            _fmt_rate(r["steps_per_s"]),
            _fmt_rate(r["agent_steps_per_s"]),
        ]
        for r in results
    ]
    return markdown_table(headers, rows, right={2, 3, 4})


def autoreset_table(
    results: list[dict[str, Any]], lengths: dict[str, dict[str, float]]
) -> str | None:
    """Autoreset cost per (config, batch), from the sim / sim-no-autoreset / reset-only rows."""
    index = {(r["config"], r["variant"], r["batch"]): r for r in results}
    rows = []
    for (cfg, variant, batch), sim in index.items():
        if variant != "sim":
            continue
        noreset = index.get((cfg, "sim-no-autoreset", batch))
        reset = index.get((cfg, "reset-only", batch))
        if noreset is None:
            continue
        # Time per step: autoreset adds (t_sim - t_noreset) on top of a plain step.
        t_sim, t_no = 1 / sim["steps_per_s"], 1 / noreset["steps_per_s"]
        extra = f"{(t_sim / t_no - 1) * 100:+.0f}%"
        row = [
            cfg,
            f"{batch:,}",
            _fmt_rate(sim["steps_per_s"]),
            _fmt_rate(noreset["steps_per_s"]),
            extra,
        ]
        if reset is not None:
            row.append(f"{(1 / reset['steps_per_s']) / t_no * 100:.0f}%")
        else:
            row.append("-")
        # Only steps that end a game need the fresh state autoreset always computes.
        row.append(f"{100 / lengths[cfg]['mean']:.1f}%" if cfg in lengths else "-")
        rows.append(row)
    if not rows:
        return None
    headers = [
        "config",
        "batch",
        "sim steps/s",
        "no-autoreset steps/s",
        "autoreset overhead",
        "reset / plain step",
        "resets used",
    ]
    return markdown_table(headers, rows, right={1, 2, 3, 4, 5, 6})


def length_table(lengths: dict[str, dict[str, float]]) -> str:
    headers = ["config", "mean", "median", "p90", "max", "finished"]
    rows = [
        [
            cfg,
            f"{s['mean']:.1f}",
            f"{s['median']:.0f}",
            f"{s['p90']:.0f}",
            f"{s['max']}",
            f"{s['finished']}/{s['games']}",
        ]
        for cfg, s in lengths.items()
    ]
    return markdown_table(headers, rows, right={1, 2, 3, 4, 5})


def device_info() -> list[str]:
    devices = jax.devices()
    kinds = sorted({d.device_kind for d in devices})
    lines = [
        f"- jax {jax.__version__}, jaxlib {jaxlib.__version__},"
        f" numpy {np.__version__}, python {platform.python_version()}",
        f"- backend: {jax.default_backend()}, {len(devices)} device(s): {', '.join(kinds)}",
        f"- host: {platform.machine()}, {os.cpu_count()} logical CPUs",
        f"- prng: {jax.config.jax_default_prng_impl}",
    ]
    if hasattr(os, "getloadavg"):
        # Other processes on the host perturb timings; make that visible.
        lines.append(f"- load average at exit: {', '.join(f'{x:.1f}' for x in os.getloadavg())}")
    if jax.default_backend() != "cpu":
        lines.append(f"- first device: {devices[0]}")
    return lines


# --- CLI ------------------------------------------------------------------------


def _csv(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--configs",
        type=_csv,
        default=list(CONFIGS),
        metavar="A,B",
        help=f"comma-separated subset of: {', '.join(CONFIGS)} (default: all)",
    )
    p.add_argument(
        "--batch-sizes",
        type=lambda s: [int(x) for x in _csv(s)],
        default=None,
        metavar="N,N",
        help=f"games per batch (default: {','.join(map(str, DEFAULT_BATCH_SIZES))})",
    )
    p.add_argument(
        "--turns",
        type=int,
        default=None,
        help=f"turns scanned inside each jitted call (default: {DEFAULT_TURNS}; 30 with --quick)",
    )
    p.add_argument(
        "--variants",
        type=_csv,
        default=None,
        metavar="A,B",
        help=f"subset of: {', '.join(VARIANTS)}, or 'all' (default: all for duel, "
        f"{', '.join(OTHER_CONFIG_VARIANTS)} for the other configs)",
    )
    p.add_argument("--policy", choices=list(POLICIES), default="random_legal")
    p.add_argument("--repeats", type=int, default=3, help="timed repeats; best is reported")
    p.add_argument("--quick", action="store_true", help="smoke test: skip batch 8192, fewer turns")
    p.add_argument("--json", metavar="PATH", help="also write all results as JSON")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        p.error(f"unknown config(s) {unknown}; choose from {list(CONFIGS)}")
    if args.variants == ["all"]:
        args.variants = list(VARIANTS)
    unknown = [v for v in args.variants or [] if v not in VARIANTS]
    if unknown:
        p.error(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
    if args.batch_sizes is None:
        args.batch_sizes = [
            b for b in DEFAULT_BATCH_SIZES if not (args.quick and b > QUICK_MAX_BATCH)
        ]
    if args.turns is None:
        args.turns = 30 if args.quick else DEFAULT_TURNS
    return args


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results: list[dict[str, Any]] = []
    lengths: dict[str, dict[str, float]] = {}

    for name in args.configs:
        config = CONFIGS[name]
        variants = args.variants or (list(VARIANTS) if name == "duel" else OTHER_CONFIG_VARIANTS)
        log(f"[{name}] game length ...")
        lengths[name] = measure_game_length(config, POLICIES[args.policy], args.seed)
        for variant in variants:
            for batch in args.batch_sizes:
                row = run_cell(name, config, variant, batch, args)
                results.append(row)
                log(
                    f"[{name}] {variant:<17} batch {batch:>5}: {row['steps_per_s']:>12,.0f} steps/s"
                    f"  (compile+warmup {row['warmup_s']:.1f}s)"
                )
        jax.clear_caches()  # keep compiled programs of finished configs from piling up

    print("# slinky throughput benchmark\n")
    print("\n".join(device_info()))
    print(f"- policy: {args.policy}, turns per call: {args.turns}, best of {args.repeats}")
    print(
        "- configs: "
        + "; ".join(
            f"{n}: {c.width}x{c.height} {c.ruleset.value} {c.num_snakes} snakes"
            for n, c in ((n, CONFIGS[n]) for n in args.configs)
        )
    )
    print("\n## Throughput\n")
    print(results_table(results))
    print("\n`steps/s` = games advanced one turn per second; `agent-steps/s` = steps/s x snakes.")
    print("Rows for `reset-only` count resets instead of steps.")
    print("`sim-no-autoreset` games stay done once over; they still pay for a full step.")
    overhead = autoreset_table(results, lengths)
    if overhead:
        print("\n## Autoreset cost\n")
        print(overhead)
        print("\n`autoreset overhead` = extra time per step of `sim` over `sim-no-autoreset`;")
        print("`reset / plain step` = time of one stand-alone `env.reset` relative to one step;")
        print("`resets used` = share of steps that end a game (1 / mean game length), i.e. the")
        print("fraction of the fresh states `step_autoreset` computes every step that is kept.")
    print(f"\n## Game length ({args.policy} policy, {LENGTH_BATCH} games, {LENGTH_TURNS} turns)\n")
    print(length_table(lengths))

    if args.json:
        payload = {
            "meta": {
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "devices": [str(d) for d in jax.devices()],
                "device_kinds": sorted({d.device_kind for d in jax.devices()}),
                "cpu_count": os.cpu_count(),
                "python": platform.python_version(),
                "args": vars(args),
            },
            "results": results,
            "game_length": lengths,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        log(f"wrote {args.json}")


if __name__ == "__main__":
    main()
