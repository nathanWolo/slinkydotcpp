"""Strength benchmark: how often agents beat fixed opponents (MCTS score vs simulations).

Plays every pair ``A x B`` of two agent lists with :func:`slinky.evaluate.run_match`
and appends one JSON line per finished matchup to ``benchmarks/results/strength.jsonl``.
``--table`` turns that file into markdown tables. The score is A's win rate with a
draw worth 1/2, with a 95% confidence interval; seats alternate (game ``g`` puts A in
seat ``g % N``), so neither agent gets the better starting seat more often. Games are cut
off after ``--max-turns`` turns (default 500) and then count as draws (also reported as
``truncated``). The rules are the official ones with no turn limit; the cut is imposed by
the harness only, and the agents do not know about it.

Agents (``--a`` and ``--b`` take comma-separated lists) are names from the registry in
:mod:`slinky.agents`: ``random_legal``, ``random``, ``heuristic``, ``dqn`` (the checkpoint
in ``baselines/checkpoints/dqn-duel-seed0``) or ``dqn:<run dir>``, ``ppo`` (greedy; the
checkpoint in ``baselines/checkpoints/ppo-duel-seed0``), ``ppo-sample`` or ``ppo:<run dir>``,
and ``mcts-<sims>[-shorthand...][:field=value...]``, e.g. ``mcts-64-rollout``,
``mcts-256-rm-c0.5`` or ``mcts-128:exploration=0.5`` (the shorthands are listed by
``--help``). Names are printed and stored in their canonical form, which lists only the
non-default settings (``mcts-64-rollout`` becomes ``mcts-64-rollout10``). The default
MCTS values a draw at -0.5 *inside the search* (``contempt``); the benchmark score still
counts a draw as 1/2.

Usage::

    python benchmarks/strength.py --a mcts-16,mcts-64,mcts-256 --b random_legal,heuristic,dqn \\
        --games 256 --seed 0 --workers 4
    python benchmarks/strength.py --table                 # tables from the results file
    python benchmarks/strength.py --a mcts-64 --b heuristic --games 256 --dry-run
    python benchmarks/strength.py --a mcts-1024 --b heuristic --games 256 --estimate

Design, in the order a matchup is run:

**Identity and resume.** A matchup's identity is a hash of both agents (canonical name
and *full* config: every ``MCTSConfig`` field, the heuristic weights, the checkpoint's
parameter hash), the game rules, ``--max-turns``, the requested number of games, the
seed and ``--target-ci`` (with ``--round-games``, which sets its stopping points). A
matchup whose identity is already in the output file is skipped, so an interrupted or
crashed sweep resumes by running the same command. The identity covers *configs*, not
*code*: changing an ``MCTSConfig`` or ``Weights`` default changes it, but a fix to the
search, the heuristic or the rules does not. Each line records the code that produced
it (``git_commit``, ``git_dirty`` and ``code_fingerprint``, a hash of the ``slinky``
modules the process loaded, ``baselines/dqn.py`` and this script, all captured when the
process started); the sweep warns when it skips lines made by other code, and
``--rerun`` plays the selected matchups again (the new line is preferred by ``--table``;
the old one stays in the file). A line is
written only when its matchup is complete (one ``write`` under an exclusive file lock).
A line torn by a kill during that write is skipped when reading and terminated before
the next append. Two sweeps may share a file: before each matchup the file is read again
and a matchup finished by another process in the meantime is skipped (both would run a
matchup that neither has finished yet).

**Randomness.** A matchup's key is ``fold_in(key(seed), h)``, where ``h`` is the first 31
bits of ``sha256("<a name>\\0<b name>")``. Game ``g`` of a matchup is game ``g`` of
``run_match`` with that key: it depends only on the seed, the two canonical names and
``g``, never on ``--slots``, ``--round-games``, ``--workers``, run order or which other
matchups are in the sweep (one exception: with ``--slots 1`` the DQN's Q-values can differ
in the last bit, which has not been seen to change a move). So the same command reproduces
every game exactly, and a larger run with the same seed (and the same config and code)
contains the games of a smaller one.

**Slots, memory and rounds.** ``run_match`` simulates ``S`` game slots at once and
starts the next game in a slot as soon as its game ends, so no slot waits for the
longest game (lengths are heavy-tailed: mean 120 and max 500 turns for MCTS against the
heuristic). Only the end of a match leaves slots idle; ``slot_utilization`` records the
fraction of slot-turns that played counted games (about 0.8 with 8 games per slot). On
one pinned core the cost per move falls up to about 32 slots and rises again at 64, so
``S`` is ``--slots`` (default 32), capped by the number of games in a round and by the
memory budget ``--mem-mb`` (default 1024 MB per process): ``S <= mem_mb / (2 * tree bytes
per game)``, where the tree of an MCTS agent holds ``sims + 1`` nodes of about 1.04 KB in
the 11x11 duel (0.9 KB of it the game state; 1.3 KB with ``rm``; 2.6 KB with 4 snakes),
summed over the MCTS agents in the matchup, and the factor 2 leaves room for XLA
temporaries. With 32 slots the cap only binds above about 15000 simulations per
matchup; the peak resident memory of the process is recorded so the model can be
checked. The number of games is rounded up to a multiple of ``N`` snakes so seats stay
balanced (both numbers are recorded). A matchup is one round unless ``--target-ci`` is
set: then rounds of about ``--round-games`` games (a multiple of ``N``; default 64) are
played, each a range of game ids, and the stop test runs after each one. Every round
ends with idle slots, so rounds cost some utilisation. Progress lines come from inside
the loop, at most every 15 seconds.

**Workers.** ``--workers N`` runs matchups in ``N`` processes at once, each pinned to its
own core (``os.sched_setaffinity``, set before the process imports jax; the processes
are started with ``spawn``) and each playing one matchup. Matchups are handed out most
expensive first, to the next free core. XLA's intra-op threads burn about 1.7x the CPU
time of one core for no wall-clock gain on this workload, while ``N`` pinned processes
give nearly ``N`` times the throughput, so ``--workers 4`` on a 4-core machine is the
fast path (results are bit-identical either way). Compilation is slower on one core
(10-25 s instead of about 6 s), which only matters for sweeps of short matchups.
Workers append their own lines under
the file lock. Ctrl-C stops the workers at once: finished matchups are saved, the
running ones are lost and run again on the next invocation. With ``--workers 1`` (the
default) matchups run in this process, cheapest first, without pinning; Ctrl-C then
takes effect when the current jitted call returns.

**Timing.** Before the first round the script runs the matchup for 1 turn (compilation),
again for 1 turn and for 9 turns, with every slot busy; the difference is the cost of one
loop iteration (every slot advancing one turn), recorded as
``seconds_per_iteration_probe``. ``seconds_per_move_probe`` is that divided by ``S``:
the cost of one move of one game (A's search, B's move and the environment step) in
the first turns of a game. ``wall_seconds`` is the whole matchup; ``play_seconds`` leaves
out compilation and the probes; ``seconds_per_game_turn = play_seconds / (sum of game
lengths)`` is the *effective* cost, about the probe divided by ``slot_utilization``
(later turns can cost more than the first ones). All are throughputs of one process
(of one pinned core with ``--workers``; ``load_avg_start`` shows how busy the machine
was), not the latency of one search.

**Results.** One JSON object per line (``SCHEMA``): ``a``, ``b``, ``a_config``, ``b_config``,
``game``, ``max_turns``, ``games_requested``, ``games``, ``wins``, ``draws``, ``losses``,
``truncated`` (games cut off by ``--max-turns`` with both snakes alive; counted as draws,
so they are inside ``draws``), ``score``, ``ci95`` (the normal-approximation half-width of
``slinky.evaluate``), ``ci95_low`` and ``ci95_high`` (the interval the tables show, see
below), ``mean_turns``, the seconds fields, ``loop_iterations``, ``slot_utilization``,
``slots``, ``rounds``, ``round_games``, ``seed``, ``git_commit``, ``git_dirty``
(uncommitted changes under ``src/``, ``baselines/`` or ``benchmarks/``),
``code_fingerprint``, ``worker_core``, ``jax_version``, ``date`` and the identity hashes
(``matchup_id``, ``config_id``). ``rounds_detail`` lists, per round, the columns named in
``rounds_detail_columns``. Lines of an older ``schema`` are ignored (their games were
drawn differently).

**Intervals.** The tables, progress lines and ``ci95_low``/``ci95_high`` use the Wilson
score interval of the score as a proportion of ``n`` games. A game's score lies in
``[0, 1]``, so its variance is at most ``p (1 - p)``, with equality when there are no
draws: draws only make the interval conservative. Unlike the normal (Wald) interval it is
monotonic in the outcomes, never has zero width, and keeps close to 95% coverage near 0
and 1 (where Wald gives 0.86-0.90 at 160-256 games). ``--target-ci`` stops when its
half-width ``(high - low) / 2`` reaches the target. Stopping on the data is a sequential
rule: a matchup tends to stop after a lucky round, so near 0 or 1 the scores of stopped
matchups are slightly optimistic (simulated at true scores of 0.95-0.97: +0.004 with
rounds of 64 games, +0.002 with rounds of 256; the Wilson interval still covered 96%).
Use a fixed ``--games`` for headline or near-saturated cells; tables mark stopped
matchups.

**Tables.** ``--table`` groups lines by ``(a, b)`` and uses the group with the newest
line when several configs exist for a pair (the others are counted in a warning). Within
a group only the line with the most games of each seed is used (it contains the games of
the smaller ones), and the seeds are pooled by adding the counts, so extra seeds can top
up a cell. The interval is recomputed from the pooled counts. A name can still mean
different configs in different cells (a default changed and only some pairs were run
again); such cells are marked and named in a warning, as are tables whose lines come
from more than one code fingerprint.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import multiprocessing
import multiprocessing.connection
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax

from slinky import agents as registry
from slinky import mcts, observations  # noqa: F401  (loaded now: in the code fingerprint)
from slinky.evaluate import run_match
from slinky.types import GameConfig

try:
    import fcntl
except ImportError:  # not on POSIX: no file locking
    fcntl = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "strength.jsonl"
SCHEMA = 2  # 1: fixed batches with batch-dependent random streams (ignored)
Z95 = 1.959963984540054
PROBE_TURNS = 8  # loop iterations timed to estimate the cost of a move
MIN_GAMES_TO_STOP = 64  # --target-ci never stops before this many games
TREE_SAFETY = 2.0  # memory model: XLA temporaries and double buffering
PROGRESS_SECONDS = 15.0  # at most one progress line per this many seconds
# --dry-run's rough model of one game-turn on one pinned core of a 2.1 GHz Xeon, fitted to
# mcts-64 ... mcts-24000 against the heuristic with 32 slots: 0.06 ms plus 0.0055 ms per
# simulation of the matchup's searches, growing by sims / 48000 (bigger trees cost more per
# node), plus 0.25 ms per DQN move and about 0.15 per PPO move (smaller network). Games
# against random agents are short. --estimate measures instead.
ROUGH_MS = {"fixed": 0.06, "per_sim": 0.0055, "sim_scale": 48000.0, "dqn": 0.25, "ppo": 0.15}
ROUGH_TURNS_PER_GAME = 120
ROUGH_TURNS_VS_RANDOM = 40
COMPILE_SECONDS = 10.0  # about what compiling a matchup takes
EXIT_SKIPPED = 3  # worker exit code: another process finished the matchup first
ROUNDS_DETAIL_COLUMNS = ["wins", "draws", "losses", "truncated", "turns", "seconds", "iterations"]
REQUIRED = frozenset(
    ("a", "b", "matchup_id", "config_id", "seed", "games", "wins", "draws", "losses")
    + ("mean_turns", "date")
)


class SpecError(ValueError):
    """A bad agent name or option."""


def log(message: str = "") -> None:
    print(message, file=sys.stderr, flush=True)


# --- Agents and provenance -----------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Agent:
    """A registry agent and everything that defines its behaviour (``config`` is JSON data)."""

    spec: registry.AgentSpec
    config: dict[str, Any]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def sims(self) -> int:
        return self.spec.sims


def parse_spec(text: str) -> registry.AgentSpec:
    try:
        return registry.parse_agent(text)
    except ValueError as e:
        raise SpecError(str(e)) from None


def load_agent(text: str) -> Agent:
    """The agent named ``text``, with its full config (a DQN's checkpoint is read)."""
    spec = parse_spec(text)
    try:
        return Agent(spec, registry.agent_config(spec))
    except (ValueError, FileNotFoundError) as e:
        raise SpecError(f"{spec.name}: {e}") from None


def parse_agents(text: str) -> list[Agent]:
    agents: dict[str, Agent] = {}
    for item in text.split(","):
        if item.strip():
            spec = parse_spec(item)
            if spec.name not in agents:
                agents[spec.name] = load_agent(spec.name)
    return list(agents.values())


def names_of(text: str | None) -> set[str] | None:
    """Canonical agent names in a comma-separated list (None: no filter; no files read)."""
    if not text:
        return None
    return {parse_spec(item).name for item in text.split(",") if item.strip()}


@dataclasses.dataclass(frozen=True)
class Provenance:
    """The code a process runs, captured once when it starts."""

    git_commit: str
    git_dirty: bool
    code_fingerprint: str
    code_files: int


def git_info() -> tuple[str, bool]:
    def run(*args: str) -> str:
        out = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=20, check=True
        )
        return out.stdout.strip()

    try:
        # The results files are not code: leave them out, or a sweep would mark itself dirty.
        status = run(
            "status",
            "--porcelain",
            "--",
            "src",
            "baselines",
            "benchmarks",
            ":(exclude)benchmarks/results",
        )
        return run("rev-parse", "HEAD"), bool(status)
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


def code_fingerprint() -> tuple[str, int]:
    """Hash of the slinky modules this process has loaded, the baselines and this file.

    Returns ``(hash, number of files)``. Every module an agent can need is imported at
    startup, so the set of files does not depend on the agents. Files are keyed by module
    name, so checkouts in different directories with the same code agree.
    """
    files = {"benchmarks/strength.py": Path(__file__)}
    for name, script in (
        ("baselines_dqn", registry.DQN_SCRIPT),
        ("baselines_ppo", registry.PPO_SCRIPT),
    ):
        if script.is_file():
            files[name] = script
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and (name == "slinky" or name.startswith("slinky.")):
            files[name] = Path(path)
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + hashlib.sha256(files[name].read_bytes()).digest())
    return digest.hexdigest()[:16], len(files)


@functools.cache
def provenance() -> Provenance:
    """Captured on the first call: call it at startup, after loading the agents' modules."""
    commit, dirty = git_info()
    return Provenance(commit, dirty, *code_fingerprint())


# --- Matchups: identity, key, plan ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Settings:
    """Everything shared by the matchups of one invocation."""

    game: GameConfig
    max_turns: int
    seed: int
    target_ci: float | None
    slots: int
    round_games: int
    mem_mb: float


@dataclasses.dataclass(frozen=True, eq=False)
class Matchup:
    a: Agent
    b: Agent
    games: int  # requested

    @property
    def cost(self) -> float:
        """Seconds for scheduling and the sweep's ETA: the rough model plus compilation."""
        return self.rough_seconds() + COMPILE_SECONDS

    def rough_seconds(self) -> float:
        """Play time on one pinned core by the rough model of ``ROUGH_MS``."""
        kinds = (self.a.spec.kind, self.b.spec.kind)
        sims = self.a.sims + self.b.sims
        ms = ROUGH_MS["fixed"] + ROUGH_MS["per_sim"] * sims * (1 + sims / ROUGH_MS["sim_scale"])
        ms += ROUGH_MS["dqn"] * kinds.count("dqn") + ROUGH_MS["ppo"] * kinds.count("ppo")
        random = {"random", "random_legal"} & set(kinds)
        turns = ROUGH_TURNS_VS_RANDOM if random else ROUGH_TURNS_PER_GAME
        return self.games * turns * ms / 1e3


def game_dict(game: GameConfig) -> dict[str, Any]:
    d = dataclasses.asdict(game)
    d["ruleset"] = game.ruleset.value
    return d


def _digest(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def config_id(m: Matchup, s: Settings) -> str:
    """Hash of everything that defines the *kind* of result (not how many games or seed)."""
    return _digest([m.a.name, m.a.config, m.b.name, m.b.config, game_dict(s.game), s.max_turns])


def matchup_id(m: Matchup, s: Settings) -> str:
    """Hash that identifies a requested run; a result with the same id is not run again."""
    stops = [s.round_games] if s.target_ci is not None else []  # where --target-ci may stop
    return _digest([config_id(m, s), m.games, s.seed, s.target_ci, *stops])


def matchup_key(seed: int, a_name: str, b_name: str) -> jax.Array:
    """PRNG key from the seed and the two names only (see the module docstring)."""
    digest = hashlib.sha256(f"{a_name}\0{b_name}".encode()).digest()
    return jax.random.fold_in(jax.random.key(seed), int.from_bytes(digest[:4], "big") >> 1)


@functools.cache
def state_bytes(game: GameConfig) -> int:
    env = registry.make_env(game, False)
    state = jax.eval_shape(env.init_state, jax.random.key(0))
    return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(state))


def tree_bytes_per_game(agent: Agent, game: GameConfig) -> int:
    """Estimated search-tree bytes of one game (0 for agents without a tree).

    Per node: the game state, the joint-action children table, the legal mask,
    the leaf value, and per player and own move a visit count and a value sum
    (regret matching and UCB1-Tuned keep more). In the 11x11 duel this gives
    1.04 KB (1.30 KB with regret matching), as measured for ``slinky.mcts``.
    """
    cfg, n = agent.spec.mcts, game.num_snakes
    if cfg is None:
        return 0
    j = 4**n
    node = state_bytes(game) + 4 * j + 4 * n + 1 + 4 * n + 32 * n
    if cfg.ucb1_tuned:
        node += 16 * n
    if cfg.selection == "rm":
        node += 32 * n + 4 * j + 4 * j * n
    sims = cfg.num_simulations
    return (sims + 1) * node + sims * n * 4 * 4  # plus the pre-drawn selection noise


@dataclasses.dataclass(frozen=True)
class Plan:
    rounds: tuple[int, ...]  # games per round; game ids run on from round to round
    slots: int
    memory_limited: bool  # --mem-mb made the slots fewer than they would be otherwise
    tree_mb: float  # estimated search-tree memory of all slots

    @property
    def games(self) -> int:
        return sum(self.rounds)


def plan_matchup(m: Matchup, s: Settings) -> Plan:
    """Rounds and slots for a matchup (the rule is in the module docstring)."""
    n = s.game.num_snakes
    games = -(-m.games // n) * n  # whole seat rotations
    count = 1 if s.target_ci is None else -(-games // max(s.round_games, n))
    while True:  # equal rounds of whole seat rotations, the last one possibly shorter
        per = -(-games // count)
        per = -(-per // n) * n
        last = games - per * (count - 1)
        if last > 0:
            break
        count -= 1
    rounds = (per,) * (count - 1) + (last,)
    per_game = sum(tree_bytes_per_game(x, s.game) for x in (m.a, m.b))
    mem_cap = max(int(s.mem_mb * 2**20 / (TREE_SAFETY * per_game)), 1) if per_game else 10**9
    wanted = min(s.slots, min(rounds))
    slots = min(wanted, mem_cap)
    return Plan(rounds, slots, slots < wanted, slots * per_game / 2**20)


# --- Statistics ----------------------------------------------------------------------


def outcome_stats(wins: int, draws: int, losses: int) -> tuple[int, float, float]:
    """``(games, score, ci95)`` exactly as ``slinky.evaluate`` computes them (Wald)."""
    n = wins + draws + losses
    if n == 0:
        return 0, float("nan"), float("nan")
    score = (wins + 0.5 * draws) / n
    var = max(wins + 0.25 * draws - n * score * score, 0.0) / (n - 1) if n > 1 else 0.0
    return n, score, Z95 * math.sqrt(var / n)


def score_interval(wins: float, draws: float, losses: float) -> tuple[float, float, float]:
    """``(score, low, high)``: the Wilson score interval of the module docstring."""
    n = wins + draws + losses
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = (wins + 0.5 * draws) / n
    z2 = Z95 * Z95
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return p, 0.0 if p == 0 else centre - half, 1.0 if p == 1 else centre + half


def interval_text(wins: int, draws: int, losses: int) -> str:
    score, low, high = score_interval(wins, draws, losses)
    return f"{score:.3f} [{low:.3f}, {high:.3f}]"


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**20 if sys.platform == "darwin" else 2**10)


def fmt_duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    if seconds < 1:
        return f"{1e3 * seconds:.0f}ms"
    if seconds < 90:
        return f"{seconds:.1f}s" if seconds < 10 else f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


# --- The results file ----------------------------------------------------------------


def read_records(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """``(records, unreadable lines, lines of an older schema)``."""
    records, bad, old = [], 0, 0
    if not path.is_file():
        return records, bad, old
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec.get("schema"), int) and rec["schema"] < SCHEMA:
                    old += 1
                    continue
                if rec["schema"] != SCHEMA or not REQUIRED.issubset(rec):
                    raise ValueError("not a result line")
                records.append(rec)
            except (ValueError, KeyError, TypeError, AttributeError):
                bad += 1
    return records, bad, old


def done_ids(path: Path) -> set[str]:
    return {r["matchup_id"] for r in read_records(path)[0]}


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one line atomically (exclusive lock, torn last line repaired first).

    SIGINT and SIGTERM are held back during the write, so stopping a worker
    can't tear the line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, sort_keys=False) + "\n").encode()
    held = {signal.SIGINT, signal.SIGTERM}
    masked = hasattr(signal, "pthread_sigmask")
    if masked:
        signal.pthread_sigmask(signal.SIG_BLOCK, held)
    try:
        with open(path, "a+b") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":  # a previous writer was killed mid-line
                    data = b"\n" + data
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    finally:
        if masked:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, held)


# --- Running a matchup ---------------------------------------------------------------


@dataclasses.dataclass
class Prepared:
    env: Any
    policy_a: Any
    policy_b: Any
    key: jax.Array
    slots: int
    compile_seconds: float = 0.0
    seconds_per_iteration: float = 0.0  # from the probe

    def play(self, key: jax.Array, games: int, max_turns: int, first_game: int = 0, progress=None):
        return run_match(
            self.env, self.policy_a, self.policy_b, key, games, max_turns, self.slots,
            first_game=first_game, progress=progress,
        )  # fmt: skip


def prepare(m: Matchup, s: Settings, plan: Plan) -> Prepared:
    """Build the policies, compile the match and time a few loop iterations (``max_turns``
    and the number of games are traced, so short calls reuse the program of the real rounds)."""
    a = registry.make_agent(m.a.spec, s.game)
    b = registry.make_agent(m.b.spec, s.game)
    env = registry.make_env(s.game, a.needs_obs or b.needs_obs)
    key = matchup_key(s.seed, m.a.name, m.b.name)
    prep = Prepared(env, a.policy, b.policy, key, plan.slots)
    warm = jax.random.fold_in(prep.key, 2**31 - 1)  # a key no counted game uses

    def timed(turns: int) -> float:
        t0 = time.perf_counter()
        prep.play(warm, plan.slots, turns)
        return time.perf_counter() - t0

    prep.compile_seconds = timed(1)
    base = timed(1)
    probe = timed(PROBE_TURNS + 1)
    per_turn = (probe - base) / PROBE_TURNS
    prep.seconds_per_iteration = per_turn if per_turn > 0 else probe / (PROBE_TURNS + 1)
    return prep


def progress_reporter(prefix: str, games: int) -> Callable[[int, int], None]:
    """A ``run_match`` progress hook that logs at most every ``PROGRESS_SECONDS``."""
    t0 = time.perf_counter()
    last = [t0]

    def report(finished: int, iterations: int) -> None:
        now = time.perf_counter()
        if now - last[0] < PROGRESS_SECONDS or finished >= games:
            return
        last[0] = now
        eta = (now - t0) * (games - finished) / max(finished, 1)
        log(
            f"{prefix} {finished}/{games} games in {fmt_duration(now - t0)} ({iterations} "
            f"iterations), rough ETA {fmt_duration(eta)} (short games finish first)"
        )

    return report


def run_matchup(m: Matchup, s: Settings, plan: Plan, tag: str) -> dict[str, Any]:
    """Play all rounds of a matchup and return its result record."""
    t_start = time.perf_counter()
    load_avg = round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None
    prov = provenance()
    prep = prepare(m, s, plan)
    per_move = prep.seconds_per_iteration / plan.slots
    rough = per_move * plan.games * ROUGH_TURNS_PER_GAME / 0.8
    log(
        f"{tag}   compile {fmt_duration(prep.compile_seconds)}; probe {1e3 * per_move:.3f} ms "
        f"per move ({plan.slots} slots), so about {fmt_duration(rough)} at "
        f"{ROUGH_TURNS_PER_GAME} turns per game"
    )
    w = d = lo = trunc = turns = iterations = 0
    rounds_detail: list[list[float]] = []
    play_seconds = 0.0
    stopped_early = False
    first = 0
    for r, size in enumerate(plan.rounds):
        t0 = time.perf_counter()
        progress = progress_reporter(f"{tag}   ... round {r + 1}:", size)
        run = prep.play(prep.key, size, s.max_turns, first_game=first, progress=progress)
        first += size
        dt = time.perf_counter() - t0
        play_seconds += dt
        res = run.result
        round_turns = round(res.mean_turns * res.num_games)
        w, d, lo = w + res.wins, d + res.draws, lo + res.losses
        trunc, turns = trunc + res.truncated, turns + round_turns
        iterations += run.iterations
        rounds_detail.append(
            [res.wins, res.draws, res.losses, res.truncated, round_turns, round(dt, 3),
             run.iterations]
        )  # fmt: skip
        n = w + d + lo
        eta = (plan.games - n) * play_seconds / n
        log(
            f"{tag}   round {r + 1}/{len(plan.rounds)}: {res.num_games} games in "
            f"{fmt_duration(dt)} (slots {100 * run.utilization:.0f}% busy) | total {n} games, "
            f"W/D/L {w}/{d}/{lo} ({trunc} truncated), score {interval_text(w, d, lo)} | "
            f"ETA {fmt_duration(eta)}"
        )
        if s.target_ci is not None and r + 1 < len(plan.rounds) and n >= MIN_GAMES_TO_STOP:
            _, low, high = score_interval(w, d, lo)
            if (high - low) / 2 <= s.target_ci:
                log(
                    f"{tag}   stopping early: CI half-width {(high - low) / 2:.3f} <= {s.target_ci}"
                )
                stopped_early = True
                break
    n, score, ci = outcome_stats(w, d, lo)
    _, low, high = score_interval(w, d, lo)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "schema": SCHEMA,
        "matchup_id": matchup_id(m, s),
        "config_id": config_id(m, s),
        "a": m.a.name,
        "b": m.b.name,
        "a_config": m.a.config,
        "b_config": m.b.config,
        "game": game_dict(s.game),
        "max_turns": s.max_turns,
        "games_requested": m.games,
        "games": n,
        "wins": w,
        "draws": d,
        "losses": lo,
        "truncated": trunc,
        "score": score,
        "ci95": ci,
        "ci95_low": low,
        "ci95_high": high,
        "mean_turns": turns / n,
        "wall_seconds": time.perf_counter() - t_start,
        "compile_seconds": prep.compile_seconds,
        "play_seconds": play_seconds,
        "loop_iterations": iterations,
        "slot_utilization": turns / (iterations * plan.slots),
        "seconds_per_game_turn": play_seconds / max(turns, 1),
        "seconds_per_iteration_probe": prep.seconds_per_iteration,
        "seconds_per_move_probe": per_move,
        "slots": plan.slots,
        "rounds": len(plan.rounds),
        "rounds_played": len(rounds_detail),
        "round_games": list(plan.rounds),
        "target_ci": s.target_ci,
        "stopped_early": stopped_early,
        "slots_memory_limited": plan.memory_limited,
        "est_tree_mb": plan.tree_mb,
        "load_avg_start": load_avg,
        "peak_rss_mb": peak_rss_mb(),
        "worker_core": affinity[0] if affinity and len(affinity) == 1 else None,
        "seed": s.seed,
        "git_commit": prov.git_commit,
        "git_dirty": prov.git_dirty,
        "code_fingerprint": prov.code_fingerprint,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "cpu_count": os.cpu_count(),
        "cpus_usable": len(affinity) if affinity else None,
        "python": platform.python_version(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rounds_detail": rounds_detail,
        "rounds_detail_columns": ROUNDS_DETAIL_COLUMNS,
    }


def describe_record(r: dict[str, Any]) -> str:
    return (
        f"score {interval_text(r['wins'], r['draws'], r['losses'])} over {r['games']} games "
        f"(W/D/L {r['wins']}/{r['draws']}/{r['losses']}) in {fmt_duration(r['wall_seconds'])} "
        f"(compile {fmt_duration(r['compile_seconds'])}), "
        f"{1e3 * r['seconds_per_game_turn']:.3f} ms per game-turn, slots "
        f"{100 * r['slot_utilization']:.0f}% busy, peak RSS {r['peak_rss_mb']:.0f} MB"
    )


def execute(m: Matchup, s: Settings, out: Path, known: set[str], tag: str) -> dict[str, Any] | None:
    """Run one matchup and append its line; None if another process finished it first.

    ``known`` holds the ids that were in the file when the sweep started: with
    ``--rerun`` they don't count as finished.
    """
    if matchup_id(m, s) in done_ids(out) - known:
        log(f"{tag}: finished by another process, skipping")
        return None
    plan = plan_matchup(m, s)
    log(f"{tag} | {describe_plan(m, plan)}")
    record = run_matchup(m, s, plan, tag)
    append_record(out, record)
    log(f"{tag} done: {describe_record(record)}")
    return record


# --- Sweep ---------------------------------------------------------------------------


def parse_overrides(text: str | None) -> dict[tuple[str, str | None], int]:
    """``--games-for`` as ``{(a, None | b): games}``; items are ``A=N`` or ``A@B=N``."""
    out: dict[tuple[str, str | None], int] = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        names, _, count = item.rpartition("=")
        if not names or not count.strip().isdigit() or int(count) < 1:
            raise SpecError(f"--games-for item {item!r}: expected A=N or A@B=N with N >= 1")
        a, _, b = names.partition("@")
        out[(parse_spec(a).name, parse_spec(b).name if b else None)] = int(count)
    return out


def games_for(a: str, b: str, default: int, overrides: dict[tuple[str, str | None], int]) -> int:
    """Pair override if any, else the smaller of the overrides of A and B, else ``default``."""
    if (a, b) in overrides:
        return overrides[(a, b)]
    return min([overrides[k] for k in ((a, None), (b, None)) if k in overrides] or [default])


def build_matchups(
    a_agents: Iterable[Agent], b_agents: Iterable[Agent], games: int, overrides: dict
) -> list[Matchup]:
    """All pairs, cheapest first (so a sweep that is stopped early has its cheap cells)."""
    b_agents = list(b_agents)
    pairs = [
        Matchup(a, b, games_for(a.name, b.name, games, overrides))
        for a in a_agents
        for b in b_agents
    ]
    names = {x.name for m in pairs for x in (m.a, m.b)}
    for a, b in overrides:
        if a not in names or (
            b is not None and (a, b) not in {(m.a.name, m.b.name) for m in pairs}
        ):
            log(f"warning: --games-for {a}{'@' + b if b else ''} matches no matchup")
    return sorted(pairs, key=lambda m: m.cost)  # stable: ties keep the command-line order


def describe_plan(m: Matchup, plan: Plan) -> str:
    sizes = Counter(plan.rounds)
    shape = " + ".join(f"{k} x {v}" if k > 1 else f"{v}" for v, k in sizes.items())
    rounds = f" in {len(plan.rounds)} rounds ({shape})" if len(plan.rounds) > 1 else ""
    extra = f", search trees ~{plan.tree_mb:.3g} MB" if plan.tree_mb else ""
    limit = " (slots cut by --mem-mb)" if plan.memory_limited else ""
    return (
        f"{m.games} games requested, {plan.games} played{rounds}, {plan.slots} slots{extra}{limit}"
    )


def report_skipped(records: list[dict[str, Any]], skipped: set[str]) -> None:
    """Warn when skipped matchups were produced by other code than this process runs."""
    prov = provenance()
    other = [
        r for r in records
        if r["matchup_id"] in skipped and r.get("code_fingerprint") != prov.code_fingerprint
    ]  # fmt: skip
    if other:
        commits = sorted({f"{r.get('git_commit', 'unknown')[:10]}" for r in other})
        log(
            f"warning: {len({r['matchup_id'] for r in other})} of the skipped matchups were "
            f"produced by other code (commits {', '.join(commits)}; now {prov.git_commit[:10]}"
            f"{'+uncommitted' if prov.git_dirty else ''}): pass --rerun to play them again"
        )


def sweep(s: Settings, matchups: list[Matchup], out: Path, workers: int, rerun: bool) -> int:
    prov = provenance()
    log(
        f"strength sweep: {len(matchups)} matchups, jax {jax.__version__} "
        f"({jax.default_backend()}, {os.cpu_count()} cpus), commit {prov.git_commit[:10]}"
        f"{'+uncommitted' if prov.git_dirty else ''}, code {prov.code_fingerprint}, seed "
        f"{s.seed}, max_turns {s.max_turns}, workers {workers}, out {out}"
    )
    records = read_records(out)[0]
    known = {r["matchup_id"] for r in records}
    ids = {m: matchup_id(m, s) for m in matchups}
    todo = matchups if rerun else [m for m in matchups if ids[m] not in known]
    skipped = ({ids[m] for m in matchups} & known) - {ids[m] for m in todo}
    rerun_note = f" ({len(set(ids.values()) & known)} again, --rerun)" if rerun else ""
    log(f"{len(skipped)} already in the results file, {len(todo)} to run{rerun_note}")
    report_skipped(records, skipped)
    if workers > 1 and todo:
        return sweep_workers(s, todo, out, known, workers)
    total_cost, done_cost, failures = sum(m.cost for m in todo), 0.0, []
    t_sweep = time.perf_counter()
    for i, m in enumerate(todo, 1):
        tag = f"[{i}/{len(todo)}] {m.a.name} vs {m.b.name}"
        try:
            execute(m, s, out, known, tag)
        except Exception as e:  # keep the sweep going; the matchup is retried on the next run
            log(f"{tag} FAILED: {type(e).__name__}: {e}")
            failures.append(f"{m.a.name} vs {m.b.name}")
        jax.clear_caches()  # compiled programs of finished matchups are never reused
        done_cost += m.cost
        elapsed = time.perf_counter() - t_sweep
        rough = elapsed * (total_cost - done_cost) / done_cost if done_cost else float("nan")
        log(
            f"sweep: {i}/{len(todo)} matchups, elapsed {fmt_duration(elapsed)}, rough ETA "
            f"{fmt_duration(rough)} (cost-weighted)"
        )
    if failures:
        log(f"FAILED matchups (rerun to retry): {', '.join(failures)}")
        return 1
    log(f"sweep finished in {fmt_duration(time.perf_counter() - t_sweep)}")
    return 0


# --- Worker processes ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Task:
    """One matchup for a worker process (agents by name: policies don't pickle)."""

    a: str
    b: str
    games: int
    matchup_id: str
    settings: Settings
    out: str
    known: frozenset[str]
    tag: str


def worker_main(task: Task) -> None:
    """Entry point of a worker process; its exit code reports the outcome."""
    provenance()  # the code this process loaded
    try:
        m = Matchup(load_agent(task.a), load_agent(task.b), task.games)
        if matchup_id(m, task.settings) != task.matchup_id:
            raise SpecError("the agents' configs changed since the sweep started")
        record = execute(m, task.settings, Path(task.out), set(task.known), task.tag)
    except Exception as e:
        log(f"{task.tag} FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
    sys.exit(0 if record is not None else EXIT_SKIPPED)


def worker_cores(workers: int) -> list[int | None]:
    if not hasattr(os, "sched_setaffinity"):
        log("warning: no os.sched_setaffinity here; workers are not pinned")
        return [None] * workers
    cores = sorted(os.sched_getaffinity(0))
    if workers > len(cores):
        log(f"warning: {workers} workers on {len(cores)} cores; some share a core")
    return [cores[i % len(cores)] for i in range(workers)]


def start_worker(ctx: Any, task: Task, core: int | None) -> Any:
    """Start a worker process pinned to ``core``, with Ctrl-C ignored.

    The child inherits the CPU affinity and the ignored SIGINT of the thread
    that starts it, so both hold before it imports anything (jax sizes its
    thread pools from the affinity). The parent stops workers with SIGTERM.
    """
    process = ctx.Process(target=worker_main, args=(task,), daemon=True)
    old_affinity = os.sched_getaffinity(0) if core is not None else None
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if core is not None:
            os.sched_setaffinity(0, {core})
        process.start()
    finally:
        if old_affinity is not None:
            os.sched_setaffinity(0, old_affinity)
        signal.signal(signal.SIGINT, old_handler)
    return process


def sweep_workers(
    s: Settings, todo: list[Matchup], out: Path, known: set[str], workers: int
) -> int:
    """Run the matchups in pinned worker processes, most expensive first."""
    ctx = multiprocessing.get_context("spawn")
    free = worker_cores(min(workers, len(todo)))
    queue = sorted(todo, key=lambda m: m.cost, reverse=True)
    numbers = {m: i for i, m in enumerate(queue, 1)}
    running: dict[Any, tuple[Matchup, int | None]] = {}
    total_cost, done_cost, failures, finished = sum(m.cost for m in todo), 0.0, [], 0
    t_sweep = time.perf_counter()
    try:
        while queue or running:
            while queue and free:
                m, core = queue.pop(0), free.pop(0)
                where = f" core {core}" if core is not None else ""
                tag = f"[{numbers[m]}/{len(todo)}{where}] {m.a.name} vs {m.b.name}"
                task = Task(
                    m.a.name, m.b.name, m.games, matchup_id(m, s), s, str(out),
                    frozenset(known), tag,
                )  # fmt: skip
                running[start_worker(ctx, task, core)] = (m, core)
            ready = multiprocessing.connection.wait([p.sentinel for p in running])
            for process in [p for p in running if p.sentinel in ready]:
                process.join()
                m, core = running.pop(process)
                free.append(core)
                finished += 1
                done_cost += m.cost
                if process.exitcode not in (0, EXIT_SKIPPED):
                    failures.append(f"{m.a.name} vs {m.b.name}")
                    if process.exitcode != 1:  # killed (e.g. out of memory) or crashed
                        log(f"{m.a.name} vs {m.b.name}: worker died (exit {process.exitcode})")
                elapsed = time.perf_counter() - t_sweep
                rough = elapsed * (total_cost - done_cost) / done_cost
                log(
                    f"sweep: {finished}/{len(todo)} matchups, elapsed {fmt_duration(elapsed)}, "
                    f"rough ETA {fmt_duration(rough)} (cost-weighted, {len(running)} running)"
                )
    except KeyboardInterrupt:
        for process in running:
            process.terminate()
        for process in running:
            process.join(10)
            if process.is_alive():
                process.kill()
        stopped = ", ".join(f"{m.a.name} vs {m.b.name}" for m, _ in running.values())
        log(f"interrupted: stopped {stopped or 'nothing'}")
        raise
    if failures:
        log(f"FAILED matchups (rerun to retry): {', '.join(failures)}")
        return 1
    log(f"sweep finished in {fmt_duration(time.perf_counter() - t_sweep)}")
    return 0


# --- Planning without playing ----------------------------------------------------------


def lpt_makespan(costs: list[float], workers: int) -> float:
    """Busiest worker's load when jobs go most expensive first to the least loaded worker."""
    loads = [0.0] * max(workers, 1)
    for c in sorted(costs, reverse=True):
        loads[loads.index(min(loads))] += c
    return max(loads)


def dry_run(s: Settings, matchups: list[Matchup], out: Path, workers: int, rerun: bool) -> int:
    known = done_ids(out)
    pending = []
    for m in matchups:
        plan = plan_matchup(m, s)
        done = matchup_id(m, s) in known
        state = ("rerun" if rerun else "done") if done else "todo"
        if state != "done":
            pending.append(m)
        print(
            f"{m.a.name} vs {m.b.name}: {describe_plan(m, plan)}; rough "
            f"{fmt_duration(m.rough_seconds())} [{state}]"
        )
    serial = sum(m.cost for m in pending)
    print(
        f"\n{len(pending)} matchups to run. Rough time on one pinned core (model in ROUGH_MS; "
        f"games of {ROUGH_TURNS_PER_GAME} turns, {ROUGH_TURNS_VS_RANDOM} against random agents; "
        f"{COMPILE_SECONDS:.0f} s per matchup to compile): {fmt_duration(serial)}"
    )
    if workers > 1:
        span = lpt_makespan([m.cost for m in pending], workers)
        print(f"with --workers {workers}: about {fmt_duration(span)} (the busiest worker)")
    print("Game lengths and costs vary by matchup: --estimate measures them.")
    return 0


def estimate(s: Settings, matchups: list[Matchup], out: Path, assume_turns: int) -> int:
    """Compile and probe each pending matchup (no games) and predict its wall time."""
    known = done_ids(out)
    total = 0.0
    log(
        f"estimate: probing {len(matchups)} matchups, assuming games of {assume_turns} turns "
        f"and slots 80% busy"
    )
    print("| matchup | games | slots | compile | ms per move | predicted play | state |")
    print("|---|--:|--:|--:|--:|--:|---|")
    for m in matchups:
        plan = plan_matchup(m, s)
        state = "done" if matchup_id(m, s) in known else "todo"
        prep = prepare(m, s, plan)
        per_move = prep.seconds_per_iteration / plan.slots
        play = per_move * plan.games * assume_turns / 0.8
        if state == "todo":
            total += play + prep.compile_seconds
        print(
            f"| {m.a.name} vs {m.b.name} | {plan.games} | {plan.slots} | "
            f"{fmt_duration(prep.compile_seconds)} | {1e3 * per_move:.3f} | "
            f"{fmt_duration(play)} | {state} |"
        )
        sys.stdout.flush()
        jax.clear_caches()
    print(
        f"\npredicted total for the todo matchups (play + compile, one core): {fmt_duration(total)}"
    )
    return 0


# --- Tables --------------------------------------------------------------------------


def agent_sort_key(name: str) -> tuple[int, int, str]:
    if name in registry.SIMPLE_AGENTS:
        return registry.SIMPLE_AGENTS.index(name), 0, name
    if name.startswith(("dqn", "ppo")):
        return 3, 0, name
    if match := re.match(r"mcts-(\d+)(.*)", name):
        return 4, int(match[1]), match[2]
    return 5, 0, name


@dataclasses.dataclass
class Cell:
    """Pooled result of one (A, B) pair."""

    wins: int = 0
    draws: int = 0
    losses: int = 0
    truncated: int = 0
    turns: float = 0.0
    play_seconds: float = 0.0
    probe_seconds: float = 0.0  # sum over lines of seconds_per_move_probe * games
    probe_games: int = 0  # games of the lines that have a probe
    seeds: list[int] = dataclasses.field(default_factory=list)
    commits: set[str] = dataclasses.field(default_factory=set)
    fingerprints: set[str] = dataclasses.field(default_factory=set)
    dirty: bool = False
    stopped_early: bool = False
    other_configs: int = 0  # lines with a different config_id, not shown
    configs: dict[str, tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )  # name: (hash, date)

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses


def pool_records(
    records: list[dict[str, Any]], seed: int | None = None
) -> dict[tuple[str, str], Cell]:
    """Pool lines per (a, b) as described in the module docstring."""
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for rec in records:
        if seed is None or rec["seed"] == seed:
            groups.setdefault((rec["a"], rec["b"]), {}).setdefault(rec["config_id"], []).append(rec)
    cells = {}
    for pair, by_config in groups.items():
        newest = max(by_config, key=lambda c: max(r["date"] for r in by_config[c]))
        best: dict[int, dict[str, Any]] = {}  # per seed: the line with the most games
        for rec in sorted(by_config[newest], key=lambda r: r["date"]):
            if rec["seed"] not in best or rec["games"] >= best[rec["seed"]]["games"]:
                best[rec["seed"]] = rec
        cell = Cell(other_configs=sum(len(v) for c, v in by_config.items() if c != newest))
        for rec in best.values():
            cell.wins += rec["wins"]
            cell.draws += rec["draws"]
            cell.losses += rec["losses"]
            cell.truncated += rec.get("truncated", 0)
            cell.turns += rec["mean_turns"] * rec["games"]
            cell.play_seconds += rec.get("play_seconds", rec.get("wall_seconds", 0.0))
            if probe := rec.get("seconds_per_move_probe"):
                cell.probe_seconds += probe * rec["games"]
                cell.probe_games += rec["games"]
            cell.seeds.append(rec["seed"])
            cell.commits.add(str(rec.get("git_commit", "unknown"))[:8])
            cell.fingerprints.add(str(rec.get("code_fingerprint", "unknown")))
            cell.dirty |= bool(rec.get("git_dirty", False))
            cell.stopped_early |= bool(rec.get("stopped_early", False))
            for side in ("a", "b"):
                digest = _digest(rec.get(f"{side}_config"))
                if rec[side] not in cell.configs or rec["date"] > cell.configs[rec[side]][1]:
                    cell.configs[rec[side]] = (digest, rec["date"])
        cells[pair] = cell
    return cells


def mixed_configs(cells: dict[tuple[str, str], Cell]) -> dict[str, str]:
    """Names shown with more than one config across cells: ``{name: newest config hash}``."""
    seen: dict[str, dict[str, str]] = {}  # name -> {config hash: newest date}
    for cell in cells.values():
        for name, (digest, date) in cell.configs.items():
            dates = seen.setdefault(name, {})
            dates[digest] = max(date, dates.get(digest, ""))
    return {name: max(d, key=d.get) for name, d in seen.items() if len(d) > 1}


def markdown_table(corner: str, rows: list[str], cols: list[str], cell_text: Callable) -> str:
    lines = [
        f"| {corner} | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for r in rows:
        lines.append(f"| **{r}** | " + " | ".join(cell_text(r, c) for c in cols) + " |")
    return "\n".join(lines)


def table_main(
    path: Path, a_names: set[str] | None, b_names: set[str] | None, seed: int | None
) -> int:
    records, bad, old = read_records(path)
    if old:
        log(f"warning: ignored {old} line(s) of an older schema in {path}")
    if not records:
        log(f"no results in {path}" + (f" ({bad} unreadable lines)" if bad else ""))
        return 1
    if bad:
        log(f"warning: skipped {bad} unreadable line(s) in {path}")
    cells = {
        pair: cell
        for pair, cell in pool_records(records, seed).items()
        if (a_names is None or pair[0] in a_names) and (b_names is None or pair[1] in b_names)
    }
    if not cells:
        log("no results match the filters")
        return 1
    rows = sorted({a for a, _ in cells}, key=agent_sort_key)
    cols = sorted({b for _, b in cells}, key=agent_sort_key)
    mixed = mixed_configs(cells)

    def at(r: str, c: str) -> Cell | None:
        return cells.get((r, c))

    def marks(cell: Cell) -> str:
        older = any(
            cell.configs[x][0] != newest for x, newest in mixed.items() if x in cell.configs
        )
        return ("‡" if older else "") + ("§" if cell.stopped_early else "")

    def score_text(r: str, c: str) -> str:
        cell = at(r, c)
        if cell is None or cell.games == 0:
            return "-"
        return f"{interval_text(cell.wins, cell.draws, cell.losses)}{marks(cell)} ({cell.games})"

    print("## Score of A (rows) against B (columns)\n")
    print("Score = (wins + draws / 2) / games, [95% Wilson interval], (games).\n")
    print(markdown_table("A \\ B", rows, cols, score_text))

    def wdl(r: str, c: str) -> str:
        cell = at(r, c)
        if cell is None:
            return "-"
        return f"{cell.wins} / {cell.draws} / {cell.losses} ({cell.truncated} t.o.)"

    print("\n## Wins / draws / losses of A (truncated games, included in the draws)\n")
    print(markdown_table("A \\ B", rows, cols, wdl))

    def mean_turns(r: str, c: str) -> str:
        cell = at(r, c)
        return f"{cell.turns / cell.games:.0f}" if cell else "-"

    print("\n## Mean game length (turns)\n")
    print(markdown_table("A \\ B", rows, cols, mean_turns))

    def ms_per_move(r: str, c: str) -> str:
        cell = at(r, c)
        if cell is None or cell.turns <= 0 or cell.play_seconds <= 0:
            return "-"
        effective = 1e3 * cell.play_seconds / cell.turns
        if cell.probe_games == 0:
            return f"- ({effective:.3f})"
        return f"{1e3 * cell.probe_seconds / cell.probe_games:.3f} ({effective:.3f})"

    print("\n## Time per move (milliseconds): probe (effective)\n")
    print(
        "One move is one search of A plus B's move and the environment step for one game, in\n"
        "one process (one pinned core with --workers). *Probe*: measured on the first turns of\n"
        "games with every slot busy. *Effective*: play time / total game turns, which also pays\n"
        "for the slots left idle at the end of a match and for costlier later turns.\n"
        "Throughputs, not the latency of one search.\n"
    )
    print(markdown_table("A \\ B", rows, cols, ms_per_move))

    commits = sorted({c for cell in cells.values() for c in cell.commits})
    fingerprints = Counter(f for cell in cells.values() for f in cell.fingerprints)
    stale = sum(cell.other_configs for cell in cells.values())
    seeds = sorted({s for cell in cells.values() for s in cell.seeds})
    print(
        f"\nSeeds pooled: {seeds}. Git commits of the shown lines: {', '.join(commits)}; "
        f"code fingerprints: {', '.join(sorted(fingerprints))}."
    )
    if any(cell.dirty for cell in cells.values()):
        print(
            "Some lines were produced with uncommitted changes in src/, baselines/ or benchmarks/."
        )
    if mixed:
        print(
            "‡ an agent of this cell has an older config than in other cells "
            f"({', '.join(sorted(mixed))}): the same name means different settings."
        )
        log(f"warning: {', '.join(sorted(mixed))} appear with different configs across cells (‡)")
    if any(cell.stopped_early for cell in cells.values()):
        print("§ stopped early by --target-ci: slightly optimistic near 0 or 1 (see the docs).")
    if len(fingerprints) > 1:
        log(
            f"warning: the table mixes results of {len(fingerprints)} code versions "
            f"(code_fingerprint {', '.join(sorted(fingerprints))}); --rerun the older matchups "
            f"to compare like with like"
        )
    if stale:
        log(
            f"warning: {stale} line(s) with a different agent config or rules than the newest "
            f"line of their pair were left out (see config_id in {path})"
        )
    return 0


# --- Command line --------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        epilog="MCTS shorthands (mcts-<sims>-<shorthand>-...):\n"
        + registry.mcts_shorthand_help()
        + "\nAny MCTSConfig field can also be set with :field=value after them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--a", help="agents under test, comma-separated (rows of the table)")
    p.add_argument("--b", help="opponents, comma-separated (columns); all pairs a x b are played")
    p.add_argument("--games", type=int, default=256, help="games per matchup (default 256)")
    p.add_argument(
        "--games-for",
        help="per-agent game counts: 'mcts-1024=64,mcts-256@dqn=128' ('A=N' applies to every "
        "matchup with that agent on either side, the smaller count wins; 'A@B=N' to one pair)",
    )
    p.add_argument(
        "--target-ci",
        type=float,
        help="stop a matchup after a round once the 95%% interval's half-width is at most this "
        "(needs >= 64 games; --games stays the maximum). Sequential: slightly optimistic "
        "near 0 or 1, so prefer a fixed --games for headline cells",
    )
    p.add_argument(
        "--round-games",
        type=int,
        default=64,
        help="with --target-ci: games per round, i.e. between stop tests (default 64)",
    )
    p.add_argument(
        "--slots",
        type=int,
        default=32,
        help="games simulated at once per matchup, before the memory cap (default 32)",
    )
    p.add_argument(
        "--mem-mb",
        type=float,
        default=1024.0,
        help="memory budget (MB) for the search trees of one process; the slots shrink to fit "
        "(default 1024)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="matchups run at once, each in a process pinned to its own core (default 1: in "
        "this process). --workers 4 on a 4-core machine is the fast path",
    )
    p.add_argument(
        "--rerun",
        action="store_true",
        help="play the selected matchups even if the results file has them (e.g. after a fix)",
    )
    p.add_argument(
        "--seed",
        type=int,
        help="base seed (default 0); with --table, only lines of this seed are shown",
    )
    p.add_argument("--max-turns", type=int, default=500, help="cut games off here (default 500)")
    p.add_argument("--size", type=int, default=11, help="board width and height (default 11)")
    p.add_argument("--snakes", type=int, default=2, help="snakes per game (default 2)")
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="results file (default benchmarks/results/strength.jsonl)",
    )
    p.add_argument("--table", action="store_true", help="print markdown tables from --out and exit")
    p.add_argument(
        "--dry-run", action="store_true", help="print the plan and a rough cost estimate, and exit"
    )
    p.add_argument(
        "--estimate",
        action="store_true",
        help="compile and probe every pending matchup (no games) and predict the wall time",
    )
    p.add_argument(
        "--assume-turns",
        type=int,
        default=ROUGH_TURNS_PER_GAME,
        help=f"--estimate: mean game length (default {ROUGH_TURNS_PER_GAME})",
    )
    return p.parse_args(argv)


def make_game(size: int, snakes: int) -> GameConfig:
    try:
        game = GameConfig(width=size, height=size, num_snakes=snakes)
        state_bytes(game)  # builds the start position: rejects boards too small for the snakes
    except ValueError as e:
        raise SpecError(f"--size {size} --snakes {snakes}: {e}") from None
    if game.solo:
        raise SpecError("need at least two snakes")
    return game


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.table:
            return table_main(args.out, names_of(args.a), names_of(args.b), args.seed)
        if not args.a or not args.b:
            raise SpecError("--a and --b are required (or use --table)")
        sizes = (args.games, args.slots, args.round_games, args.workers, args.max_turns)
        if min(*sizes, args.size, args.snakes, args.assume_turns) < 1:
            raise SpecError(
                "--games, --slots, --round-games, --workers, --max-turns, --size, --snakes and "
                "--assume-turns must be >= 1"
            )
        game = make_game(args.size, args.snakes)
        s = Settings(
            game, args.max_turns, args.seed or 0, args.target_ci, args.slots, args.round_games,
            args.mem_mb,
        )  # fmt: skip
        overrides = parse_overrides(args.games_for)
        a_agents, b_agents = parse_agents(args.a), parse_agents(args.b)
        for agent in {x.name: x for x in a_agents + b_agents}.values():
            try:
                registry.check_game(agent.spec, game)
            except (ValueError, FileNotFoundError) as e:
                raise SpecError(str(e)) from None
        matchups = build_matchups(a_agents, b_agents, args.games, overrides)
        provenance()  # now: every module the agents need is loaded
        if args.dry_run:
            return dry_run(s, matchups, args.out, args.workers, args.rerun)
        if args.estimate:
            return estimate(s, matchups, args.out, args.assume_turns)
        return sweep(s, matchups, args.out, args.workers, args.rerun)
    except SpecError as e:
        log(f"error: {e}")
        return 2
    except KeyboardInterrupt:
        log("interrupted: finished matchups are saved; rerun the same command to resume")
        return 130


if __name__ == "__main__":
    sys.exit(main())
