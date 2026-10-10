"""Strength benchmark: how often agents beat fixed opponents (MCTS score vs simulations).

Plays every pair ``A x B`` of two agent lists with :func:`slinky.evaluate.play_match`
and appends one JSON line per finished matchup to ``benchmarks/results/strength.jsonl``.
``--table`` turns that file into markdown tables. The score is A's win rate with a
draw worth 1/2, with a 95% confidence interval; seats alternate, so neither agent
gets the better starting seat more often.

Agents (``--a`` and ``--b`` take comma-separated lists):

* ``random_legal``  uniform over the moves that do not certainly hit a wall or body.
* ``heuristic``     :func:`slinky.heuristic.heuristic` with its default weights.
* ``dqn``           the greedy self-play DQN of ``baselines/checkpoints/dqn-duel-seed0``
                    (``--dqn-checkpoint`` picks another run; a name other than the default
                    one is reported as ``dqn@<directory>``). 11x11 duel only.
* ``mcts-<sims>``   :func:`slinky.mcts.mcts` with ``<sims>`` simulations per move and
                    the default :class:`~slinky.mcts.MCTSConfig`. Dash-separated suffixes
                    change config fields (see ``--help`` for the full list), e.g.
                    ``mcts-64-rollout`` or ``mcts-256-rm-c0.5``. Suffixes may come in any
                    order; names are printed in a canonical form that lists only the
                    non-default settings (``mcts-64-rollout`` becomes ``mcts-64-rollout10``).

Usage::

    python benchmarks/strength.py --a mcts-16,mcts-64,mcts-256 --b random_legal,heuristic,dqn \\
        --games 256 --seed 0
    python benchmarks/strength.py --table                 # tables from the results file
    python benchmarks/strength.py --a mcts-64 --b heuristic --games 256 --dry-run
    python benchmarks/strength.py --a mcts-1024 --b heuristic --games 256 --estimate

Design, in the order a matchup is run:

**Identity and resume.** A matchup's identity is a hash of both agents (name and
*full* config, including every ``MCTSConfig`` field, the heuristic weights and the
checkpoint's parameter hash), the game rules, ``--max-turns``, the requested number
of games, the seed and ``--target-ci``. Before each matchup the output file is read
again, and a matchup whose identity is already in it is skipped, so an interrupted or
crashed sweep resumes by running the same command. Changing an MCTS default in the code
changes the identity, so old results are not silently reused for the new agent. A line
is written only when its matchup is complete (one ``write`` under an exclusive file
lock, after repairing a torn last line), so a kill never leaves half a result.

**Randomness.** A matchup's key is ``fold_in(key(seed), h)``, where ``h`` is the first 31
bits of ``sha256("<a name>\\0<b name>")``. It depends on the seed and the two canonical
names only, never on run order or on which other matchups are in the sweep. Round ``r``
of a matchup uses ``fold_in(key, r)``. The batch size shapes the random streams inside
``play_match``, so the exact numbers reproduce only with the same ``--batch-size``
(recorded in the line); the statistics do not depend on it.

**Batches, memory and rounds.** ``play_match`` runs one jitted batch at a time and
every game in a batch is computed until the *longest* one ends, so a round costs about
``batch x longest game x cost per move``. Rounds are full batches of one size ``B``
(``play_match`` recompiles for every batch shape, so a short last batch would compile
twice). With ``G`` games requested, ``--batch-size`` as the cap ``C`` and ``N`` snakes:
``rounds = ceil(G / C)`` and ``B = ceil(G / rounds)`` rounded up to a multiple of ``N``
(so seats stay balanced), and ``rounds * B`` games are played (at most ``N - 1``
more per round than requested; both numbers are recorded). **Auto-shrink:** ``C`` is
``--batch-size`` capped by the memory budget ``--mem-mb`` (default 1024 MB): ``C <=
mem_mb / (2 * tree bytes per game)``, where the tree of an MCTS agent holds
``sims + 1`` nodes of about 1.04 KB in the 11x11 duel (0.9 KB of it the game
state; 1.3 KB with ``rm``; 2.6 KB with 4 snakes), summed over the MCTS agents in
the matchup, and the factor 2 leaves room for XLA temporaries. The cap rarely binds
below about 1000 simulations; the peak resident memory of the process is recorded
so the model can be checked. With ``--target-ci`` the stop test runs after every round,
so use a smaller ``--batch-size`` for finer stopping.

**Timing.** Before the first round the script runs the matchup for 1 turn
(compilation), again for 1 turn and for 9 turns (the cost of one batch-turn from the
difference), which gives an upper bound for the time of a round: ``--max-turns``
batch-turns, reached whenever one game in the batch stalls. Progress lines then
use the observed time per round. ``wall_seconds`` is the whole matchup;
``play_seconds`` leaves out compilation and the probes;
``seconds_per_game_turn = play_seconds / (sum of game lengths)``. That is a throughput
(one MCTS search per game-turn serves both seats, and the opponent and the environment
step are included), amortised over a batch on this machine, with the waiting of finished
games included; it is not the latency of one search.

**Results.** One JSON object per line (``SCHEMA``): ``a``, ``b``, ``a_config``, ``b_config``,
``game``, ``max_turns``, ``games_requested``, ``games``, ``wins``, ``draws``, ``losses``,
``truncated`` (games cut off by ``--max-turns`` with both snakes alive; counted as
draws, so they are inside ``draws``), ``score``, ``ci95``, ``mean_turns``, the seconds
fields, ``batch_size``, ``rounds``, ``seed``, ``git_commit``, ``git_dirty`` (uncommitted
changes under ``src/`` or ``baselines/``), ``jax_version``, ``date`` and the identity hashes
(``matchup_id``, ``config_id``). ``rounds_detail`` lists W/D/L/truncated/turns/seconds per
round.

**Tables.** ``--table`` groups lines by ``(a, b)`` and uses the group with the newest
line when several configs exist for a pair (the others are counted in a warning).
Within a group only the line with the most games of each seed is used (a larger run
with the same seed contains the games of a smaller one), and the seeds are pooled by
adding the counts, so extra seeds can top up a cell. The interval is recomputed from
the pooled counts; when all games had the same outcome (zero sample variance) the
reported half-width is the rule of three, ``3 / games``, and marked with a dagger.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax

from slinky import heuristic as heuristic_lib
from slinky import mcts as mcts_lib
from slinky.env import BattlesnakeEnv
from slinky.evaluate import Policy, greedy_from_q, play_match, random_legal
from slinky.types import GameConfig

try:
    import fcntl
except ImportError:  # not on POSIX: no file locking
    fcntl = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "strength.jsonl"
DEFAULT_DQN = ROOT / "baselines" / "checkpoints" / "dqn-duel-seed0"
SCHEMA = 1
Z95 = 1.959963984540054
PROBE_TURNS = 8  # batch-turns timed to estimate the cost of a round
MIN_GAMES_TO_STOP = 64  # --target-ci never stops before this many games
TREE_SAFETY = 2.0  # memory model: XLA temporaries and double buffering
SIMPLE_AGENTS = ("random_legal", "heuristic")
REQUIRED = frozenset(
    ("a", "b", "matchup_id", "config_id", "seed", "games", "wins", "draws", "losses")
    + ("mean_turns", "date")
)


class SpecError(ValueError):
    """A bad agent name or option."""


def log(message: str = "") -> None:
    print(message, file=sys.stderr, flush=True)


# --- MCTS name suffixes -------------------------------------------------------------

_D = mcts_lib.DEFAULT_CONFIG


@dataclasses.dataclass(frozen=True)
class Suffix:
    """A dash-separated suffix of ``mcts-<sims>`` that sets config fields.

    ``number`` is None for a flag and ``int`` or ``float`` for ``name<value>``;
    ``default`` is the value when the number is omitted (None: required).
    ``fields(value)`` are the config fields it sets; ``value(config)`` is what
    the canonical name shows (None or False when the config has the default).
    """

    name: str
    doc: str
    fields: Callable[[Any], dict[str, Any]]
    value: Callable[[mcts_lib.MCTSConfig], Any]
    number: type | None = None
    default: float | None = None

    def token(self, config: mcts_lib.MCTSConfig) -> str | None:
        v = self.value(config)
        if v is None or v is False:
            return None
        return self.name if self.number is None else f"{self.name}{v:g}"


def _differs(field: str, config: mcts_lib.MCTSConfig) -> Any:
    v = getattr(config, field)
    return v if v != getattr(_D, field) else None


def _rollout(policy: str) -> Callable[[mcts_lib.MCTSConfig], Any]:
    return lambda c: c.rollout_steps or None if c.rollout_policy == policy else None


# fmt: off
SUFFIXES: tuple[Suffix, ...] = (
    Suffix("rm", "regret-matching selection instead of DUCT (selection='rm')",
           lambda v: {"selection": "rm"}, lambda c: c.selection == "rm"),
    Suffix("tuned", "UCB1-Tuned variance bound in DUCT (ucb1_tuned=True)",
           lambda v: {"ucb1_tuned": True}, lambda c: c.ucb1_tuned),
    Suffix("c", "c<x>: exploration constant, e.g. c1.4 (exploration=x)",
           lambda v: {"exploration": v}, lambda c: _differs("exploration", c), float),
    Suffix("noheur", "no heuristic leaf evaluation: living snakes are worth 0 (leaf='none')",
           lambda v: {"leaf": "none"}, lambda c: c.leaf == "none"),
    Suffix("rollout", "rollout[<k>]: k random-policy rollout turns before the leaf (default 10)",
           lambda v: {"rollout_steps": v, "rollout_policy": "random"}, _rollout("random"),
           int, 10),
    Suffix("hrollout", "hrollout[<k>]: like rollout, with the heuristic policy (default 10)",
           lambda v: {"rollout_steps": v, "rollout_policy": "heuristic"}, _rollout("heuristic"),
           int, 10),
    Suffix("spawn", "sample food spawns inside the tree (spawn_food=True)",
           lambda v: {"spawn_food": True}, lambda c: c.spawn_food),
    Suffix("sample", "sample the final move from the visit counts (final='sample')",
           lambda v: {"final": "sample"}, lambda c: c.final == "sample"),
    Suffix("contempt", "contempt<x>: a mutual elimination is worth -x in the search (draw_value)",
           lambda v: {"draw_value": 0.0 - v},
           lambda c: 0.0 - c.draw_value if c.draw_value != _D.draw_value else None, float),
    Suffix("noise", "noise<x>: scale of the tie-breaking noise (tie_noise=x)",
           lambda v: {"tie_noise": v}, lambda c: _differs("tie_noise", c), float),
    Suffix("gamma", "gamma<x>: regret matching's exploration mix (rm_gamma=x)",
           lambda v: {"rm_gamma": v}, lambda c: _differs("rm_gamma", c), float),
    Suffix("depth", "depth<k>: deepest descent from the root (max_depth=k)",
           lambda v: {"max_depth": v}, lambda c: _differs("max_depth", c), int),
)
# fmt: on
_SUFFIX_BY_NAME = {s.name: s for s in SUFFIXES}


def parse_mcts(text: str) -> tuple[str, mcts_lib.MCTSConfig]:
    """``(canonical name, config)`` for ``mcts-<sims>[-suffix...]``."""
    parts = text.split("-")
    if parts[0] != "mcts" or len(parts) < 2 or not parts[1].isdigit() or int(parts[1]) < 1:
        raise SpecError(f"{text!r}: expected mcts-<sims>[-suffix...] with sims >= 1")
    fields: dict[str, Any] = {"num_simulations": int(parts[1])}
    seen: set[str] = set()
    for token in parts[2:]:
        match = re.fullmatch(r"([a-z]+)([0-9.]*)", token)
        suffix = _SUFFIX_BY_NAME.get(match[1]) if match else None
        if suffix is None:
            valid = ", ".join(s.name for s in SUFFIXES)
            raise SpecError(f"{text!r}: unknown suffix {token!r} (valid: {valid})")
        if suffix.name in seen:
            raise SpecError(f"{text!r}: suffix {suffix.name!r} given twice")
        seen.add(suffix.name)
        digits = match[2]
        if suffix.number is None:
            if digits:
                raise SpecError(f"{text!r}: suffix {suffix.name!r} takes no number")
            value = None
        elif digits:
            try:
                value = suffix.number(digits)
            except ValueError:
                raise SpecError(f"{text!r}: bad number in {token!r}") from None
        elif suffix.default is not None:
            value = suffix.default
        else:
            raise SpecError(f"{text!r}: suffix {suffix.name!r} needs a number, e.g. {suffix.name}1")
        if suffix.number is int and value < 1:
            raise SpecError(f"{text!r}: {token!r} must be >= 1")
        if {"rollout", "hrollout"} <= seen:
            raise SpecError(f"{text!r}: rollout and hrollout are exclusive")
        fields.update(suffix.fields(value))
    try:
        config = mcts_lib.MCTSConfig(**fields)
    except ValueError as e:
        raise SpecError(f"{text!r}: {e}") from None
    tokens = [t for s in SUFFIXES if (t := s.token(config))]
    return "-".join(["mcts", str(config.num_simulations), *tokens]), config


def suffix_help() -> str:
    width = max(len(s.name) for s in SUFFIXES) + 3
    return "\n".join(f"  {s.name:<{width}}{s.doc}" for s in SUFFIXES)


# --- Agents -------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Agent:
    """A named agent and everything that defines its behaviour (``config`` is JSON-able)."""

    name: str
    kind: str  # random_legal | heuristic | dqn | mcts
    config: dict[str, Any]
    mcts: mcts_lib.MCTSConfig | None = None
    checkpoint: Path | None = None

    @property
    def sims(self) -> int:
        return self.mcts.num_simulations if self.mcts else 0


def _jsonable(x: Any) -> Any:
    """Round-trip through JSON so that configs compare and hash the same after reloading."""
    return json.loads(json.dumps(x, sort_keys=True))


def _rel(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


@functools.cache
def load_dqn_module():
    """``baselines/dqn.py`` imported from its path (``baselines/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("baselines_dqn", ROOT / "baselines" / "dqn.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def parse_agent(text: str, dqn_checkpoint: Path = DEFAULT_DQN, describe: bool = True) -> Agent:
    """The agent named ``text``. With ``describe=False`` a DQN's checkpoint is not read."""
    text = text.strip().lower()
    if text == "random_legal":
        return Agent(text, "random_legal", {"type": "random_legal"})
    if text == "heuristic":
        weights = heuristic_lib.DEFAULT_WEIGHTS._asdict()
        return Agent(text, "heuristic", _jsonable({"type": "heuristic", "weights": weights}))
    if text == "dqn":
        path = dqn_checkpoint.resolve()
        name = "dqn" if path == DEFAULT_DQN.resolve() else f"dqn@{path.name}"
        config: dict[str, Any] = {"type": "dqn", "checkpoint": _rel(path), "greedy": True}
        if describe:
            if not (path / "params.npz").is_file():
                raise SpecError(f"dqn: no checkpoint at {path} (params.npz missing)")
            cfg = load_dqn_module().load_config(str(path))
            config["params_sha256"] = _file_hash(path / "params.npz")
            config["network"] = dataclasses.asdict(cfg)
        return Agent(name, "dqn", _jsonable(config), checkpoint=path)
    if text.startswith("mcts-"):
        name, cfg = parse_mcts(text)
        config = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        config["weights"] = cfg.weights._asdict()
        return Agent(name, "mcts", _jsonable({"type": "mcts", **config}), mcts=cfg)
    raise SpecError(f"unknown agent {text!r}: use random_legal, heuristic, dqn or mcts-<sims>")


def parse_agents(text: str, dqn_checkpoint: Path, describe: bool = True) -> list[Agent]:
    agents: dict[str, Agent] = {}
    for item in text.split(","):
        if item.strip():
            agent = parse_agent(item, dqn_checkpoint, describe)
            agents.setdefault(agent.name, agent)
    return list(agents.values())


@functools.cache
def make_env(game: GameConfig, egocentric: bool) -> BattlesnakeEnv:
    """One env per (rules, observations): policies and compiled matches are cached on it."""
    return BattlesnakeEnv(game, obs="egocentric" if egocentric else None)


@functools.cache
def dqn_policy(checkpoint: Path) -> Policy:
    dqn = load_dqn_module()
    cfg = dqn.load_config(str(checkpoint))
    params = dqn.load_params(str(checkpoint), cfg)

    def q_fn(obs: jax.Array) -> jax.Array:
        return dqn.q_network(params, obs, cfg)

    return greedy_from_q(q_fn)


def build_policy(agent: Agent, env: BattlesnakeEnv) -> Policy:
    if agent.kind == "random_legal":
        return random_legal(env)
    if agent.kind == "heuristic":
        return heuristic_lib.heuristic(env)
    if agent.kind == "mcts":
        return mcts_lib.mcts(env, agent.mcts)
    return dqn_policy(agent.checkpoint)


def check_dqn_game(agent: Agent, game: GameConfig) -> None:
    """The DQN's observation shape and training rules fix the game it can play."""
    cfg = load_dqn_module().load_config(str(agent.checkpoint))
    trained = cfg.game
    want = (trained.width, trained.height, trained.num_snakes, trained.ruleset.value)
    have = (game.width, game.height, game.num_snakes, game.ruleset.value)
    if want != have:
        raise SpecError(f"{agent.name} was trained for {want} but the game is {have}")


# --- Matchups: identity, key, batches ------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Settings:
    """Everything shared by the matchups of one invocation."""

    game: GameConfig
    max_turns: int
    seed: int
    target_ci: float | None
    batch_size: int
    mem_mb: float


@dataclasses.dataclass(frozen=True, eq=False)
class Matchup:
    a: Agent
    b: Agent
    games: int  # requested

    @property
    def uses_dqn(self) -> bool:
        return "dqn" in (self.a.kind, self.b.kind)

    @property
    def cost(self) -> float:
        """Relative cost for ordering and the rough sweep ETA (a move costs ~ sims + 4)."""
        return self.games * (self.a.sims + self.b.sims + 4)


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
    return _digest([config_id(m, s), m.games, s.seed, s.target_ci])


def matchup_key(seed: int, a_name: str, b_name: str) -> jax.Array:
    """PRNG key from the seed and the two names only (see the module docstring)."""
    digest = hashlib.sha256(f"{a_name}\0{b_name}".encode()).digest()
    return jax.random.fold_in(jax.random.key(seed), int.from_bytes(digest[:4], "big") >> 1)


def tree_bytes_per_game(agent: Agent, game: GameConfig) -> int:
    """Estimated search-tree bytes of one game (0 for agents without a tree).

    Per node: the game state, the joint-action children table, the legal mask,
    the leaf value, and per player and own move a visit count and a value sum
    (regret matching and UCB1-Tuned keep more). In the 11x11 duel this gives
    1.04 KB (1.30 KB with regret matching), as measured for ``slinky.mcts``.
    """
    if agent.mcts is None:
        return 0
    cfg, n = agent.mcts, game.num_snakes
    j = 4**n
    state = jax.eval_shape(make_env(game, False).init_state, jax.random.key(0))
    state_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(state))
    node = state_bytes + 4 * j + 4 * n + 1 + 4 * n + 32 * n
    if cfg.ucb1_tuned:
        node += 16 * n
    if cfg.selection == "rm":
        node += 32 * n + 4 * j + 4 * j * n
    sims = cfg.num_simulations
    return (sims + 1) * node + sims * n * 4 * 4  # plus the pre-drawn selection noise


@dataclasses.dataclass(frozen=True)
class Plan:
    rounds: int
    batch: int  # games per round
    mem_cap: int  # largest batch the memory budget allows
    tree_mb: float  # estimated search-tree memory of one batch

    @property
    def games(self) -> int:
        return self.rounds * self.batch


def plan_batches(m: Matchup, s: Settings) -> Plan:
    """Rounds and batch size for a matchup (the rule is in the module docstring)."""
    n = s.game.num_snakes
    per_game = sum(tree_bytes_per_game(x, s.game) for x in (m.a, m.b))
    mem_cap = max(int(s.mem_mb * 2**20 / (TREE_SAFETY * per_game)), 1) if per_game else 10**9
    cap = max(min(s.batch_size, mem_cap) // n * n, n)  # whole seat rotations
    rounds = -(-m.games // cap)
    per_round = -(-m.games // rounds)
    batch = -(-per_round // n) * n  # a whole number of seat rotations
    return Plan(rounds, batch, mem_cap, batch * per_game / 2**20)


# --- Statistics ----------------------------------------------------------------------


def outcome_stats(wins: int, draws: int, losses: int) -> tuple[int, float, float]:
    """``(games, score, ci95)`` exactly as ``slinky.evaluate`` computes them."""
    n = wins + draws + losses
    if n == 0:
        return 0, float("nan"), float("nan")
    score = (wins + 0.5 * draws) / n
    var = max(wins + 0.25 * draws - n * score * score, 0.0) / (n - 1) if n > 1 else 0.0
    return n, score, Z95 * math.sqrt(var / n)


def stop_ci(wins: int, draws: int, losses: int) -> float:
    """CI half-width for the early-stop test: +0.5 of each outcome, so it is never 0."""
    return outcome_stats(wins + 0.5, draws + 0.5, losses + 0.5)[2]


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**20 if sys.platform == "darwin" else 2**10)


def fmt_duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    if seconds < 90:
        return f"{seconds:.1f}s" if seconds < 10 else f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


# --- Environment info and the results file -------------------------------------------


def git_info() -> tuple[str, bool]:
    def run(*args: str) -> str:
        out = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=20, check=True
        )
        return out.stdout.strip()

    try:
        return run("rev-parse", "HEAD"), bool(
            run("status", "--porcelain", "--", "src", "baselines")
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


def read_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """``(records, number of unreadable lines)``; a torn or foreign line is skipped."""
    records, bad = [], 0
    if not path.is_file():
        return records, bad
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec["schema"] != SCHEMA or not REQUIRED.issubset(rec):
                    raise ValueError("not a result line")
                records.append(rec)
            except (ValueError, KeyError, TypeError):
                bad += 1
    return records, bad


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one line atomically (exclusive lock, torn last line repaired first)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, sort_keys=False) + "\n").encode()
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


# --- Running a matchup ---------------------------------------------------------------


@dataclasses.dataclass
class Prepared:
    env: BattlesnakeEnv
    policy_a: Policy
    policy_b: Policy
    key: jax.Array
    compile_seconds: float
    seconds_per_batch_turn: float  # from the probe

    def play(self, key: jax.Array, plan: Plan, max_turns: int):
        return play_match(
            self.env, self.policy_a, self.policy_b, key, plan.batch, max_turns, plan.batch
        )


def prepare(m: Matchup, s: Settings, plan: Plan) -> Prepared:
    """Build the policies, compile the match and time a few batch-turns (``max_turns`` is traced,
    so short calls reuse the compiled program of the real rounds)."""
    env = make_env(s.game, m.uses_dqn)
    key = matchup_key(s.seed, m.a.name, m.b.name)
    prep = Prepared(env, build_policy(m.a, env), build_policy(m.b, env), key, 0.0, 0.0)
    warm = jax.random.fold_in(prep.key, 2**31 - 1)  # a key no counted round uses

    def timed(turns: int) -> float:
        t0 = time.perf_counter()
        prep.play(warm, plan, turns)
        return time.perf_counter() - t0

    prep.compile_seconds = timed(1)
    base = timed(1)
    probe = timed(PROBE_TURNS + 1)
    per_turn = (probe - base) / PROBE_TURNS
    prep.seconds_per_batch_turn = per_turn if per_turn > 0 else probe / (PROBE_TURNS + 1)
    return prep


def run_matchup(m: Matchup, s: Settings, plan: Plan, tag: str) -> dict[str, Any]:
    """Play all rounds of a matchup and return its result record."""
    t_start = time.perf_counter()
    prep = prepare(m, s, plan)
    upper = prep.seconds_per_batch_turn * s.max_turns
    log(
        f"{tag}   compile {fmt_duration(prep.compile_seconds)}; probe "
        f"{1e3 * prep.seconds_per_batch_turn:.2f} ms per batch-turn -> a round of "
        f"{s.max_turns} turns takes at most {fmt_duration(upper)} "
        f"({plan.rounds} round(s): at most {fmt_duration(upper * plan.rounds)})"
    )
    w = d = lo = trunc = turns = 0
    rounds_detail: list[list[float]] = []
    play_seconds = 0.0
    stopped_early = False
    for r in range(plan.rounds):
        t0 = time.perf_counter()
        res = prep.play(jax.random.fold_in(prep.key, r), plan, s.max_turns)
        dt = time.perf_counter() - t0
        play_seconds += dt
        round_turns = round(res.mean_turns * res.num_games)
        w, d, lo = w + res.wins, d + res.draws, lo + res.losses
        trunc, turns = trunc + res.truncated, turns + round_turns
        rounds_detail.append(
            [res.wins, res.draws, res.losses, res.truncated, round_turns, round(dt, 3)]
        )
        n, score, ci = outcome_stats(w, d, lo)
        eta = (plan.rounds - r - 1) * play_seconds / (r + 1)
        log(
            f"{tag}   round {r + 1}/{plan.rounds}: {res.num_games} games in {fmt_duration(dt)} | "
            f"total {n} games, W/D/L {w}/{d}/{lo} ({trunc} truncated), score {score:.3f} ± {ci:.3f}"
            f" | ETA {fmt_duration(eta)}"
        )
        if (
            s.target_ci is not None
            and r + 1 < plan.rounds
            and n >= MIN_GAMES_TO_STOP
            and stop_ci(w, d, lo) <= s.target_ci
        ):
            log(f"{tag}   stopping early: CI half-width {stop_ci(w, d, lo):.3f} <= {s.target_ci}")
            stopped_early = True
            break
    n, score, ci = outcome_stats(w, d, lo)
    commit, dirty = git_info()
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
        "mean_turns": turns / n,
        "wall_seconds": time.perf_counter() - t_start,
        "compile_seconds": prep.compile_seconds,
        "play_seconds": play_seconds,
        "seconds_per_game_turn": play_seconds / max(turns, 1),
        "seconds_per_batch_turn_probe": prep.seconds_per_batch_turn,
        "batch_size": plan.batch,
        "rounds": plan.rounds,
        "rounds_played": len(rounds_detail),
        "target_ci": s.target_ci,
        "stopped_early": stopped_early,
        "est_tree_mb": plan.tree_mb,
        "peak_rss_mb": peak_rss_mb(),
        "seed": s.seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rounds_detail": rounds_detail,
        "rounds_detail_columns": ["wins", "draws", "losses", "truncated", "turns", "seconds"],
    }


# --- Sweep ---------------------------------------------------------------------------


def parse_overrides(text: str | None, checkpoint: Path) -> dict[tuple[str, str | None], int]:
    """``--games-for`` as ``{(a, None | b): games}``; items are ``A=N`` or ``A:B=N``."""
    out: dict[tuple[str, str | None], int] = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        names, _, count = item.rpartition("=")
        if not names or not count.strip().isdigit() or int(count) < 1:
            raise SpecError(f"--games-for item {item!r}: expected A=N or A:B=N with N >= 1")
        a, _, b = names.partition(":")
        key_a = parse_agent(a, checkpoint, describe=False).name
        key_b = parse_agent(b, checkpoint, describe=False).name if b else None
        out[(key_a, key_b)] = int(count)
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
    return sorted(pairs, key=lambda m: m.cost)  # stable: ties keep the command-line order


def describe_plan(m: Matchup, plan: Plan) -> str:
    shape = f"{plan.rounds} x {plan.batch}" if plan.rounds > 1 else f"{plan.batch}"
    extra = f", search trees ~{plan.tree_mb:.3g} MB per batch" if plan.tree_mb else ""
    return f"{m.games} games requested, {shape} played{extra}"


def sweep(s: Settings, matchups: list[Matchup], out: Path) -> int:
    commit, dirty = git_info()
    log(
        f"strength sweep: {len(matchups)} matchups, jax {jax.__version__} "
        f"({jax.default_backend()}, {os.cpu_count()} cpus), commit {commit[:10]}"
        f"{'+uncommitted' if dirty else ''}, seed {s.seed}, max_turns {s.max_turns}, out {out}"
    )
    done_ids = {r["matchup_id"] for r in read_records(out)[0]}
    todo = [m for m in matchups if matchup_id(m, s) not in done_ids]
    log(f"{len(matchups) - len(todo)} already in the results file, {len(todo)} to run")
    total_cost, done_cost, failures = sum(m.cost for m in todo), 0.0, []
    t_sweep = time.perf_counter()
    for i, m in enumerate(todo, 1):
        if matchup_id(m, s) in {r["matchup_id"] for r in read_records(out)[0]}:
            log(
                f"[{i}/{len(todo)}] {m.a.name} vs {m.b.name}: finished by another process, skipping"
            )
            done_cost += m.cost
            continue
        plan = plan_batches(m, s)
        tag = f"[{i}/{len(todo)}] {m.a.name} vs {m.b.name}"
        log(f"{tag} | {describe_plan(m, plan)}")
        try:
            record = run_matchup(m, s, plan, tag)
        except Exception as e:  # keep the sweep going; the matchup is retried on the next run
            log(f"{tag} FAILED: {type(e).__name__}: {e}")
            failures.append(f"{m.a.name} vs {m.b.name}")
            jax.clear_caches()
            continue
        append_record(out, record)
        jax.clear_caches()  # compiled programs of finished matchups are never reused
        done_cost += m.cost
        elapsed = time.perf_counter() - t_sweep
        r = record
        log(
            f"{tag} done: score {r['score']:.3f} ± {r['ci95']:.3f} over {r['games']} games "
            f"(W/D/L {r['wins']}/{r['draws']}/{r['losses']}) in {fmt_duration(r['wall_seconds'])} "
            f"(compile {fmt_duration(r['compile_seconds'])}), "
            f"{1e3 * r['seconds_per_game_turn']:.3f} ms per game-turn, peak RSS "
            f"{r['peak_rss_mb']:.0f} MB"
        )
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


def estimate(s: Settings, matchups: list[Matchup], out: Path, assume_turns: int) -> int:
    """Compile and probe each pending matchup (no games) and predict its wall time."""
    done_ids = {r["matchup_id"] for r in read_records(out)[0]}
    total = 0.0
    log(f"estimate: probing {len(matchups)} matchups, assuming rounds of {assume_turns} turns")
    print(
        "| matchup | games (rounds x batch) | compile | ms per game-turn | predicted play | state |"
    )
    print("|---|---|--:|--:|--:|---|")
    for m in matchups:
        plan = plan_batches(m, s)
        state = "done" if matchup_id(m, s) in done_ids else "todo"
        prep = prepare(m, s, plan)
        play = prep.seconds_per_batch_turn * assume_turns * plan.rounds
        per_turn = 1e3 * prep.seconds_per_batch_turn / plan.batch
        if state == "todo":
            total += play + prep.compile_seconds
        print(
            f"| {m.a.name} vs {m.b.name} | {plan.games} ({plan.rounds} x {plan.batch}) | "
            f"{fmt_duration(prep.compile_seconds)} | {per_turn:.3f} | "
            f"{fmt_duration(play)} | {state} |"
        )
        sys.stdout.flush()
        jax.clear_caches()
    print(f"\npredicted total for the todo matchups (play + compile): {fmt_duration(total)}")
    return 0


# --- Tables --------------------------------------------------------------------------


def agent_sort_key(name: str) -> tuple[int, int, str]:
    if name in SIMPLE_AGENTS:
        return SIMPLE_AGENTS.index(name), 0, name
    if name.startswith("dqn"):
        return 2, 0, name
    if match := re.match(r"mcts-(\d+)(.*)", name):
        return 3, int(match[1]), match[2]
    return 4, 0, name


@dataclasses.dataclass
class Cell:
    """Pooled result of one (A, B) pair."""

    wins: int = 0
    draws: int = 0
    losses: int = 0
    truncated: int = 0
    turns: float = 0.0
    play_seconds: float = 0.0
    seeds: list[int] = dataclasses.field(default_factory=list)
    commits: set[str] = dataclasses.field(default_factory=set)
    dirty: bool = False
    other_configs: int = 0  # lines with a different config_id, not shown

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
            cell.seeds.append(rec["seed"])
            cell.commits.add(str(rec.get("git_commit", "unknown"))[:8])
            cell.dirty |= bool(rec.get("git_dirty", False))
        cells[pair] = cell
    return cells


def markdown_table(corner: str, rows: list[str], cols: list[str], cell_text: Callable) -> str:
    lines = [
        f"| {corner} | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for r in rows:
        lines.append(f"| **{r}** | " + " | ".join(cell_text(r, c) for c in cols) + " |")
    return "\n".join(lines)


def score_text(cell: Cell | None) -> str:
    if cell is None or cell.games == 0:
        return "-"
    n, score, ci = outcome_stats(cell.wins, cell.draws, cell.losses)
    flat = n in (cell.wins, cell.draws, cell.losses)  # every game had the same outcome
    return f"{score:.3f} ± {3 / n if flat else ci:.3f}{'†' if flat else ''} ({n})"


def table_main(
    path: Path, a_names: set[str] | None, b_names: set[str] | None, seed: int | None
) -> int:
    records, bad = read_records(path)
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

    def at(r: str, c: str) -> Cell | None:
        return cells.get((r, c))

    print("## Score of A (rows) against B (columns)\n")
    print("Score = (wins + draws / 2) / games, ± 95% interval, (games).\n")
    print(markdown_table("A \\ B", rows, cols, lambda r, c: score_text(at(r, c))))

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
        return f"{1e3 * cell.play_seconds / cell.turns:.3f}"

    print("\n## Time per move (milliseconds per game-turn)\n")
    print(
        "One game-turn is one search of A plus B's move and the environment step, amortised\n"
        "over a batch on this CPU and including the waiting of finished games (see the module\n"
        "docstring): a throughput, not the latency of one search. Compare rows within a column.\n"
    )
    print(markdown_table("A \\ B", rows, cols, ms_per_move))

    commits = sorted({c for cell in cells.values() for c in cell.commits})
    stale = sum(cell.other_configs for cell in cells.values())
    seeds = sorted({s for cell in cells.values() for s in cell.seeds})
    print(f"\nSeeds pooled: {seeds}. Git commits of the shown lines: {', '.join(commits)}.")
    print("† all games had the same outcome: the half-width is the rule of three, 3 / games.")
    if any(cell.dirty for cell in cells.values()):
        print("Some lines were produced with uncommitted changes in src/ or baselines/.")
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
        epilog="MCTS suffixes (mcts-<sims>-<suffix>-...):\n" + suffix_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--a", help="agents under test, comma-separated (rows of the table)")
    p.add_argument("--b", help="opponents, comma-separated (columns); all pairs a x b are played")
    p.add_argument("--games", type=int, default=256, help="games per matchup (default 256)")
    p.add_argument(
        "--games-for",
        help="per-agent game counts: 'mcts-1024=64,mcts-256:dqn=128' ('A=N' applies to every "
        "matchup with that agent on either side, the smaller count wins; 'A:B=N' to one pair)",
    )
    p.add_argument(
        "--target-ci",
        type=float,
        help="stop a matchup after a round once the 95%% CI half-width "
        "is at most this (needs >= 64 games; --games stays the maximum)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="games per round, before the memory cap (default 256)",
    )
    p.add_argument(
        "--mem-mb",
        type=float,
        default=1024.0,
        help="memory budget (MB) for the search trees of one batch; the batch shrinks to fit "
        "(default 1024)",
    )
    p.add_argument(
        "--seed",
        type=int,
        help="base seed (default 0); with --table, only lines of this seed are shown",
    )
    p.add_argument("--max-turns", type=int, default=500, help="cut games off here (default 500)")
    p.add_argument("--size", type=int, default=11, help="board width and height (default 11)")
    p.add_argument("--snakes", type=int, default=2, help="snakes per game (default 2)")
    p.add_argument("--dqn-checkpoint", type=Path, default=DEFAULT_DQN, help="run dir of the DQN")
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="results file (default benchmarks/results/strength.jsonl)",
    )
    p.add_argument("--table", action="store_true", help="print markdown tables from --out and exit")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument(
        "--estimate",
        action="store_true",
        help="compile and probe every pending matchup (no games) and predict the wall time",
    )
    p.add_argument(
        "--assume-turns",
        type=int,
        help="--estimate: turns per round (default --max-turns, the worst case)",
    )
    return p.parse_args(argv)


def names_of(text: str | None, checkpoint: Path) -> set[str] | None:
    """Canonical agent names in a comma-separated list (None: no filter)."""
    if not text:
        return None
    return {a.name for a in parse_agents(text, checkpoint, describe=False)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.table:
            a_names = names_of(args.a, args.dqn_checkpoint)
            b_names = names_of(args.b, args.dqn_checkpoint)
            return table_main(args.out, a_names, b_names, args.seed)
        if not args.a or not args.b:
            raise SpecError("--a and --b are required (or use --table)")
        if min(args.games, args.batch_size, args.max_turns, args.size, args.snakes) < 1:
            raise SpecError("--games, --batch-size, --max-turns, --size and --snakes must be >= 1")
        game = GameConfig(width=args.size, height=args.size, num_snakes=args.snakes)
        s = Settings(
            game, args.max_turns, args.seed or 0, args.target_ci, args.batch_size, args.mem_mb
        )
        overrides = parse_overrides(args.games_for, args.dqn_checkpoint)
        matchups = build_matchups(
            parse_agents(args.a, args.dqn_checkpoint),
            parse_agents(args.b, args.dqn_checkpoint),
            args.games,
            overrides,
        )
        if any(m.uses_dqn for m in matchups):
            check_dqn_game(next(x for m in matchups for x in (m.a, m.b) if x.kind == "dqn"), game)
        if args.snakes > mcts_lib.MAX_SNAKES and any(
            x.kind == "mcts" for m in matchups for x in (m.a, m.b)
        ):
            raise SpecError(f"MCTS supports at most {mcts_lib.MAX_SNAKES} snakes")
        if game.solo:
            raise SpecError("need at least two snakes")
        if args.dry_run:
            done_ids = {r["matchup_id"] for r in read_records(args.out)[0]}
            for m in matchups:
                state = "done" if matchup_id(m, s) in done_ids else "todo"
                print(f"{m.a.name} vs {m.b.name}: {describe_plan(m, plan_batches(m, s))} [{state}]")
            return 0
        if args.estimate:
            return estimate(s, matchups, args.out, args.assume_turns or args.max_turns)
        return sweep(s, matchups, args.out)
    except SpecError as e:
        log(f"error: {e}")
        return 2
    except KeyboardInterrupt:
        log("interrupted: finished matchups are saved; rerun the same command to resume")
        return 130


if __name__ == "__main__":
    sys.exit(main())
