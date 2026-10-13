"""Mode benchmark: a generalist and its specialists at 4-player tables, per game mode.

Plays *cells* with :func:`slinky.evaluate.play_table` and appends one JSON line per
finished cell to ``benchmarks/results/modes.jsonl``. A cell is one game mode
(:mod:`slinky.modes`: ``standard``, ``duel``, ``royale`` or ``constrictor``), a line-up of
seats and the agent behind each seat symbol, for example::

    --game-mode standard --seats G,S,G,S --agent G=ppo_modes --agent S=ppo_modes:runs/std

Seat symbols are free names (letters, digits, ``_ . @ -``). Game ``g`` rotates the seats
by ``g`` (seat ``s`` holds ``seats[(s - g) % N]``), so every symbol spends equal time in
every seat of its line-up, and policies of one symbol share one table entry however many
seats they hold. The built-in symbols are ``H`` = ``heuristic``, ``R`` = ``random_legal`` and
``P`` = ``ppo`` (the duel checkpoint, a zero-shot reference in the 4-player modes);
``--agent`` sets or overrides any symbol. In the protocol ``G`` is the generalist
(``ppo_modes:<dir>``), ``S`` the mode's specialist and ``G@<mode>`` the generalist told a
different mode.

Usage::

    python benchmarks/modes.py --game-mode standard --seats G,H,H,H \\
        --agent G=ppo_modes:runs/ppo-modes-seed0 --agent H=heuristic --games 1024
    python benchmarks/modes.py --protocol --generalist runs/ppo-modes-seed0 \\
        --specialist standard=runs/ppo-modes-standard-seed0 --specialist duel=runs/... \\
        [--only duel,royale]
    python benchmarks/modes.py --table                    # tables from the results file
    python benchmarks/modes.py --protocol --generalist DIR --dry-run

**Measures** (per policy, from its seats; ``slinky.evaluate.TableResult``). *Share of the
win* is the primary measure: a sole survivor gets 1, ``k`` snakes tied at the end get
``1/k`` each (all eliminated on the last turn, or alive at the turn cap), summed over a
policy's seats, so the shares of a game sum to 1 and a policy with ``s`` of the ``N``
seats has parity ``s/N`` (0.25 for one seat in four, 0.5 for two). In a duel it is the
score of ``benchmarks/strength.py``. *Placement* is ``1 + (snakes that outlasted the seat)
+ 0.5 * (others out on the same turn)`` averaged over the policy's seats (parity
``(N+1)/2``). *W/T/L* counts games by the policy's share (1 / strictly between / 0).
Games are played with ``early_stop`` (decided once every living snake belongs to one
policy, which is exact for share and placement) and cut off after ``--max-turns`` turns
(default 1000, the trainer's), counting as ties between the survivors.

**Identity and resume.** A cell's id is a hash of the game mode and rules, the seats, the
canonical name and *full config* of every agent used (parameter hash, run config, the
mode it is told, the snapshot, the heuristic weights), ``--max-turns`` and ``--seed``:
everything but the number of games. A cell is finished if the file has its id with at least
the requested games; a larger request plays the cell again (game ``g`` depends only on the
key and ``g``, so the larger run contains the smaller one, up to the float rounding of a
network's batched inference, below) and the table uses the line with the most games. The
id covers configs, not code; each line records ``git_commit``, ``git_dirty``,
``code_fingerprint`` (the ``slinky`` modules loaded, the baseline scripts and this file)
and ``heuristic_fingerprint`` (``src/slinky/heuristic.py`` alone, the reference opponent
whose behaviour changes with its fixes). Several processes may share the file (one per core
with ``--only MODE`` each): lines are appended under a lock, and the file is read again
before each cell. ``--rerun`` plays the selected cells again.

**Randomness and pairing.** The key of a cell depends on ``--seed`` and the game mode only,
not on the agents or seats: every cell of a mode plays the same games ``0, 1, ...``, with the
same start positions and food (the environment's random stream does not depend on who
plays), so two cells that differ in one agent can be compared game by game with
*common random numbers*. For agents without a network (the heuristic, random play) the
results are the same for every ``--slots``. A network agent's moves come from batched
inference, of ``min(--slots, games)`` games at a time, which may round differently at
another batch size; that can flip a near-tied greedy argmax, after which that one game
goes on differently. So lines of one cell with different ``slots`` or game counts agree
game for game up to such rare flips (``slots`` is in each line). They do not break the
paired intervals: these are intervals of per-game differences, valid whether or not the
games are correlated; correlation only makes them narrower.

**Results.** One JSON object per line (``SCHEMA``): ``cell_id``, ``game_mode``, ``game``
(the :class:`slinky.types.GameConfig` fields), ``lineup`` (``"G,S,G,S"``), ``seats``,
``seat_names`` (canonical agent name per seat of game 0), ``agents`` (``{symbol: {name,
config}}``), ``max_turns``, ``seed``, ``games_requested``, ``games``, ``early_stop``,
``results`` (``{symbol: {seats, share, share_ci95, share_wilson, placement,
placement_ci95, wins, ties, losses, deaths}}``), ``truncated``, ``mean_turns`` (the turn
games were decided at), ``loop_iterations``, ``slots``, ``slot_utilization``,
``wall_seconds``, the provenance fields, ``jax_version``, ``backend`` and ``date``. The
per-game values are in ``per_game`` as compact strings (:func:`decode_per_game` reads them
back): per symbol, one fixed-width base-36 number per game id, the share in units of
``1/share_scale`` and twice the placement summed over the symbol's seats, which is
all that paired differences need.

**Protocol** (``--protocol``). For every mode (``--only`` selects), with ``G`` the
generalist, ``S`` the mode's specialist (cells with ``S`` are skipped without one),
``H`` the heuristic, ``R`` ``random_legal`` and ``P`` the duel ``ppo`` checkpoint:

* 4-snake modes: ``X,R,R,R`` (256 games); ``X,H,H,H``, ``X,P,P,P``, ``X,H,R,R`` (1024)
  for ``X`` in ``G`` and ``S``, which are mid-strength fields, so neither floor nor
  ceiling hides a gap; ``P,H,H,H`` (512, the zero-shot duel row); ``G,S,G,S`` (1024),
  **the pre-registered primary endpoint** (G's team share, parity 0.5), and ``G,S,H,H``
  (1024), the co-headline (G's share minus S's, paired within each game).
* the duel: ``G,S`` (1024, the primary endpoint); ``X,H`` and ``X,P`` (1024); ``X,R``
  (256).
* every mode: ``G@m,H,H,H`` (``G,H`` in a duel) for each other mode ``m`` (256 games each):
  the generalist told a wrong mode, which shows whether the mode input is used. Food is
  hidden by the real game, so this does not measure food hiding.
* ``--matrix`` adds each other specialist, told its own mode, in this mode against three
  heuristics (256 games), the cross-mode transfer matrix. The specialist of mode ``m``
  sits under its own symbol ``S_<m>``, so each has its own row (``S_duel,H,H,H``).

Cells run most important first (the primary endpoint, the co-headline, then the rest).
``--games N`` plays every cell with ``N`` games instead (smoke runs).

**Tables.** ``--table`` prints, per game mode, one row per cell: the games, and per policy
the share with its 95% Wilson interval (the right choice near 0 and 1, where the
mid-strength fields are meant to keep results away from but the floor cells are not),
mean placement +- 95% half-width, and W/T/L; then **paired differences**, the G-versus-S
comparison in *share per seat* (``share / seats``, so parity is 0 whatever the line-up):
across two cells that differ only in ``G`` versus ``S`` in the same seats (``G,H,H,H`` and
``S,H,H,H``), per game over the games both played, and within one cell holding both
(``G,S,H,H``, ``G,S,G,S``, the duel's ``G,S``). The paired half-width is the 95% normal
interval of the per-game differences; the unpaired one (cells compared as independent
samples) is printed beside it. Two lines are paired across cells only if they were
produced by the same code, the same ``code_fingerprint`` and in particular the same
``heuristic_fingerprint`` (the opponent's play changes with its fixes, and the cell id
does not cover code); otherwise the row says so and shows the unpaired comparison only.
A second table gives the wrong-mode ablation the same way: ``G`` minus ``G@m`` in
``G,H,H,H`` against ``G@m,H,H,H``, the same games, which answers whether the mode input is
used (positive: telling the right mode helps). Lines of the same cell id are not pooled:
the one with the most games is shown. With several seeds in the file (``--seed`` selects
one) every row and comparison carries its seed.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # strength.py's helpers (a sibling)
import strength  # noqa: E402

from slinky import agents as registry  # noqa: E402
from slinky import heuristic, mcts, modes, observations  # noqa: E402,F401  (in the fingerprint)
from slinky.evaluate import TableResult, run_table  # noqa: E402
from slinky.types import GameConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "modes.jsonl"
SCHEMA = 1
MAX_TURNS = 1000  # the trainer's
SLOTS = 32  # games simulated at once: the cheapest per game-turn on one core (see strength.py)
Z95 = strength.Z95
# Seat symbols with a built-in meaning; --agent overrides them.
DEFAULT_AGENTS = {"H": "heuristic", "R": "random_legal", "P": "ppo"}
SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_.@-]*")
REQUIRED = frozenset(
    ("cell_id", "game_mode", "lineup", "seats", "agents", "games", "results", "per_game", "date")
)
SpecError = strength.SpecError
log = strength.log


# --- Cells: what is played ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Cell:
    """One game mode, a line-up and the agents behind its seat symbols."""

    game_mode: str
    seats: tuple[str, ...]  # the symbol of each seat in game 0
    agents: dict[str, strength.Agent]  # symbol -> agent, for the symbols used
    games: int  # requested

    @property
    def policies(self) -> tuple[str, ...]:
        """The distinct symbols, in order of first seat (the table's policy order)."""
        return tuple(dict.fromkeys(self.seats))

    @property
    def lineup(self) -> tuple[int, ...]:
        return tuple(self.policies.index(s) for s in self.seats)

    @property
    def label(self) -> str:
        return ",".join(self.seats)

    @property
    def game(self) -> GameConfig:
        return modes.mode_config(self.game_mode, len(self.seats))


@dataclasses.dataclass(frozen=True)
class Settings:
    max_turns: int
    seed: int
    slots: int


def game_dict(game: GameConfig) -> dict[str, Any]:
    return strength.game_dict(game)


def _digest(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def cell_id(cell: Cell, s: Settings) -> str:
    """Hash of everything that defines a cell's result but the number of games."""
    agents = [[x, cell.agents[x].name, cell.agents[x].config] for x in cell.policies]
    return _digest(
        [cell.game_mode, game_dict(cell.game), list(cell.seats), agents, s.max_turns, s.seed]
    )


def cell_key(seed: int, game_mode: str) -> jax.Array:
    """The key of every cell of a mode: game ``g`` is the same game in all of them."""
    digest = hashlib.sha256(f"modes\0{game_mode}".encode()).digest()
    return jax.random.fold_in(jax.random.key(seed), int.from_bytes(digest[:4], "big") >> 1)


def check_symbol(symbol: str) -> str:
    if not SYMBOL.fullmatch(symbol):
        raise SpecError(
            f"bad seat symbol {symbol!r}: start with a letter, then letters, digits and _ . @ -"
        )
    return symbol


def make_cell(game_mode: str, seats: list[str], agent_names: dict[str, str], games: int) -> Cell:
    """A cell, with its agents loaded and checked against the game."""
    if game_mode not in modes.MODE_NAMES:
        raise SpecError(f"unknown game mode {game_mode!r}; expected one of {modes.MODE_NAMES}")
    seats = [check_symbol(x) for x in seats]
    try:
        game = modes.mode_config(game_mode, len(seats))
    except ValueError as e:
        raise SpecError(f"--seats {','.join(seats)}: {e}") from None
    loaded = {}
    for symbol in dict.fromkeys(seats):
        if symbol not in agent_names:
            raise SpecError(f"no agent for seat symbol {symbol!r}; pass --agent {symbol}=<name>")
        agent = strength.load_agent(agent_names[symbol])
        try:
            registry.check_game(agent.spec, game)
        except (ValueError, FileNotFoundError) as e:
            raise SpecError(f"{symbol}={agent.name}: {e}") from None
        loaded[symbol] = agent
    if games < 1:
        raise SpecError("--games must be >= 1")
    return Cell(game_mode, tuple(seats), loaded, games)


# --- The protocol ----------------------------------------------------------------------


def protocol_cells(
    generalist: str,
    specialists: dict[str, str],
    only: tuple[str, ...] | None = None,
    games: int | None = None,
    matrix: bool = False,
) -> list[Cell]:
    """The cells of the module docstring's protocol, most important first.

    Args:
      generalist: the generalist's run directory (``ppo_modes:<dir>``).
      specialists: ``{game mode: run directory}``; a mode without one has no ``S`` cells.
      only: restrict to these game modes (default: all four).
      games: play every cell this many games instead of the protocol's counts.
      matrix: also the cross-mode transfer cells.
    """
    for mode in specialists:
        if mode not in modes.MODE_NAMES:
            raise SpecError(
                f"--specialist {mode}=...: unknown game mode; expected {modes.MODE_NAMES}"
            )
    for mode in only or ():
        if mode not in modes.MODE_NAMES:
            raise SpecError(f"--only {mode}: unknown game mode; expected one of {modes.MODE_NAMES}")
    cells: list[Cell] = []
    for mode in only or modes.MODE_NAMES:
        agents = {**DEFAULT_AGENTS, "G": f"ppo_modes:{generalist}"}
        for other in modes.MODE_NAMES:
            if other != mode:
                agents[f"G@{other}"] = f"ppo_modes:{generalist}:game_mode={other}"
        if mode in specialists:
            agents["S"] = f"ppo_modes:{specialists[mode]}"
        xs = "GS" if "S" in agents else "G"  # the policies under test
        four = modes.DEFAULT_SNAKES[mode] > 2
        core: list[tuple[str, int]] = []  # (seats, games), most important first
        if four:
            if "S" in agents:
                core += [("G,S,G,S", 1024), ("G,S,H,H", 1024)]  # primary endpoint, co-headline
            for x in xs:
                core += [(f"{x},H,H,H", 1024), (f"{x},P,P,P", 1024), (f"{x},H,R,R", 1024)]
            core.append(("P,H,H,H", 512))
            core += [(f"{x},R,R,R", 256) for x in xs]
        else:
            if "S" in agents:
                core.append(("G,S", 1024))  # the primary endpoint
            core += [(f"{x},{o}", n) for o, n in (("H", 1024), ("P", 1024), ("R", 256)) for x in xs]
        for seats, n in core:
            cells.append(make_cell(mode, seats.split(","), agents, games or n))
        # The generalist told a wrong mode, and the cross-mode matrix: a run that was not
        # trained on a mode can't be told it, which skips the cell.
        extra = [
            (f"G@{o}" + (",H,H,H" if four else ",H"), agents, 256)
            for o in modes.MODE_NAMES
            if o != mode
        ]
        if matrix:
            for other, run in specialists.items():
                if other != mode:
                    sym = f"S_{other}"  # one symbol each: the rows must tell them apart
                    told = {**agents, sym: f"ppo_modes:{run}:game_mode={other}"}
                    extra.append((f"{sym},H,H,H" if four else f"{sym},H", told, 256))
        for seats, names, n in extra:
            try:
                cells.append(make_cell(mode, seats.split(","), names, games or n))
            except SpecError as e:
                log(f"skipping {mode} {seats}: {e}")
    return cells


# --- Running a cell ---------------------------------------------------------------------


def base36(values: np.ndarray, width: int) -> str:
    return "".join(np.base_repr(int(v), 36).zfill(width) for v in values)


def unbase36(text: str, width: int) -> np.ndarray:
    return np.array([int(text[i : i + width], 36) for i in range(0, len(text), width)], np.int64)


def per_game_record(cell: Cell, r: TableResult) -> dict[str, Any]:
    """The per-game values of a result as compact strings (see :func:`decode_per_game`)."""
    n = len(cell.seats)
    scale = math.lcm(*range(1, n + 1))
    width = len(np.base_repr(max(scale, 2 * n * n), 36))
    assert r.game_share is not None and r.game_placement is not None
    share = np.rint(r.game_share.astype(np.float64) * scale).astype(np.int64)
    twice = np.rint(r.game_placement.astype(np.float64) * 2 * np.array(r.seats)).astype(np.int64)
    return {
        "encoding": "base36",
        "width": width,
        "share_scale": scale,
        "share": {x: base36(share[:, i], width) for i, x in enumerate(cell.policies)},
        "place2": {x: base36(twice[:, i], width) for i, x in enumerate(cell.policies)},
    }


def decode_per_game(record: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """``{symbol: {"share": float[games], "placement": float[games]}}`` of a result line.

    The share is the policy's share of the win in each game (sum of its seats), the placement
    its mean placement over its seats; both indexed by game id.
    """
    pg = record["per_game"]
    out = {}
    for symbol, text in pg["share"].items():
        held = record["results"][symbol]["seats"]
        share = unbase36(text, pg["width"]) / pg["share_scale"]
        place = unbase36(pg["place2"][symbol], pg["width"]) / (2.0 * held)
        out[symbol] = {"share": share, "placement": place}
    return out


@functools.cache
def code_fingerprint() -> tuple[str, int]:
    """Hash of the slinky modules loaded, the baseline scripts and this file, with the count."""
    files = {"benchmarks/modes.py": Path(__file__)}
    for script, name, _ in registry.BASELINES.values():
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


def heuristic_fingerprint() -> str:
    """Hash of ``slinky/heuristic.py``: the reference opponent, whose play changes with fixes."""
    return hashlib.sha256(Path(heuristic.__file__).read_bytes()).hexdigest()[:16]


@functools.cache
def provenance() -> dict[str, Any]:
    commit, dirty = strength.git_info()
    fingerprint, files = code_fingerprint()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "code_fingerprint": fingerprint,
        "code_files": files,
        "heuristic_fingerprint": heuristic_fingerprint(),
    }


def run_cell(cell: Cell, s: Settings, tag: str) -> dict[str, Any]:
    """Play a cell and return its result line."""
    t0 = time.perf_counter()
    game = cell.game
    built = {x: registry.make_agent(cell.agents[x].spec, game) for x in cell.policies}
    env = registry.make_env(game, any(a.needs_obs for a in built.values()))
    policies = tuple(built[x].policy for x in cell.policies)
    run = run_table(
        env,
        policies,
        cell.lineup,
        cell_key(s.seed, cell.game_mode),
        cell.games,
        max_turns=s.max_turns,
        batch_size=s.slots,
        per_game=True,
        progress=strength.progress_reporter(f"{tag}   ...", cell.games),
    )
    r = run.result
    wall = time.perf_counter() - t0
    names = {x: cell.agents[x].name for x in cell.policies}
    results = {
        x: {
            "seats": r.seats[i],
            "share": r.share[i],
            "share_ci95": r.share_ci95[i],
            "share_wilson": list(r.share_wilson[i]),
            "placement": r.placement[i],
            "placement_ci95": r.placement_ci95[i],
            "wins": r.wins[i],
            "ties": r.ties[i],
            "losses": r.losses[i],
            "deaths": r.deaths[i],
        }
        for i, x in enumerate(cell.policies)
    }
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "schema": SCHEMA,
        "cell_id": cell_id(cell, s),
        "game_mode": cell.game_mode,
        "game": game_dict(game),
        "lineup": cell.label,
        "seats": list(cell.seats),
        "seat_names": [names[x] for x in cell.seats],
        "agents": {x: {"name": names[x], "config": cell.agents[x].config} for x in cell.policies},
        "max_turns": s.max_turns,
        "seed": s.seed,
        "games_requested": cell.games,
        "games": r.num_games,
        "early_stop": True,
        "results": results,
        "truncated": r.truncated,
        "mean_turns": r.mean_turns,
        "loop_iterations": run.iterations,
        "slots": run.slots,
        "slot_utilization": run.utilization,
        "wall_seconds": wall,
        "per_game": per_game_record(cell, r),
        **provenance(),
        "worker_core": affinity[0] if affinity and len(affinity) == 1 else None,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --- The results file --------------------------------------------------------------------


def read_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """``(result lines, number of unreadable or foreign lines)``."""
    records, bad = [], 0
    if not path.is_file():
        return records, bad
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("schema") != SCHEMA or not REQUIRED.issubset(rec):
                    raise ValueError("not a result line")
                records.append(rec)
            except (ValueError, TypeError, AttributeError):
                bad += 1
    return records, bad


def finished_games(path: Path) -> dict[str, int]:
    """``{cell id: most games any line of the file has played}``."""
    best: dict[str, int] = {}
    for r in read_records(path)[0]:
        best[r["cell_id"]] = max(best.get(r["cell_id"], 0), r["games"])
    return best


def is_done(cell: Cell, s: Settings, finished: dict[str, int]) -> bool:
    return finished.get(cell_id(cell, s), 0) >= cell.games


def describe(r: dict[str, Any]) -> str:
    parts = []
    for x, v in r["results"].items():
        lo, hi = v["share_wilson"]
        parts.append(f"{x} share {v['share']:.3f} [{lo:.3f}, {hi:.3f}] place {v['placement']:.2f}")
    return (
        f"{r['games']} games ({r['truncated']} truncated, decided at turn {r['mean_turns']:.0f}): "
        + "; ".join(parts)
        + f" in {strength.fmt_duration(r['wall_seconds'])}"
    )


def agents_text(cell: Cell) -> str:
    return " ".join(f"{x}={cell.agents[x].name}" for x in cell.policies)


def sweep(cells: list[Cell], s: Settings, out: Path, rerun: bool, dry_run: bool = False) -> int:
    """Play the cells that are not finished (all of them with ``rerun``)."""
    started = finished_games(out)
    todo = [c for c in cells if rerun or not is_done(c, s, started)]
    log(f"{len(cells)} cell(s), {len(cells) - len(todo)} finished, {len(todo)} to play")
    if dry_run:
        for c in cells:
            state = "play" if c in todo else "done"
            log(f"  {state}  {c.game_mode:>11}  {c.label:<14} {c.games:>5} games  {agents_text(c)}")
        return 0
    for i, cell in enumerate(todo, 1):
        tag = f"[{i}/{len(todo)}] {cell.game_mode} {cell.label}"
        # Another process may have finished it since we started.
        if not rerun and is_done(cell, s, finished_games(out)):
            log(f"{tag}: finished by another process, skipping")
            continue
        log(f"{tag}: {cell.games} games, {agents_text(cell)}")
        record = run_cell(cell, s, tag)
        strength.append_record(out, record)
        log(f"{tag} done: {describe(record)}")
    return 0


# --- Tables ------------------------------------------------------------------------------


def best_records(records: list[dict[str, Any]], seed: int | None, mode: str | None):
    """One line per cell id (the one with the most games, then the newest), in file order."""
    chosen: dict[str, tuple[int, dict[str, Any]]] = {}
    for i, r in enumerate(records):
        if (seed is not None and r["seed"] != seed) or (
            mode is not None and r["game_mode"] != mode
        ):
            continue
        old = chosen.get(r["cell_id"])
        if old is None:
            chosen[r["cell_id"]] = (i, r)
        elif (r["games"], r["date"]) >= (old[1]["games"], old[1]["date"]):
            chosen[r["cell_id"]] = (old[0], r)  # keeps the first position
    return [r for _, r in sorted(chosen.values(), key=lambda t: t[0])]


@dataclasses.dataclass(frozen=True)
class Paired:
    """A paired comparison of two policies' share per seat, game by game."""

    games: int
    mean: float
    half: float  # 95% half-width of the mean of the per-game differences
    unpaired_half: float | None  # if the two sets of games were independent samples
    place_mean: float
    place_half: float
    note: str = ""  # why the two lines are not paired (then every interval is the unpaired one)


def paired_stats(
    a: np.ndarray,
    b: np.ndarray,
    pa: np.ndarray,
    pb: np.ndarray,
    independent: bool,
    note: str = "",
) -> Paired:
    """Mean and interval of ``a - b`` (shares per seat) and of ``pa - pb`` (placements).

    With a ``note`` the two sets of games are not to be taken as the same games (the lines
    come from different code): the intervals are those of independent samples.
    """
    n = len(a)
    nan = float("nan")

    def half(x: np.ndarray) -> float:
        return Z95 * float(np.std(x, ddof=1)) / math.sqrt(n) if n > 1 else nan

    def apart(x: np.ndarray, y: np.ndarray) -> float:
        if n < 2:
            return nan
        return Z95 * math.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / n)

    d, dp = a - b, pa - pb
    if note:
        return Paired(
            n, float(d.mean()), apart(a, b), apart(a, b), float(dp.mean()), apart(pa, pb), note
        )
    unpaired = apart(a, b) if independent and n > 1 else None
    return Paired(n, float(d.mean()), half(d), unpaired, float(dp.mean()), half(dp))


def version_note(ra: dict[str, Any], rb: dict[str, Any]) -> str:
    """Why two lines can't be paired across cells (different code), or ``""``."""
    if ra.get("heuristic_fingerprint") != rb.get("heuristic_fingerprint"):
        return "different heuristic version"
    if ra.get("code_fingerprint") != rb.get("code_fingerprint"):
        return "different code version"
    return ""


def seed_tag(record: dict[str, Any], several: bool) -> str:
    return f" [seed {record['seed']}]" if several else ""


def pair_records(
    records: list[dict[str, Any]], a: str = "G", b: str = "S"
) -> list[tuple[str, Paired]]:
    """Paired ``a - b`` comparisons among the cells of one game mode.

    ``(title, stats)``. Across cells: ``a``'s cell and ``b``'s cell have the same seats
    once ``a`` is replaced by ``b``, the same other agents, rules, ``max_turns`` and seed.
    They are paired game by game only if both lines come from the same code (same
    ``code_fingerprint``, same ``heuristic_fingerprint``); lines of different code give a
    row with a ``note`` and the unpaired intervals. Within a cell: both symbols have seats.
    The seed is in the title when the records have several.
    """
    out = []
    several = len({r["seed"] for r in records}) > 1
    for r in records:
        if a in r["results"] and b in r["results"]:
            d = decode_per_game(r)
            held = r["results"]
            out.append(
                (
                    f"{r['lineup']} (within){seed_tag(r, several)}",
                    paired_stats(
                        d[a]["share"] / held[a]["seats"],
                        d[b]["share"] / held[b]["seats"],
                        d[a]["placement"],
                        d[b]["placement"],
                        independent=False,
                    ),
                )
            )
    for ra, rb in itertools.permutations(records, 2):
        if (
            a not in ra["results"]
            or b in ra["results"]
            or b not in rb["results"]
            or a in rb["results"]
            or ra["game_mode"] != rb["game_mode"]
            or ra["seed"] != rb["seed"]
            or ra["max_turns"] != rb["max_turns"]
            or ra["game"] != rb["game"]
            or [b if x == a else x for x in ra["seats"]] != rb["seats"]
            or any(ra["agents"][x] != rb["agents"][x] for x in ra["agents"] if x != a)
        ):
            continue
        da, db = decode_per_game(ra), decode_per_game(rb)
        n = min(len(da[a]["share"]), len(db[b]["share"]))
        sa, sb = ra["results"][a]["seats"], rb["results"][b]["seats"]
        out.append(
            (
                f"{ra['lineup']} vs {rb['lineup']}{seed_tag(ra, several)}",
                paired_stats(
                    da[a]["share"][:n] / sa,
                    db[b]["share"][:n] / sb,
                    da[a]["placement"][:n],
                    db[b]["placement"][:n],
                    independent=True,
                    note=version_note(ra, rb),
                ),
            )
        )
    return out


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


ROLES = {"G,S,G,S": "primary", "G,S": "primary", "G,S,H,H": "co-headline"}


def pairs_table(pairs: list[tuple[str, Paired]], a: str, b: str) -> str:
    """The markdown table of paired ``a - b`` comparisons."""
    rows = []
    for title, p in pairs:
        share = f"{p.mean:+.3f} +- {p.half:.3f}"
        if p.note:  # the intervals are the unpaired ones
            share += " (unpaired)"
            unpaired = f"not paired: {p.note}"
        else:
            unpaired = "-" if p.unpaired_half is None else f"+- {p.unpaired_half:.3f}"
        rows.append(
            [title, str(p.games), share, unpaired, f"{p.place_mean:+.2f} +- {p.place_half:.2f}"]
        )
    header = ["Comparison", "Games", f"Share {a} - {b}", "Unpaired half-width"]
    return markdown_table([*header, f"Placement {a} - {b}"], rows)


def table_main(path: Path, mode: str | None, seed: int | None) -> int:
    records, bad = read_records(path)
    if bad:
        log(f"warning: skipped {bad} unreadable line(s) in {path}")
    chosen = best_records(records, seed, mode)
    if not chosen:
        log(f"no results in {path}" + (" for these filters" if records else ""))
        return 1
    print("Share = share of the win (sole survivor 1, k tied snakes 1/k each, summed over the")
    print("policy's seats), [95% Wilson interval]; parity is seats/N. Placement: 1 is best,")
    print("+- 95% half-width, parity (N+1)/2. W/T/L: games with share 1 / between / 0.\n")
    for game_mode in modes.MODE_NAMES:
        recs = [r for r in chosen if r["game_mode"] == game_mode]
        if not recs:
            continue
        several = len({r["seed"] for r in recs}) > 1  # then every row says which seed
        print(f"## {game_mode}\n")
        rows = []
        for r in recs:
            share, place, wtl = [], [], []
            for x, v in r["results"].items():
                lo, hi = v["share_wilson"]
                share.append(f"{x} {v['share']:.3f} [{lo:.3f}, {hi:.3f}]")
                place.append(f"{x} {v['placement']:.2f} +- {v['placement_ci95']:.2f}")
                wtl.append(f"{x} {v['wins']}/{v['ties']}/{v['losses']}")
            role = ROLES.get(r["lineup"]) if {"G", "S"} <= set(r["results"]) else None
            rows.append(
                [
                    r["lineup"] + (f" ({role})" if role else "") + seed_tag(r, several),
                    str(r["games"]),
                    "; ".join(share),
                    "; ".join(place),
                    "; ".join(wtl),
                    str(r["truncated"]),
                    f"{r['mean_turns']:.0f}",
                ]
            )
        print(
            markdown_table(
                ["Line-up", "Games", "Share", "Placement", "W / T / L", "Truncated", "Decided at"],
                rows,
            )
        )
        names: dict[str, set[str]] = {}
        for r in recs:
            for x, v in r["agents"].items():
                names.setdefault(x, set()).add(v["name"])
        print("\nAgents: " + "; ".join(f"{x} = {' / '.join(sorted(n))}" for x, n in names.items()))
        pairs = pair_records(recs)
        if pairs:
            print(
                "\nG - S, paired game by game, in share per seat (positive favours G; placement"
                " difference negative favours G):\n"
            )
            print(pairs_table(pairs, "G", "S"))
        # The wrong-mode ablation: is the mode input used?
        wrong = [f"G@{m}" for m in modes.MODE_NAMES if any(f"G@{m}" in r["results"] for r in recs)]
        ablation = [pair for x in wrong for pair in pair_records(recs, a="G", b=x)]
        if ablation:
            print(
                "\nWrong-mode ablation, G - G@m: G told the game's mode minus G told another mode"
                " m, paired game by game, in share per seat (positive means the mode input helps;"
                " placement difference negative means it helps):\n"
            )
            print(pairs_table(ablation, "G", "G@m"))
        print()
    commits = sorted({r["git_commit"] for r in chosen})
    fingerprints = sorted({r["code_fingerprint"] for r in chosen})
    heuristics = sorted(
        {r["heuristic_fingerprint"] for r in chosen if "heuristic_fingerprint" in r}
    )
    print(
        f"Git commits of the shown lines: {', '.join(commits)}; code fingerprints: "
        f"{', '.join(fingerprints)}; heuristic fingerprints: {', '.join(heuristics)}."
    )
    if any(r["git_dirty"] for r in chosen):
        print(
            "Some lines were produced with uncommitted changes in src/, baselines/ or benchmarks/."
        )
    if len(fingerprints) > 1:
        log(
            f"warning: the table mixes results of {len(fingerprints)} code versions "
            f"(code_fingerprint {', '.join(fingerprints)}); --rerun the older cells"
        )
    if len(heuristics) > 1:
        log("warning: the table mixes results against different versions of the heuristic")
    return 0


# --- Command line ------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        epilog="Agents are names from slinky.agents (random_legal, heuristic, ppo, "
        "ppo_modes[:<run dir>][:game_mode=<mode>], ...). Built-in seat symbols: "
        + ", ".join(f"{k}={v}" for k, v in DEFAULT_AGENTS.items())
        + ".",
        allow_abbrev=False,
    )
    p.add_argument("--game-mode", choices=modes.MODE_NAMES, help="single cell: the game mode")
    p.add_argument("--seats", help="single cell: one seat symbol per snake, e.g. G,H,H,H")
    p.add_argument(
        "--agent",
        action="append",
        default=[],
        metavar="SYMBOL=NAME",
        help="single cell: the agent behind a seat symbol (repeatable)",
    )
    p.add_argument(
        "--games",
        type=int,
        default=None,
        help="games per cell (single cell: default 1024; --protocol: every cell plays this many "
        "instead of the protocol's counts)",
    )
    p.add_argument("--protocol", action="store_true", help="play the protocol's cells")
    p.add_argument("--generalist", help="--protocol: the generalist's run directory")
    p.add_argument(
        "--specialist",
        action="append",
        default=[],
        metavar="MODE=DIR",
        help="--protocol: the specialist of a game mode (repeatable)",
    )
    p.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="MODE[,MODE...]",
        help="--protocol: only these game modes (repeatable); run one process per mode, each "
        "pinned to a core, to use the whole machine",
    )
    p.add_argument(
        "--matrix",
        action="store_true",
        help="--protocol: also the cross-mode matrix (each other specialist told its own mode)",
    )
    p.add_argument("--table", action="store_true", help="print markdown tables from --out and exit")
    p.add_argument("--dry-run", action="store_true", help="list the cells and which are done")
    p.add_argument("--rerun", action="store_true", help="play the selected cells again")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base seed (default 0); with --table, only lines of this seed",
    )
    p.add_argument("--max-turns", type=int, default=MAX_TURNS, help="cut games off here (1000)")
    p.add_argument("--slots", type=int, default=SLOTS, help=f"games at once (default {SLOTS})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results file")
    return p.parse_args(argv)


def parse_pairs(items: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise SpecError(f"{what} {item!r}: write {what} NAME=VALUE")
        if key.strip() in out:
            raise SpecError(f"{what} {key.strip()!r} given twice")
        out[key.strip()] = value.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.table:
            return table_main(args.out, args.game_mode, args.seed)
        if min(args.max_turns, args.slots) < 1 or (args.games is not None and args.games < 1):
            raise SpecError("--games, --max-turns and --slots must be >= 1")
        s = Settings(args.max_turns, args.seed or 0, args.slots)
        if args.protocol:
            if args.game_mode or args.seats or args.agent:
                raise SpecError(
                    "--protocol takes --generalist, --specialist and --only, not a cell"
                )
            if not args.generalist:
                raise SpecError("--protocol needs --generalist <run dir>")
            only = tuple(m.strip() for item in args.only for m in item.split(",") if m.strip())
            cells = protocol_cells(
                args.generalist,
                parse_pairs(args.specialist, "--specialist"),
                only or None,
                args.games,
                args.matrix,
            )
        else:
            if not (args.game_mode and args.seats):
                raise SpecError(
                    "name a cell with --game-mode and --seats, or use --protocol or --table"
                )
            if args.generalist or args.specialist or args.only or args.matrix:
                raise SpecError(
                    "--generalist, --specialist, --only and --matrix are for --protocol"
                )
            names = {**DEFAULT_AGENTS, **parse_pairs(args.agent, "--agent")}
            seats = [x.strip() for x in args.seats.split(",")]
            cells = [make_cell(args.game_mode, seats, names, args.games or 1024)]
        return sweep(cells, s, args.out, args.rerun, args.dry_run)
    except SpecError as e:
        log(f"error: {e}")
        return 2
    except KeyboardInterrupt:
        log("interrupted: finished cells are saved; rerun the same command to resume")
        return 130


if __name__ == "__main__":
    sys.exit(main())
