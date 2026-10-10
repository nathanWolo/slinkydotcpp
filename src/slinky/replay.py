"""Record games between agents as JSON replays and watch them in a self-contained HTML viewer.

::

    replay = record_games(GameConfig(), ["heuristic", "dqn"], jax.random.key(0), num_games=8)
    Path("replay.json").write_text(json.dumps(replay))
    Path("replay.html").write_text(render_html(replay))      # open in a browser

or from the command line (agents are names from :mod:`slinky.agents`)::

    python -m slinky.replay --a heuristic --b dqn --games 8 --seed 0 --max-turns 500 \\
        --out replays/heuristic-vs-dqn.json --html replays/heuristic-vs-dqn.html
    python -m slinky.replay --agents heuristic,random_legal,random_legal,random --games 4 \\
        --html replays/four-way.html

The viewer (``viewer.html``, inlined by :func:`render_html`) needs no server and
no network beyond Google Fonts. It can also open other replay files.

**Replay format** ``slinky-replay/1``, plain JSON::

    {"format": "slinky-replay/1",
     "board": {"width": 11, "height": 11}, "ruleset": "standard", "map": "standard",
     "agents": ["heuristic", "dqn"],          # the roster, in the order given; repeated
                                              # names get roster letters ("heuristic A")
     "max_turns": 500,
     "games": [
       {"id": 0,
        "seats": ["heuristic", "dqn"],        # agent name per seat (snake index)
        "result": {"winner": 0, "draw": false, "truncated": false, "turns": 187,
                   "elims": [{"seat": 1, "cause": "head-collision", "turn": 187}]},
        "frames": [
          {"turn": 0,
           "snakes": [{"body": [[5, 1], [5, 1], [5, 1]], "health": 100, "alive": true,
                       "length": 3}, ...],
           "food": [[5, 5], ...],
           "hazards": [[0, 10, 1], ...]},     # [x, y, number of stacked hazards]
          ...]},
       ...]}

* Coordinates are the Battlesnake API's: ``(0, 0)`` is the bottom-left cell.
* Frame ``t`` is the state at turn ``t``: frame 0 is the start and a game has
  ``turns + 1`` frames.
* ``body`` is head first and includes stacked segments, so ``len(body) ==
  length`` and the last two points are equal right after a snake eats.
* A snake eliminated on turn ``t`` is still drawn on frame ``t``: ``"alive":
  false``, ``"cause"`` (the engine's name: ``wall-collision``,
  ``snake-self-collision``, ``snake-collision``, ``head-collision``,
  ``out-of-health``, ``hazard``) and the body after its final move (for a
  ``wall-collision`` the head is off the board). Later frames keep
  ``alive``/``cause``/``health``/``length`` with an empty body.
* ``result``: ``winner`` is the seat of the last snake standing (always
  ``null`` in solo games). ``draw`` means every remaining snake was eliminated
  on the final turn. ``truncated`` means ``max_turns`` cut the game off before
  the rules ended it (no winner, not a draw). ``elims`` lists every elimination
  in turn order.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections.abc import Callable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from slinky.agents import Agent, make_agent, make_env
from slinky.engine_json import countdown_to_body
from slinky.env import BattlesnakeEnv
from slinky.evaluate import Policy
from slinky.types import CAUSE_TO_ENGINE, Cause, GameConfig, Ruleset, State

FORMAT = "slinky-replay/1"
_DATA_TAG = '<script type="application/json" id="replay-data">null</script>'
_RECORDED = (
    "body", "head", "length", "health", "alive", "food", "hazard", "turn",
    "elim_cause", "elim_turn", "done",
)  # fmt: skip


@functools.lru_cache(maxsize=16)
def _compiled(
    env: BattlesnakeEnv, policies: tuple[Policy, ...]
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Jitted ``(reset(keys), step(loop_keys, t, states, ts, seat_policy))`` for a batch."""
    k = len(policies)

    def step(loop_keys, t, states, ts, seat_policy):
        # Per game and turn: one key per distinct policy, one for the env.
        ks = jax.vmap(lambda key: jax.random.split(jax.random.fold_in(key, t), k + 1))(loop_keys)
        acts = jnp.stack([jax.vmap(p)(ks[:, i], states, ts) for i, p in enumerate(policies)])
        actions = jnp.take_along_axis(acts, seat_policy[None], axis=0)[0].astype(jnp.int32)
        return jax.vmap(env.step)(ks[:, k], states, actions)

    return jax.jit(jax.vmap(env.reset)), jax.jit(step)


def _seating(num_agents: int, num_snakes: int, num_games: int, rotate: bool | None) -> np.ndarray:
    """int[G, N]: the roster index of the agent in each seat of each game."""
    if rotate is None:
        rotate = num_agents == 2
    games = np.arange(num_games)[:, None]
    seats = np.arange(num_snakes)[None, :]
    if num_agents == 2 and rotate:
        return np.where(seats == games % num_snakes, 0, 1)  # A in seat g % N, as play_match
    if num_agents == num_snakes:
        if rotate:
            return (seats - games) % num_snakes
        return np.broadcast_to(seats, (num_games, num_snakes))
    raise ValueError(
        f"give one agent per seat ({num_snakes}) or two agents to rotate across seats, "
        f"not {num_agents}"
    )


def _roster_names(agents: Sequence[Agent]) -> list[str]:
    """Agent names, with duplicates told apart by roster letter (``"heuristic A"``, ``"... B"``).

    Letters, not numbers, so a suffix can't be mistaken for the (0-based) seat the
    viewer shows: with rotation the roster position and the seat differ anyway.
    """
    names = [a.name for a in agents]
    if len(set(names)) == len(names):
        return names
    tag = [chr(ord("A") + i) if i < 26 else str(i + 1) for i in range(len(names))]
    return [f"{n} {tag[i]}" if names.count(n) > 1 else n for i, n in enumerate(names)]


def record_games(
    config: GameConfig,
    agents: Sequence[str | Agent],
    key: jax.Array,
    num_games: int,
    max_turns: int = 500,
    rotate: bool | None = None,
) -> dict[str, Any]:
    """Play ``num_games`` games at once and return them as a replay dict (module docstring).

    Args:
      config: the game; ``config.num_snakes`` seats per game.
      agents: agent names (:func:`slinky.agents.make_agent`) or :class:`Agent`
        objects, one per seat, or two to rotate across seats: game ``g`` puts
        the first in seat ``g % N`` and the second in every other seat, as
        :func:`slinky.evaluate.play_match` does.
      key: PRNG key; the same key, agents and settings replay the same games.
      num_games: games to play (all in one batch).
      max_turns: games still going after this many turns are cut off
        (``truncated``). ``config.max_turns`` also applies if it is smaller.
      rotate: with two agents (default) rotate them as above; with one agent
        per seat (default: fixed seats) shift the seating by one seat per game.
    """
    if num_games < 1 or max_turns < 1:
        raise ValueError("num_games and max_turns must be >= 1")
    roster = [make_agent(a, config) if isinstance(a, str) else a for a in agents]
    n = config.num_snakes
    seating = _seating(len(roster), n, num_games, rotate)
    names = _roster_names(roster)

    # Distinct policies are evaluated once per turn; seats pick their own entries.
    policies: list[Policy] = []
    roster_policy = []
    for agent in roster:
        if agent.policy not in policies:
            policies.append(agent.policy)
        roster_policy.append(policies.index(agent.policy))
    seat_policy = jnp.asarray(np.asarray(roster_policy)[seating], jnp.int32)

    env = make_env(config, any(a.needs_obs for a in roster))
    reset, step = _compiled(env, tuple(policies))
    game_keys = jax.vmap(jax.random.split)(jax.random.split(key, num_games))
    states, ts = reset(game_keys[:, 0])
    history = [jax.device_get({f: getattr(states, f) for f in _RECORDED})]
    for t in range(max_turns):
        if history[-1]["done"].all():
            break
        states, ts = step(game_keys[:, 1], jnp.int32(t), states, ts, seat_policy)
        history.append(jax.device_get({f: getattr(states, f) for f in _RECORDED}))
    stacked = {f: np.stack([h[f] for h in history]) for f in _RECORDED}  # [T, G, ...]

    games = []
    for g in range(num_games):
        arrays = {f: v[:, g] for f, v in stacked.items()}
        game = _game_record(arrays, config)
        games.append({"id": g, "seats": [names[i] for i in seating[g]], **game})
    return {
        "format": FORMAT,
        "board": {"width": config.width, "height": config.height},
        "ruleset": config.ruleset.value,
        "map": config.map,
        "agents": names,
        "max_turns": max_turns if config.max_turns is None else min(max_turns, config.max_turns),
        "games": games,
    }


def _points(mask: np.ndarray) -> list[list[int]]:
    ys, xs = np.nonzero(mask)
    return [[int(x), int(y)] for x, y in zip(xs, ys, strict=True)]


def game_from_states(
    states: Sequence[State],
    config: GameConfig,
    seats: Sequence[str] | None = None,
    game_id: int = 0,
) -> dict[str, Any]:
    """One replay game (``{"id", "seats", "result", "frames"}``) from your own game loop.

    Args:
      states: the unbatched states of one game, one per turn from turn 0 (frame
        ``t`` is turn ``t``). Frames stop at the first finished state; later
        states are ignored.
      seats: agent name per seat (default ``"seat 0"``, ``"seat 1"``, ...).

    Raises:
      ValueError: if the states up to the first finished one are not turns
        0, 1, 2, ... (e.g. a game picked up mid-way, or a skipped turn).
    """
    if not states:
        raise ValueError("no states")
    host = jax.device_get([{f: getattr(s, f) for f in _RECORDED} for s in states])
    arrays = {f: np.stack([np.asarray(h[f]) for h in host]) for f in _RECORDED}
    done = np.nonzero(arrays["done"])[0]
    turns = arrays["turn"][: int(done[0]) + 1 if done.size else len(states)]
    if not np.array_equal(turns, np.arange(len(turns))):
        shown = ", ".join(str(int(t)) for t in turns[:6]) + (", ..." if len(turns) > 6 else "")
        raise ValueError(
            f"states must be consecutive turns from turn 0 (frame t is turn t), got turns {shown}"
        )
    seats = list(seats) if seats is not None else [f"seat {i}" for i in range(config.num_snakes)]
    return {"id": game_id, "seats": seats, **_game_record(arrays, config)}


def _game_record(a: dict[str, np.ndarray], config: GameConfig) -> dict[str, Any]:
    """``{"result", "frames"}`` for one game from its per-turn arrays (leading axis = turn)."""
    done = np.nonzero(a["done"])[0]
    end = int(done[0]) if done.size else len(a["done"]) - 1
    n, wrapped = config.num_snakes, config.ruleset.wrapped
    prev_body: list[list[list[int]]] = [[] for _ in range(n)]
    frames = []
    for t in range(end + 1):
        turn = int(a["turn"][t])
        snakes = []
        for i in range(n):
            length, head = int(a["length"][t, i]), a["head"][t, i]
            entry: dict[str, Any] = {"body": [], "health": int(a["health"][t, i])}
            if a["alive"][t, i]:
                body = countdown_to_body(a["body"][t, i], (head[0], head[1]), length, wrapped)
                prev_body[i] = body
                entry.update(body=body, alive=True, length=length)
            else:
                cause = CAUSE_TO_ENGINE[Cause(int(a["elim_cause"][t, i]))]
                if int(a["elim_turn"][t, i]) == turn and prev_body[i]:
                    # The engine moves every snake before eliminating any: the
                    # body after the fatal move (tail popped; grown if it ate).
                    body = [[int(head[0]), int(head[1])], *prev_body[i][:-1]]
                    body += [body[-1]] * (length - len(body))
                    entry["body"] = body
                entry.update(alive=False, length=length, cause=cause)
            snakes.append(entry)
        hazards = [[x, y, int(a["hazard"][t, y, x])] for x, y in _points(a["hazard"][t] > 0)]
        frames.append(
            {"turn": turn, "snakes": snakes, "food": _points(a["food"][t]), "hazards": hazards}
        )

    alive = a["alive"][end]
    remaining = int(alive.sum())
    solo = config.solo
    truncated = remaining >= (1 if solo else 2)
    elims = sorted(
        (int(a["elim_turn"][end, i]), i, CAUSE_TO_ENGINE[Cause(int(a["elim_cause"][end, i]))])
        for i in range(n)
        if a["elim_cause"][end, i] != Cause.NONE
    )
    result = {
        "winner": int(np.argmax(alive)) if not solo and not truncated and remaining == 1 else None,
        "draw": bool(not solo and remaining == 0),
        "truncated": bool(truncated),
        "turns": int(a["turn"][end]),
        "elims": [{"seat": i, "cause": c, "turn": t} for t, i, c in elims],
    }
    return {"result": result, "frames": frames}


def merge_replays(*replays: dict[str, Any]) -> dict[str, Any]:
    """One replay holding every game of ``replays`` (same board, ruleset and map), renumbered."""
    if not replays:
        raise ValueError("nothing to merge")
    first = replays[0]
    setting = ("board", "ruleset", "map")
    for r in replays[1:]:
        if any(r[k] != first[k] for k in setting):
            raise ValueError("can only merge replays with the same board, ruleset and map")
    agents = list(dict.fromkeys(a for r in replays for a in r["agents"]))
    games = [g for r in replays for g in r["games"]]
    return {
        **first,
        "agents": agents,
        "max_turns": max(r["max_turns"] for r in replays),
        "games": [{**g, "id": i} for i, g in enumerate(games)],
    }


def to_json(replay: dict[str, Any]) -> str:
    """Compact JSON for a replay (no whitespace)."""
    return json.dumps(replay, separators=(",", ":"))


def render_html(replay: dict[str, Any], standalone: bool = True) -> str:
    """The viewer page with ``replay`` embedded.

    Args:
      standalone: a complete HTML document to open locally. If False, the bare
        fragment (``<title>``, ``<style>``, markup and ``<script>``, no
        ``<html>``/``<body>``) for hosts that supply the document skeleton.
    """
    template = resources.files("slinky").joinpath("viewer.html").read_text(encoding="utf-8")
    if template.count(_DATA_TAG) != 1:
        raise RuntimeError("viewer.html must contain the replay-data placeholder exactly once")
    # "<" -> < keeps "</script>" and "<!--" in names from closing the tag; still valid JSON.
    data = to_json(replay).replace("<", "\\u003c")
    fragment = template.replace(_DATA_TAG, _DATA_TAG.replace(">null<", f">{data}<"))
    if not standalone:
        return fragment
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"</head>\n<body>\n{fragment}\n</body>\n</html>\n"
    )


def describe_game(game: dict[str, Any]) -> str:
    """One line: who played, who won and how the others went out."""
    seats, r = game["seats"], game["result"]
    if r["winner"] is not None:
        outcome = f"{seats[r['winner']]} (seat {r['winner']}) wins"
    elif r["truncated"]:
        outcome = "cut off by max_turns"
    elif r["draw"]:
        outcome = "draw"
    else:
        outcome = "over"
    elims = ", ".join(f"seat {e['seat']} {e['cause']} t{e['turn']}" for e in r["elims"])
    return f"game {game['id']:>3}  {' vs '.join(seats)}  {outcome} after {r['turns']} turns" + (
        f"  [{elims}]" if elims else ""
    )


def summarize(replay: dict[str, Any]) -> str:
    """``"heuristic 6 - 2 dqn (0 draws, 0 cut off)"`` (wins per roster agent)."""
    wins = dict.fromkeys(replay["agents"], 0)
    draws = cut = 0
    for game in replay["games"]:
        r = game["result"]
        if r["winner"] is not None:
            wins[game["seats"][r["winner"]]] += 1
        draws += r["draw"]
        cut += r["truncated"]
    if len(wins) == 2:
        (a, wa), (b, wb) = wins.items()
        head = f"{a} {wa} - {wb} {b}"
    else:
        head = ", ".join(f"{name} {w}" for name, w in wins.items())
    return f"{head} ({draws} draws, {cut} cut off)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m slinky.replay",
        description="Play games between agents and save them as a replay (JSON and/or HTML).",
        epilog="Agents: random_legal, random, heuristic, dqn, dqn:<run dir>, "
        "mcts-<n>[-shorthand...][:field=value...], e.g. mcts-256-rm or "
        "mcts-128:exploration=0.5 (see slinky.agents).",
    )
    p.add_argument("--a", help="first agent (rotated across seats with --b)")
    p.add_argument("--b", help="second agent")
    p.add_argument("--agents", help="comma-separated agents, one per seat (sets --snakes)")
    p.add_argument("--rotate", action="store_true", help="with --agents: shift seats each game")
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=500)
    p.add_argument("--snakes", type=int, default=None, help="seats per game (default 2)")
    p.add_argument("--width", type=int, default=11)
    p.add_argument("--height", type=int, default=11)
    p.add_argument("--ruleset", default="standard", choices=[r.value for r in Ruleset])
    p.add_argument("--map", default="standard")
    p.add_argument("--out", type=Path, help="write the replay JSON here")
    p.add_argument("--html", type=Path, help="write the viewer page here")
    p.add_argument(
        "--fragment", action="store_true", help="write the HTML as a fragment (no <html>/<body>)"
    )
    args = p.parse_args(argv)
    if args.agents and (args.a or args.b):
        p.error("use either --a/--b or --agents")
    if not args.agents and not (args.a and args.b):
        p.error("name two agents with --a and --b, or one per seat with --agents")
    if not args.out and not args.html:
        p.error("nothing to write: pass --out and/or --html")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.agents:
        agents = [a for a in args.agents.split(",") if a.strip()]
        snakes = args.snakes or len(agents)
        rotate: bool | None = args.rotate
    else:
        agents, snakes, rotate = [args.a, args.b], args.snakes or 2, None
    config = GameConfig(
        width=args.width,
        height=args.height,
        num_snakes=snakes,
        ruleset=Ruleset(args.ruleset),
        map=args.map,
    )
    try:
        replay = record_games(
            config, agents, jax.random.key(args.seed), args.games, args.max_turns, rotate
        )
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")
    for game in replay["games"]:
        print(describe_game(game))
    print(summarize(replay))
    for path, text in (
        (args.out, args.out and to_json(replay)),
        (args.html, args.html and render_html(replay, standalone=not args.fragment)),
    ):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path} ({len(text.encode()) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
