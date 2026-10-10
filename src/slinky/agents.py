"""A registry of named agents, shared by tools (replays) and benchmarks.

An agent is an ``evaluate.Policy`` with a name::

    agent = make_agent("heuristic", GameConfig())
    actions = agent.policy(key, state, ts)        # int32[N], one move per snake

Names:

* ``random_legal``: uniformly random among the moves in ``env.action_mask``
  (:func:`slinky.evaluate.random_legal`).
* ``random``: uniformly random over all four moves, legal or not.
* ``heuristic``: the hand-written snake (:func:`slinky.heuristic.heuristic`).
* ``dqn`` or ``dqn:<run dir>``: the greedy self-play DQN baseline
  (``baselines/dqn.py``, legal argmax of the Q-values). The default checkpoint
  is ``baselines/checkpoints/dqn-duel-seed0``. It reads egocentric
  observations (``needs_obs``), so it only plays on the board size it was
  trained on (11x11, not wrapped). ``baselines/`` is not part of the installed
  package: this agent needs a checkout of the repository.
* ``mcts-<n>``: simultaneous-move MCTS (:func:`slinky.mcts.mcts`) with ``n``
  simulations per move. Other :class:`slinky.mcts.MCTSConfig` fields can be set
  after colons, e.g. ``mcts-128:exploration=1.0:selection=rm``. Values are
  converted to the type of the field's default (``true``/``false`` for
  booleans).

Agents are cached per ``(name, config)``: calling :func:`make_agent` twice
returns the same policy object, which keeps ``play_match``'s jit cache warm.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import jax

from slinky.env import BattlesnakeEnv
from slinky.evaluate import Policy, greedy_from_q, random_legal
from slinky.policies import random_policy
from slinky.types import GameConfig, State, TimeStep

REPO_ROOT = Path(__file__).resolve().parents[2]
DQN_SCRIPT = REPO_ROOT / "baselines" / "dqn.py"
DEFAULT_DQN_CHECKPOINT = REPO_ROOT / "baselines" / "checkpoints" / "dqn-duel-seed0"

AGENT_NAMES = ("random_legal", "random", "heuristic", "dqn", "dqn:<run dir>", "mcts-<n>[:k=v...]")


class Agent(NamedTuple):
    """A named policy.

    Attributes:
      name: the registry name it was made from (e.g. ``"mcts-128"``).
      policy: ``(key, state, timestep) -> int32[N]`` for one game; only the
        entries of the seats the agent controls are used.
      needs_obs: the policy reads ``timestep.obs``, so the env that steps the
        game must compute egocentric observations. Other agents only read the
        state and can share an observation-free env.
      description: one line for listings and logs.
    """

    name: str
    policy: Policy
    needs_obs: bool
    description: str


_CACHE: dict[tuple[str, GameConfig], Agent] = {}


def make_agent(name: str, config: GameConfig | None = None) -> Agent:
    """The agent called ``name`` (see the module docstring) for games with ``config``.

    Raises:
      ValueError: unknown name, bad ``mcts`` override, or a board the agent can't play.
      FileNotFoundError: ``dqn`` without ``baselines/dqn.py`` or its checkpoint.
    """
    config = config or GameConfig()
    name = name.strip()
    key = (name, config)
    if key not in _CACHE:
        _CACHE[key] = _build(name, config)
    return _CACHE[key]


@functools.lru_cache(maxsize=32)
def make_env(config: GameConfig, obs: bool) -> BattlesnakeEnv:
    """A cached env for ``config``: egocentric observations if ``obs``, none otherwise."""
    return BattlesnakeEnv(config, obs="egocentric" if obs else None)


def _build(name: str, config: GameConfig) -> Agent:
    env = make_env(config, False)
    if name == "random_legal":
        desc = "uniformly random among the moves that are not certainly fatal"
        return Agent(name, random_legal(env), False, desc)
    if name == "random":
        desc = "uniformly random over all four moves"
        return Agent(name, _random(env), False, desc)
    if name == "heuristic":
        from slinky.heuristic import heuristic

        desc = "hand-written snake: safety tiers, then a one-ply duel search"
        return Agent(name, heuristic(env), False, desc)
    if name == "dqn" or name.startswith("dqn:"):
        return _dqn_agent(name, config)
    if re.fullmatch(r"mcts-\d+(:.*)?", name):
        return _mcts_agent(name, config)
    raise ValueError(f"unknown agent {name!r}; known agents: {', '.join(AGENT_NAMES)}")


@functools.lru_cache(maxsize=16)
def _random(env: BattlesnakeEnv) -> Policy:
    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        return random_policy(key, state, env)

    return policy


# --- DQN -------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_dqn_module() -> ModuleType:
    """Import ``baselines/dqn.py`` from the repository (``baselines/`` is not a package)."""
    if "baselines_dqn" in sys.modules:  # already imported, e.g. by tests/test_dqn.py
        return sys.modules["baselines_dqn"]
    if not DQN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"the dqn agent needs baselines/dqn.py from a checkout of the slinky repository "
            f"(looked for {DQN_SCRIPT}); it is not part of the installed package"
        )
    spec = importlib.util.spec_from_file_location("baselines_dqn", DQN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return module


def _resolve_run_dir(text: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute() and not path.exists() and (REPO_ROOT / path).exists():
        path = REPO_ROOT / path
    if not (path / "params.npz").is_file() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"no DQN checkpoint in {path} (expected params.npz and config.json there)"
        )
    return path.resolve()


def _dqn_agent(name: str, config: GameConfig) -> Agent:
    _, sep, path = name.partition(":")
    if sep and not path:
        raise ValueError("write dqn:<run dir>, e.g. dqn:baselines/checkpoints/dqn-duel-seed0")
    run_dir = _resolve_run_dir(path) if sep else _resolve_run_dir(str(DEFAULT_DQN_CHECKPOINT))
    dqn = load_dqn_module()
    cfg = dqn.load_config(str(run_dir))
    trained = cfg.game
    if (config.width, config.height) != (trained.width, trained.height) or (
        config.ruleset.wrapped != trained.ruleset.wrapped
    ):
        raise ValueError(
            f"{name} was trained on {trained.width}x{trained.height} boards "
            f"({'wrapped' if trained.ruleset.wrapped else 'not wrapped'}) and can't play "
            f"{config.width}x{config.height} ({config.ruleset.value})"
        )
    params = dqn.load_params(str(run_dir), cfg)

    def q_fn(obs: jax.Array) -> jax.Array:
        return dqn.q_network(params, obs, cfg)

    desc = f"greedy self-play DQN ({run_dir.name})"
    return Agent(name, greedy_from_q(q_fn), True, desc)


# --- MCTS ------------------------------------------------------------------------


def _coerce(field: str, text: str, default: Any) -> Any:
    kind = type(default)
    try:
        if kind is bool:
            lowered = text.lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            raise ValueError(text)
        if kind in (int, float, str):
            return kind(text)
    except ValueError:
        raise ValueError(f"mcts option {field}={text!r}: expected a {kind.__name__}") from None
    raise ValueError(f"mcts option {field} ({kind.__name__}) can't be set from an agent name")


def parse_mcts_name(name: str) -> tuple[int, dict[str, Any]]:
    """``"mcts-128:exploration=1.0"`` -> ``(128, {"exploration": 1.0})``.

    Field names and types come from :class:`slinky.mcts.MCTSConfig` at runtime.
    """
    m = re.fullmatch(r"mcts-(\d+)((?::[^:]*)*)", name)
    if m is None:
        raise ValueError(f"bad mcts agent name {name!r}; write mcts-<n>[:field=value...]")
    num_simulations = int(m.group(1))
    mcts = importlib.import_module("slinky.mcts")
    fields = {f.name for f in dataclasses.fields(mcts.MCTSConfig)}
    defaults = mcts.MCTSConfig()
    overrides: dict[str, Any] = {}
    for item in filter(None, m.group(2).split(":")):
        field, sep, text = item.partition("=")
        field = field.strip()
        if not sep:
            raise ValueError(f"mcts option {item!r} in {name!r}: write field=value")
        if field == "num_simulations":
            raise ValueError("set num_simulations with the name itself: mcts-<n>")
        if field not in fields:
            known = ", ".join(sorted(fields - {"num_simulations"}))
            raise ValueError(f"unknown mcts option {field!r} in {name!r}; known options: {known}")
        overrides[field] = _coerce(field, text.strip(), getattr(defaults, field))
    return num_simulations, overrides


def _mcts_agent(name: str, config: GameConfig) -> Agent:
    num_simulations, overrides = parse_mcts_name(name)
    mcts = importlib.import_module("slinky.mcts")
    search = mcts.MCTSConfig(num_simulations=num_simulations, **overrides)
    extra = "".join(f", {k}={v}" for k, v in overrides.items())
    desc = f"simultaneous-move MCTS, {num_simulations} simulations{extra}"
    return Agent(name, mcts.mcts(make_env(config, False), search), False, desc)
