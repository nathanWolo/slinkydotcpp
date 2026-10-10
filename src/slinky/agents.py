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
  package: this agent needs a checkout of the repository. A relative run dir
  is looked up in the working directory, then in the repository.
* ``ppo`` or ``ppo:<run dir>``: the self-play PPO baseline (``baselines/ppo.py``),
  default checkpoint ``baselines/checkpoints/ppo-duel-seed0``, otherwise like
  ``dqn``. It plays in mode :data:`PPO_DEFAULT_MODE`; ``ppo-greedy`` (the legal
  argmax of the policy logits) and ``ppo-sample`` (a move sampled from the
  policy over the legal moves, with the match's random keys) choose the mode
  explicitly, also with ``:<run dir>``. The canonical name leaves the default
  mode out. The default is greedy: in the pilot runs it scored at least as well
  as sampled play against the heuristic and the DQN.
* ``rainbow`` or ``rainbow:<run dir>``: the self-play Rainbow DQN baseline
  (``baselines/rainbow.py``, legal argmax of the expected return, noise off),
  default checkpoint ``baselines/checkpoints/rainbow-duel-seed0``, otherwise
  like ``dqn``.
* ``mcts-<n>``: simultaneous-move MCTS (:func:`slinky.mcts.mcts`) with ``n``
  simulations per move and the default :class:`slinky.mcts.MCTSConfig`. Other
  fields are set with dash-separated shorthands (``mcts-256-rm-c0.5``; the list
  is in :data:`MCTS_SHORTHANDS`) or with ``:field=value`` overrides after them
  (``mcts-128:exploration=1.0:selection=rm``), in any order. Override values
  are converted to the type of the field's default (``true``/``false`` for
  booleans).

**Canonical names.** :func:`parse_agent` returns an :class:`AgentSpec` whose
``name`` is canonical: it lists only the settings that differ from the
current defaults, shorthands first (in the order of :data:`MCTS_SHORTHANDS`),
then ``:field=value`` for settings no shorthand can express. A canonical name
parses back to the same config, so two different configs never share a name
(``mcts-64-rollout`` and ``mcts-64:rollout_steps=10`` are both
``mcts-64-rollout10``). A field set twice is an error, and so is a setting
with no effect on the search, which would otherwise give one agent two names:
``gamma`` without ``rm``; ``c``, ``noise`` or ``tuned`` with ``rm``; ``c``
with ``tuned``; ``rollout_policy`` without rollout steps; ``depth`` above the
number of simulations (the search caps it there).

Agents are cached per ``(canonical name, config)``: calling :func:`make_agent`
twice returns the same policy object, which keeps ``play_match``'s jit cache
warm.
"""

from __future__ import annotations

import dataclasses
import decimal
import functools
import hashlib
import importlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Callable
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
PPO_SCRIPT = REPO_ROOT / "baselines" / "ppo.py"
DEFAULT_PPO_CHECKPOINT = REPO_ROOT / "baselines" / "checkpoints" / "ppo-duel-seed0"
RAINBOW_SCRIPT = REPO_ROOT / "baselines" / "rainbow.py"
DEFAULT_RAINBOW_CHECKPOINT = REPO_ROOT / "baselines" / "checkpoints" / "rainbow-duel-seed0"
# How ``ppo`` plays: "greedy" (legal argmax of the logits) or "sample" (from the policy).
PPO_MODES = ("greedy", "sample")
PPO_DEFAULT_MODE = "greedy"

AGENT_NAMES = (
    "random_legal", "random", "heuristic", "dqn", "dqn:<run dir>",
    "ppo[-greedy|-sample]", "ppo[-greedy|-sample]:<run dir>", "rainbow", "rainbow:<run dir>",
    "mcts-<n>[-shorthand...][:field=value...]",
)  # fmt: skip
SIMPLE_AGENTS = ("random_legal", "random", "heuristic")


class Agent(NamedTuple):
    """A named policy.

    Attributes:
      name: the canonical registry name it was made from (e.g. ``"mcts-128"``).
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


@dataclasses.dataclass(frozen=True)
class AgentSpec:
    """What an agent name means, before anything is built (parsing reads no files).

    Attributes:
      name: the canonical name (see the module docstring).
      kind: ``"random_legal"``, ``"random"``, ``"heuristic"``, ``"dqn"``, ``"ppo"``,
        ``"rainbow"`` or ``"mcts"``.
      mcts: the search settings of an ``mcts`` agent.
      checkpoint: the resolved run directory of a ``dqn``, ``ppo`` or ``rainbow`` agent.
      greedy: how a ``ppo`` agent plays: the legal argmax of its logits (True) or a
        move sampled from its policy (False). None for other agents.
    """

    name: str
    kind: str
    mcts: Any = None  # slinky.mcts.MCTSConfig (imported lazily)
    checkpoint: Path | None = None
    greedy: bool | None = None

    @property
    def sims(self) -> int:
        """Simulations per move (0 for agents without a search)."""
        return self.mcts.num_simulations if self.mcts is not None else 0


def parse_agent(name: str) -> AgentSpec:
    """The :class:`AgentSpec` of ``name``, with its canonical name.

    Raises:
      ValueError: unknown name, bad ``mcts`` setting, or a setting with no effect.
    """
    name = name.strip()
    if name in SIMPLE_AGENTS:
        return AgentSpec(name, name)
    if name == "dqn" or name.startswith("dqn:"):
        return _parse_dqn(name)
    if name.startswith("dqn@"):
        raise ValueError(f"{name!r}: write dqn:<run dir>")
    if name == "ppo" or name.startswith(("ppo:", "ppo-", "ppo@")):
        return _parse_ppo(name)
    if name == "rainbow" or name.startswith("rainbow:"):
        return _parse_rainbow(name)
    if name.startswith("rainbow@"):
        raise ValueError(f"{name!r}: write rainbow:<run dir>")
    if name.startswith("mcts-"):
        return _parse_mcts(name)
    raise ValueError(f"unknown agent {name!r}; known agents: {', '.join(AGENT_NAMES)}")


def canonical_name(name: str) -> str:
    """``parse_agent(name).name``."""
    return parse_agent(name).name


def make_agent(name: str | AgentSpec, config: GameConfig | None = None) -> Agent:
    """The agent called ``name`` (see the module docstring) for games with ``config``.

    Raises:
      ValueError: unknown name, bad ``mcts`` setting, or a game the agent can't play.
      FileNotFoundError: ``dqn``, ``ppo`` or ``rainbow`` without its script in
        ``baselines/`` or its checkpoint.
    """
    config = config or GameConfig()
    spec = parse_agent(name) if isinstance(name, str) else name
    key = (spec.name, config)
    if key not in _CACHE:
        check_game(spec, config)
        _CACHE[key] = _build(spec, config)
    return _CACHE[key]


_CACHE: dict[tuple[str, GameConfig], Agent] = {}


@functools.lru_cache(maxsize=32)
def make_env(config: GameConfig, obs: bool) -> BattlesnakeEnv:
    """A cached env for ``config``: egocentric observations if ``obs``, none otherwise."""
    return BattlesnakeEnv(config, obs="egocentric" if obs else None)


def check_game(spec: AgentSpec, config: GameConfig) -> None:
    """Raise ``ValueError`` if the agent can't play games with ``config``.

    Raises ``FileNotFoundError`` for a ``dqn``, ``ppo`` or ``rainbow`` agent without its
    checkpoint.
    """
    if spec.kind == "mcts":
        mcts = importlib.import_module("slinky.mcts")
        if config.num_snakes > mcts.MAX_SNAKES:
            raise ValueError(f"{spec.name}: MCTS supports at most {mcts.MAX_SNAKES} snakes")
    elif spec.kind in BASELINES:
        trained = _run_config(spec).game
        if (config.width, config.height) != (trained.width, trained.height) or (
            config.ruleset.wrapped != trained.ruleset.wrapped
        ):
            raise ValueError(
                f"{spec.name} was trained on {trained.width}x{trained.height} boards "
                f"({'wrapped' if trained.ruleset.wrapped else 'not wrapped'}) and can't play "
                f"{config.width}x{config.height} ({config.ruleset.value})"
            )


def agent_config(spec: AgentSpec) -> dict[str, Any]:
    """Everything that defines the agent's play, as plain JSON data.

    The full config, not only the non-default settings: every ``MCTSConfig``
    field, the heuristic weights, a DQN's, PPO's or Rainbow's run config
    (network included), how it picks moves and a hash of its parameters. Benchmarks
    hash it to tell results of different agents apart.
    """
    if spec.kind in ("random_legal", "random"):
        config: dict[str, Any] = {"type": spec.kind}
    elif spec.kind == "heuristic":
        heuristic = importlib.import_module("slinky.heuristic")
        config = {"type": "heuristic", "weights": heuristic.DEFAULT_WEIGHTS._asdict()}
    elif spec.kind in BASELINES:
        path = spec.checkpoint
        params = _checkpoint_files(spec)[0]
        config = {
            "type": spec.kind,
            "checkpoint": _display_path(path),
            "greedy": spec.greedy is not False,  # the DQN and Rainbow are always greedy
            "params_sha256": hashlib.sha256(params.read_bytes()).hexdigest()[:16],
            "network": dataclasses.asdict(_run_config(spec)),
        }
    else:
        fields = {f.name: getattr(spec.mcts, f.name) for f in dataclasses.fields(spec.mcts)}
        fields = {k: v._asdict() if hasattr(v, "_asdict") else v for k, v in fields.items()}
        config = {"type": "mcts", **fields}
    return json.loads(json.dumps(config, sort_keys=True))  # as it reads back from JSON


def _build(spec: AgentSpec, config: GameConfig) -> Agent:
    env = make_env(config, False)
    if spec.kind == "random_legal":
        desc = "uniformly random among the moves that are not certainly fatal"
        return Agent(spec.name, random_legal(env), False, desc)
    if spec.kind == "random":
        desc = "uniformly random over all four moves"
        return Agent(spec.name, _random(env), False, desc)
    if spec.kind == "heuristic":
        from slinky.heuristic import heuristic

        desc = "hand-written snake: safety tiers, then a one-ply duel search"
        return Agent(spec.name, heuristic(env), False, desc)
    if spec.kind == "dqn":
        return _dqn_agent(spec)
    if spec.kind == "ppo":
        return _ppo_agent(spec)
    if spec.kind == "rainbow":
        return _rainbow_agent(spec)
    return _mcts_agent(spec, env)


@functools.lru_cache(maxsize=16)
def _random(env: BattlesnakeEnv) -> Policy:
    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        return random_policy(key, state, env)

    return policy


# --- Trained baselines (DQN, PPO, Rainbow) -----------------------------------------

# kind -> (script in baselines/, module name it is imported as, default checkpoint)
BASELINES = {
    "dqn": (DQN_SCRIPT, "baselines_dqn", DEFAULT_DQN_CHECKPOINT),
    "ppo": (PPO_SCRIPT, "baselines_ppo", DEFAULT_PPO_CHECKPOINT),
    "rainbow": (RAINBOW_SCRIPT, "baselines_rainbow", DEFAULT_RAINBOW_CHECKPOINT),
}
# For messages ("no DQN checkpoint in ...").
BASELINE_LABELS = {"dqn": "DQN", "ppo": "PPO", "rainbow": "Rainbow"}


@functools.cache
def load_baseline_module(kind: str) -> ModuleType:
    """Import ``baselines/<kind>.py`` from the repository (``baselines/`` is not a package)."""
    script, module_name, _ = BASELINES[kind]
    if module_name in sys.modules:  # already imported, e.g. by tests/test_dqn.py
        return sys.modules[module_name]
    if not script.is_file():
        raise FileNotFoundError(
            f"the {kind} agent needs baselines/{script.name} from a checkout of the slinky "
            f"repository (looked for {script}); it is not part of the installed package"
        )
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return module


def load_dqn_module() -> ModuleType:
    """Import ``baselines/dqn.py`` from the repository."""
    return load_baseline_module("dqn")


def load_ppo_module() -> ModuleType:
    """Import ``baselines/ppo.py`` from the repository."""
    return load_baseline_module("ppo")


def load_rainbow_module() -> ModuleType:
    """Import ``baselines/rainbow.py`` from the repository."""
    return load_baseline_module("rainbow")


def _run_dir(kind: str, text: str | None) -> Path:
    """The resolved run directory named by ``text`` (None: the default checkpoint)."""
    if text is None:
        return BASELINES[kind][2].resolve()
    if not text.strip():
        raise ValueError(
            f"write {kind}:<run dir>, e.g. {kind}:baselines/checkpoints/{kind}-duel-seed0"
        )
    path = Path(text.strip()).expanduser()
    if not path.is_absolute() and not path.exists() and (REPO_ROOT / path).exists():
        path = REPO_ROOT / path
    return path.resolve()


def _with_run_dir(base: str, kind: str, path: Path) -> str:
    """``base`` for the default checkpoint, else ``base:<run dir>``."""
    return base if path == BASELINES[kind][2].resolve() else f"{base}:{_display_path(path)}"


def _parse_dqn(name: str) -> AgentSpec:
    _, sep, text = name.partition(":")
    path = _run_dir("dqn", text if sep else None)
    return AgentSpec(_with_run_dir("dqn", "dqn", path), "dqn", checkpoint=path)


def _parse_rainbow(name: str) -> AgentSpec:
    _, sep, text = name.partition(":")
    path = _run_dir("rainbow", text if sep else None)
    return AgentSpec(_with_run_dir("rainbow", "rainbow", path), "rainbow", checkpoint=path)


def _parse_ppo(name: str) -> AgentSpec:
    head, sep, text = name.partition(":")
    if head.startswith("ppo@"):
        raise ValueError(f"{name!r}: write ppo:<run dir>")
    mode = PPO_DEFAULT_MODE
    if head != "ppo":
        mode = head.removeprefix("ppo-")
        if mode not in PPO_MODES:
            raise ValueError(
                f"{name!r}: unknown ppo mode {mode!r}; write ppo[-greedy|-sample][:<run dir>]"
            )
    path = _run_dir("ppo", text if sep else None)
    base = "ppo" if mode == PPO_DEFAULT_MODE else f"ppo-{mode}"
    return AgentSpec(
        _with_run_dir(base, "ppo", path), "ppo", checkpoint=path, greedy=mode == "greedy"
    )


def _display_path(path: Path) -> str:
    """Relative to the repository if inside it, else absolute."""
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def _checkpoint_files(spec: AgentSpec) -> tuple[Path, Path]:
    path = spec.checkpoint
    params, config = path / "params.npz", path / "config.json"
    if not params.is_file() or not config.is_file():
        raise FileNotFoundError(
            f"no {BASELINE_LABELS[spec.kind]} checkpoint in {path} (expected params.npz and "
            "config.json there)"
        )
    return params, config


def _run_config(spec: AgentSpec) -> Any:
    """The training config saved with a ``dqn``, ``ppo`` or ``rainbow`` agent's checkpoint.

    The three save the same files. A config that names its algorithm (``rainbow.py``
    writes ``"algorithm"``) must match the agent's kind, so ``dqn:<Rainbow run>`` is
    refused here rather than failing on a parameter shape later.
    """
    config = _checkpoint_files(spec)[1]
    try:
        algorithm = json.loads(config.read_text()).get("algorithm")
    except (OSError, ValueError, AttributeError):
        algorithm = None  # let load_config report the problem
    if algorithm in BASELINES and algorithm != spec.kind:
        raise ValueError(
            f"{spec.name}: {_display_path(spec.checkpoint)} is a {algorithm} run; "
            f"write {algorithm}:<run dir>"
        )
    return load_baseline_module(spec.kind).load_config(str(spec.checkpoint))


def _dqn_agent(spec: AgentSpec) -> Agent:
    dqn = load_dqn_module()
    cfg = _run_config(spec)
    params = dqn.load_params(str(spec.checkpoint), cfg)

    def q_fn(obs: jax.Array) -> jax.Array:
        return dqn.q_network(params, obs, cfg)

    desc = f"greedy self-play DQN ({spec.checkpoint.name})"
    return Agent(spec.name, greedy_from_q(q_fn), True, desc)


def _rainbow_agent(spec: AgentSpec) -> Agent:
    rainbow = load_rainbow_module()
    cfg = _run_config(spec)
    params = rainbow.load_params(str(spec.checkpoint), cfg)
    desc = f"greedy self-play Rainbow DQN, noise off ({spec.checkpoint.name})"
    return Agent(spec.name, greedy_from_q(rainbow.q_function(params, cfg)), True, desc)


def _ppo_agent(spec: AgentSpec) -> Agent:
    ppo = load_ppo_module()
    cfg = _run_config(spec)
    params = ppo.load_params(str(spec.checkpoint), cfg)
    how = "greedy" if spec.greedy else "sampled"
    desc = f"{how} self-play PPO ({spec.checkpoint.name})"
    return Agent(spec.name, ppo.make_policy(params, cfg, bool(spec.greedy)), True, desc)


# --- MCTS ------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Shorthand:
    """A dash-separated shorthand of ``mcts-<n>`` that sets config fields.

    ``number`` is None for a flag and ``int`` or ``float`` for ``name<value>``;
    ``default`` is the value when the number is omitted (None: required).
    ``apply(value)`` gives the fields it sets. ``show(config, defaults)`` is the
    value the canonical name shows: None or False when the config does not use
    the shorthand (default settings, or a value it can't spell, such as a
    negative number, which then goes in a ``:field=value`` override).
    """

    name: str
    doc: str
    fields: tuple[str, ...]
    apply: Callable[[Any], dict[str, Any]]
    show: Callable[[Any, Any], Any]
    number: type | None = None
    default: float | None = None

    def token(self, config: Any, defaults: Any) -> str | None:
        v = self.show(config, defaults)
        if v is None or v is False:
            return None
        return self.name if self.number is None else f"{self.name}{format_number(v)}"


def _changed(field: str) -> Callable[[Any, Any], Any]:
    """Show the field's value if it differs from the default and is a plain number."""

    def show(c: Any, d: Any) -> Any:
        v = getattr(c, field)
        return v if v != getattr(d, field) and _spellable(v) else None

    return show


def _rollout(policy: str) -> Callable[[Any, Any], Any]:
    return lambda c, d: c.rollout_steps if c.rollout_steps and c.rollout_policy == policy else None


def _contempt(c: Any, d: Any) -> Any:
    v = 0.0 - c.draw_value
    return v if c.draw_value != d.draw_value and _spellable(v) else None


# fmt: off
MCTS_SHORTHANDS: tuple[Shorthand, ...] = (
    Shorthand("rm", "regret-matching selection instead of DUCT (selection='rm')", ("selection",),
              lambda v: {"selection": "rm"}, lambda c, d: c.selection == "rm"),
    Shorthand("tuned", "UCB1-Tuned variance bound in DUCT (ucb1_tuned=True)", ("ucb1_tuned",),
              lambda v: {"ucb1_tuned": True}, lambda c, d: c.ucb1_tuned is True),
    Shorthand("c", "c<x>: DUCT's exploration constant, e.g. c0.5 (exploration=x)",
              ("exploration",), lambda v: {"exploration": v}, _changed("exploration"), float),
    Shorthand("noheur", "no heuristic leaf evaluation: living snakes are worth 0 (leaf='none')",
              ("leaf",), lambda v: {"leaf": "none"}, lambda c, d: c.leaf == "none"),
    Shorthand("rollout", "rollout[<k>]: k random-policy rollout turns before the leaf (default 10)",
              ("rollout_steps", "rollout_policy"),
              lambda v: {"rollout_steps": v, "rollout_policy": "random"}, _rollout("random"),
              int, 10),
    Shorthand("hrollout", "hrollout[<k>]: like rollout, with the heuristic policy (default 10)",
              ("rollout_steps", "rollout_policy"),
              lambda v: {"rollout_steps": v, "rollout_policy": "heuristic"},
              _rollout("heuristic"), int, 10),
    Shorthand("spawn", "sample food spawns inside the tree (spawn_food=True)", ("spawn_food",),
              lambda v: {"spawn_food": True}, lambda c, d: c.spawn_food is True),
    Shorthand("sample", "sample the final move from the visit counts (final='sample')",
              ("final",), lambda v: {"final": "sample"}, lambda c, d: c.final == "sample"),
    Shorthand("contempt", "contempt<x>: a mutual elimination is worth -x inside the search "
              "(draw_value=-x; not the heuristic's Weights.contempt)", ("draw_value",),
              lambda v: {"draw_value": 0.0 - v}, _contempt, float),
    Shorthand("noise", "noise<x>: scale of DUCT's tie-breaking noise (tie_noise=x)",
              ("tie_noise",), lambda v: {"tie_noise": v}, _changed("tie_noise"), float),
    Shorthand("gamma", "gamma<x>: regret matching's exploration mix (rm_gamma=x); needs rm",
              ("rm_gamma",), lambda v: {"rm_gamma": v}, _changed("rm_gamma"), float),
    Shorthand("depth", "depth<k>: deepest descent from the root (max_depth=k), at most <n>",
              ("max_depth",), lambda v: {"max_depth": v}, _changed("max_depth"), int),
)
# fmt: on
_SHORTHAND_BY_NAME = {s.name: s for s in MCTS_SHORTHANDS}


def mcts_shorthand_help() -> str:
    """One line per shorthand, for ``--help`` texts."""
    width = max(len(s.name) for s in MCTS_SHORTHANDS) + 3
    return "\n".join(f"  {s.name:<{width}}{s.doc}" for s in MCTS_SHORTHANDS)


def format_number(v: float | int) -> str:
    """The shortest text that reads back as ``v``, with no exponent (``1e-05``: ``0.00001``)."""
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    text = format(decimal.Decimal(repr(float(v) + 0.0)), "f")  # + 0.0: -0.0 reads as 0
    return text.rstrip("0").rstrip(".") if "." in text else text


def _spellable(v: Any) -> bool:
    """A number a dash shorthand can spell (digits and a point)."""
    return isinstance(v, (int, float)) and math.isfinite(v) and v >= 0


def _parse_number(kind: type, text: str, what: str) -> Any:
    try:
        value = kind(text)
    except ValueError:
        article = "an" if kind is int else "a"
        raise ValueError(f"{what}: expected {article} {kind.__name__}") from None
    if kind is float and not math.isfinite(value):
        raise ValueError(f"{what}: expected a finite number")
    return value


def _coerce(field: str, text: str, default: Any) -> Any:
    kind = type(default)
    what = f"mcts option {field}={text!r}"
    if kind is bool:
        lowered = text.lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{what}: expected true or false")
    if kind in (int, float):
        return _parse_number(kind, text, what)
    if kind is str:
        return text
    raise ValueError(f"mcts option {field} ({kind.__name__}) can't be set from an agent name")


def _parse_mcts(name: str) -> AgentSpec:
    mcts = importlib.import_module("slinky.mcts")
    defaults = mcts.MCTSConfig()
    head, *overrides = name.split(":")
    parts = head.split("-")
    if len(parts) < 2 or not parts[1].isdigit() or int(parts[1]) < 1:
        raise ValueError(
            f"bad mcts agent name {name!r}; write mcts-<n>[-shorthand...][:field=value...] "
            "with n >= 1"
        )
    fields: dict[str, Any] = {"num_simulations": int(parts[1])}
    given: dict[str, str] = {}  # field -> the item that set it

    def put(item: str, values: dict[str, Any]) -> None:
        for field, value in values.items():
            if field in given:
                raise ValueError(f"{name!r}: {given[field]!r} and {item!r} both set {field}")
            given[field] = item
            fields[field] = value

    for token in parts[2:]:
        match = re.fullmatch(r"([a-z]+)([0-9.]*)", token)
        shorthand = _SHORTHAND_BY_NAME.get(match[1]) if match else None
        if shorthand is None:
            valid = ", ".join(s.name for s in MCTS_SHORTHANDS)
            raise ValueError(f"{name!r}: unknown shorthand {token!r} (valid: {valid})")
        digits = match[2]
        if shorthand.number is None:
            if digits:
                raise ValueError(f"{name!r}: shorthand {shorthand.name!r} takes no number")
            value = None
        elif digits:
            value = _parse_number(shorthand.number, digits, f"{name!r}: {token!r}")
            if shorthand.number is int and value < 1:
                raise ValueError(f"{name!r}: {token!r} must be >= 1")
        elif shorthand.default is not None:
            value = shorthand.number(shorthand.default)
        else:
            raise ValueError(
                f"{name!r}: shorthand {shorthand.name!r} needs a number, e.g. {shorthand.name}1"
            )
        put(token, shorthand.apply(value))

    known = {f.name for f in dataclasses.fields(mcts.MCTSConfig)} - {"num_simulations"}
    for item in overrides:
        field, sep, text = item.partition("=")
        field = field.strip()
        if not sep:
            raise ValueError(f"mcts option {item!r} in {name!r}: write field=value")
        if field == "num_simulations":
            raise ValueError("set num_simulations with the name itself: mcts-<n>")
        if field not in known:
            raise ValueError(
                f"unknown mcts option {field!r} in {name!r}; known options: "
                f"{', '.join(sorted(known))}"
            )
        put(item, {field: _coerce(field, text.strip(), getattr(defaults, field))})

    try:
        config = mcts.MCTSConfig(**fields)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name!r}: {e}") from None
    if inert := _inert_settings(config, defaults):
        raise ValueError(f"{name!r}: {'; '.join(inert)}")
    return AgentSpec(mcts_name(config), "mcts", mcts=config)


def _inert_settings(c: Any, d: Any) -> list[str]:
    """Settings that differ from the defaults but do not change the search (see mcts.search)."""
    out = []

    def changed(field: str) -> bool:
        return getattr(c, field) != getattr(d, field)

    if c.selection == "rm":
        duct_only = (("c", "exploration"), ("tuned", "ucb1_tuned"), ("noise", "tie_noise"))
        out += [
            f"{what} ({field}) has no effect with rm" for what, field in duct_only if changed(field)
        ]
    elif changed("rm_gamma"):
        out.append("gamma (rm_gamma) has no effect without rm")
    if c.selection != "rm" and c.ucb1_tuned and changed("exploration"):
        out.append("c (exploration) has no effect with tuned")
    if c.rollout_steps == 0 and changed("rollout_policy"):
        out.append("rollout_policy has no effect without rollout steps")
    sims = c.num_simulations
    if changed("max_depth") and (
        c.max_depth > sims or (c.max_depth == sims and d.max_depth >= sims)
    ):
        out.append(
            f"depth{c.max_depth} has no effect: the search caps the depth at the number of "
            f"simulations ({sims}, default depth {d.max_depth})"
        )
    return out


def mcts_name(config: Any) -> str:
    """The canonical name of an ``MCTSConfig`` (see the module docstring)."""
    mcts = importlib.import_module("slinky.mcts")
    defaults = mcts.MCTSConfig()
    tokens, covered = [], {"num_simulations"}
    for shorthand in MCTS_SHORTHANDS:
        if token := shorthand.token(config, defaults):
            tokens.append(token)
            covered.update(shorthand.fields)
    overrides = []
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if f.name not in covered and value != getattr(defaults, f.name):
            if isinstance(value, bool):
                text = str(value).lower()
            elif isinstance(value, (int, float)):
                text = format_number(value)
            elif isinstance(value, str) and not re.search(r"[:=\s]", value):
                text = value
            else:
                raise ValueError(f"mcts field {f.name}={value!r} has no agent-name spelling")
            overrides.append(f":{f.name}={text}")
    return "-".join(["mcts", str(config.num_simulations), *tokens]) + "".join(overrides)


def _mcts_agent(spec: AgentSpec, env: BattlesnakeEnv) -> Agent:
    mcts = importlib.import_module("slinky.mcts")
    defaults = mcts.MCTSConfig()
    extra = "".join(
        f", {f.name}={getattr(spec.mcts, f.name)}"
        for f in dataclasses.fields(spec.mcts)
        if f.name != "num_simulations" and getattr(spec.mcts, f.name) != getattr(defaults, f.name)
    )
    desc = f"simultaneous-move MCTS, {spec.sims} simulations{extra}"
    return Agent(spec.name, mcts.mcts(env, spec.mcts), False, desc)
