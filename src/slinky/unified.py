"""Shared machinery of the multi-mode ("unified") baselines: one network for every game mode.

``baselines/{reinforce,dqn,rainbow,az}_modes.py`` each train one network that plays all
four modes of :mod:`slinky.modes` and is told which one it is in, following the design of
``baselines/ppo_modes.py`` (whose docstring argues it; ``ppo_modes.py`` itself is
self-contained and does not use this module). What they share is here:

* **Groups.** Modes cannot share a ``vmap`` (the snake count changes the ``[N, ...]``
  shapes, royale carries its own ``map_state``, the ruleset and map are static), so each
  mode is a :class:`Group` of games with its own config and envs. A turn builds every
  group's inputs, makes one network call on their concatenation (:func:`join`) and splits
  the outputs back (:func:`split`).
* **Inputs.** The egocentric board with the food channel zeroed in constrictor (where food
  is inert) and the mode features of :func:`slinky.modes.mode_features`
  (:func:`policy_inputs`), exactly as ``ppo_modes.py`` computes them.
* **The torso.** ``ppo.py``'s 3x3 convolutions and dense hidden layer plus the
  zero-initialised feature matrix ``feat/w`` added to the hidden pre-activation
  (:func:`init_torso`, :func:`torso`); each algorithm puts its own heads on top. The
  sizes are settings, so one algorithm can be trained at several. ``compute_dtype``
  runs the convolutions and the hidden matmul in bfloat16 (mixed precision: parameters,
  optimizer, biases, activations and heads stay float32).
* **Learning rate.** Constant, then optionally a linear *cooldown* to 0 over the last
  env steps of the run (:func:`lr_factor`): the campaign trains at a constant rate until
  the strength plateaus, then cools down from there (``--cooldown``).
* **The run loop** (:func:`train`): jitted chunks, ``metrics.jsonl``, snapshots
  ``params-<env steps per mode>.npz`` every ``snapshot_every`` env steps per mode, full
  checkpoints (``runner.npz``) for ``--resume``, and a ``STOP`` file in the run directory
  that makes the run checkpoint and exit after the current chunk (the plateau watcher
  writes it). Each algorithm module provides ``ALGORITHM``, a config class with the
  :class:`ModesSettings` fields, ``init_runner``, ``make_train_chunk``, ``chunk_record``,
  ``format_record`` and ``make_policy``; its ``RunnerState`` has ``params`` and
  ``env_steps``.

An *env step per mode* is the mean over the modes of their games advanced one turn (the
modes take equal steps, except REINFORCE's whole games).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import glob
import json
import math
import os
import time
import types
import typing
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from slinky import modes
from slinky.env import BattlesnakeEnv, placement_reward, win_loss_reward
from slinky.evaluate import TableResult, play_table
from slinky.observations import obs_shape
from slinky.types import Cause, GameConfig, State

Params = dict[str, dict[str, jax.Array]]

# Elimination causes reported in the logs, in display order.
DEATH_CAUSES = {
    "wall": Cause.OUT_OF_BOUNDS,
    "self": Cause.SELF_COLLISION,
    "body": Cause.COLLISION,
    "h2h": Cause.HEAD_COLLISION,
    "starve": Cause.OUT_OF_HEALTH,
    "hazard": Cause.HAZARD,
}
NUM_CAUSES = len(Cause)
REWARDS = {"placement": placement_reward, "win_loss": win_loss_reward}
FEATURES = ("all", "no_mode", "none")
COMPUTE_DTYPES = ("float32", "bfloat16")
# Logit of an illegal move: exp() underflows to exactly 0, but unlike -inf it keeps
# p * log p and every gradient finite.
MASKED_LOGIT = -1e9
# The run directory file that stops a run after its current chunk (see train()).
STOP_FILE = "STOP"
# Feature columns that features="no_mode" keeps (it zeroes the mode one-hot and the
# royale shrink phase, which tell the modes apart).
_NO_MODE_KEEP = np.array(
    [name in ("turn", "opponents_alive") for name in modes.FEATURE_NAMES], np.float32
)
# The network sizes the campaign trains every algorithm at: conv channels and hidden units
# (strides (2, 2, 1) for all). "M" is ppo_modes.py's default network.
SIZES = {
    "S": ((8, 16, 16), 128),
    "M": ((16, 32, 32), 256),
    "L": ((32, 64, 64), 512),
}


@dataclasses.dataclass(frozen=True)
class ModesSettings:
    """The fields every multi-mode config has; each algorithm's config adds its own.

    Budgets are *env steps per mode*. Subclasses call ``super().__post_init__()``.
    """

    # Game modes (slinky.modes.MODE_NAMES): one group of games each. The first is the
    # headline of the top-level train metrics.
    game_modes: tuple[str, ...] = modes.MODE_NAMES
    mode_snakes: tuple[str, ...] = ()  # "constrictor=2": overrides modes.DEFAULT_SNAKES
    max_turns: int = 1000  # truncation, in training and evaluation
    reward: str = "placement"  # "placement" (slinky.env.placement_reward) or "win_loss"
    features: str = "all"  # "all", "no_mode" (no one-hot or shrink phase) or "none" (no layer)
    hide_constrictor_food: bool = True  # zero the food channel in constrictor games
    # Network torso: 3x3 convs, a dense hidden layer (plus feat/w); see SIZES.
    conv_channels: tuple[int, ...] = (16, 32, 32)
    conv_strides: tuple[int, ...] = (2, 2, 1)
    hidden: int = 256
    compute_dtype: str = "float32"  # of the convs and the hidden matmul: float32 or bfloat16
    # Optimization: a constant learning rate, then a linear cooldown to 0 at
    # total_env_steps from cooldown_from env steps per mode (-1: no cooldown).
    lr: float = 1e-3
    cooldown_from: int = -1
    # Run length, snapshots and checkpoints (env steps per mode).
    seed: int = 0
    total_env_steps: int = 33_554_432  # the campaign's cap: the plateau watcher stops it sooner
    snapshot_every: int = 1_048_576  # params-<steps>.npz, for the plateau checks
    checkpoint_every: int = 262_144  # runner.npz, for --resume
    # Evaluation at the end of the run (the campaign evaluates snapshots elsewhere).
    eval_games: int = 256  # per mode and opponent; 0 = none
    eval_opponents: tuple[str, ...] = ("random_legal", "heuristic")  # slinky.agents
    run_dir: str | None = None  # default: runs/<algorithm>-<timestamp>

    def __post_init__(self) -> None:
        if not self.game_modes:
            raise ValueError("game_modes must name at least one mode")
        if bad := [m for m in self.game_modes if m not in modes.MODE_NAMES]:
            raise ValueError(f"unknown game_modes {bad}; expected some of {modes.MODE_NAMES}")
        if len(set(self.game_modes)) != len(self.game_modes):
            raise ValueError(f"game_modes has duplicates: {self.game_modes}")
        parse_mode_snakes(self.mode_snakes, self.game_modes)
        if self.reward not in REWARDS:
            raise ValueError(f"unknown reward {self.reward!r}; expected one of {tuple(REWARDS)}")
        if self.features not in FEATURES:
            raise ValueError(f"unknown features {self.features!r}; expected one of {FEATURES}")
        if self.compute_dtype not in COMPUTE_DTYPES:
            raise ValueError(f"compute_dtype must be one of {COMPUTE_DTYPES}")
        if len(self.conv_channels) != len(self.conv_strides):
            raise ValueError("conv_channels and conv_strides must have the same length")
        positive = ("max_turns", "hidden", "total_env_steps", "snapshot_every", "checkpoint_every")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not -1 <= self.cooldown_from < self.total_env_steps:
            raise ValueError("cooldown_from must be -1 (none) or in [0, total_env_steps)")

    @property
    def snakes(self) -> dict[str, int]:
        """Snakes per game mode: ``modes.DEFAULT_SNAKES`` with the ``mode_snakes`` overrides."""
        overrides = parse_mode_snakes(self.mode_snakes, self.game_modes)
        return {m: overrides.get(m, modes.DEFAULT_SNAKES[m]) for m in self.game_modes}

    @property
    def games(self) -> dict[str, GameConfig]:
        """The game of each mode, in ``game_modes`` order (``max_turns`` included)."""
        return {m: modes.mode_config(m, n, self.max_turns) for m, n in self.snakes.items()}

    @property
    def game(self) -> GameConfig:
        """The first (headline) mode's game; its board size is every mode's."""
        return self.games[self.game_modes[0]]


def parse_mode_snakes(items: tuple[str, ...], game_modes: tuple[str, ...]) -> dict[str, int]:
    """``{"constrictor": 2}`` from ``("constrictor=2",)``, checked against the modes."""
    out: dict[str, int] = {}
    for item in items:
        name, sep, count = (t.strip() for t in item.partition("="))
        if not sep or not count.lstrip("-").isdigit():
            raise ValueError(f"mode_snakes items are <mode>=<snakes>, got {item!r}")
        if name not in game_modes:
            raise ValueError(f"mode_snakes names {name!r}, which is not in game_modes")
        if name in out:
            raise ValueError(f"mode_snakes sets {name!r} twice")
        modes.mode_config(name, int(count))  # raises for a count outside the mode
        out[name] = int(count)
    return out


# --- Groups ----------------------------------------------------------------------------


class Group(NamedTuple):
    """The games of one mode: its config, envs and place in the network's batch."""

    name: str  # the game mode
    mode_id: int  # its index in slinky.modes.MODE_NAMES
    game: GameConfig
    env: BattlesnakeEnv  # egocentric observations
    sim: BattlesnakeEnv  # no observations; steps with the run's reward

    @property
    def num_snakes(self) -> int:
        return self.game.num_snakes


def make_groups(cfg: ModesSettings) -> tuple[Group, ...]:
    """One :class:`Group` per entry of ``cfg.game_modes``, in that order."""
    reward_fn = REWARDS[cfg.reward]
    return tuple(
        Group(
            name=name,
            mode_id=modes.MODE_NAMES.index(name),
            game=game,
            env=BattlesnakeEnv(game, obs="egocentric"),
            sim=BattlesnakeEnv(game, obs=None, reward_fn=reward_fn),
        )
        for name, game in cfg.games.items()
    )


def group_rows(groups: Sequence[Group], num_envs: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """``(first row, rows)`` of each group in a concatenated batch of ``num_envs[g]`` games each."""
    out, start = [], 0
    for g, e in zip(groups, num_envs, strict=True):
        out.append((start, e * g.num_snakes))
        start += e * g.num_snakes
    return tuple(out)


def join(xs: Sequence[jax.Array]) -> jax.Array:
    """Per-group ``[E_g, N_g, ...]`` arrays as one ``[sum E_g * N_g, ...]`` batch."""
    if len(xs) == 1:
        return xs[0].reshape(-1, *xs[0].shape[2:])
    return jnp.concatenate([x.reshape(-1, *x.shape[2:]) for x in xs])


def split(x: jax.Array, groups: Sequence[Group], num_envs: Sequence[int]) -> tuple[jax.Array, ...]:
    """The inverse of :func:`join`: one ``[E_g, N_g, ...]`` array per group."""
    return tuple(
        x[start : start + rows].reshape(e, g.num_snakes, *x.shape[1:])
        for g, e, (start, rows) in zip(groups, num_envs, group_rows(groups, num_envs), strict=True)
    )


def group_key(key: jax.Array, group: Group) -> jax.Array:
    """The key of one group's randomness (one stream per mode)."""
    return jax.random.fold_in(key, group.mode_id)


def tree_where(cond: jax.Array, a: Any, b: Any) -> Any:
    """``where(cond, a, b)`` leaf by leaf (``cond`` broadcasts from the leading axes)."""

    def pick(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.where(cond.reshape(cond.shape + (1,) * (x.ndim - cond.ndim)), x, y)

    return jax.tree.map(pick, a, b)


# --- Inputs ------------------------------------------------------------------------------


def policy_features(
    state: State, config: GameConfig, cfg: ModesSettings, game_mode: str | None = None
) -> jax.Array:
    """The feature input ``f32[N, NUM_FEATURES]`` of one (unbatched) game.

    :func:`slinky.modes.mode_features` with the mode the network is told (``game_mode``,
    by default the config's own), with the one-hot and the shrink phase zeroed for
    ``features="no_mode"``, all zeros for ``"none"`` (whose network has no feature layer).
    """
    if cfg.features == "none":
        return jnp.zeros((config.num_snakes, modes.NUM_FEATURES), jnp.float32)
    feats = modes.mode_features(state, config, game_mode)
    if cfg.features == "no_mode":
        feats = feats * _NO_MODE_KEEP
    return feats


def policy_inputs(
    obs: jax.Array,
    state: State,
    config: GameConfig,
    cfg: ModesSettings,
    game_mode: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """The network inputs of one game: ``(boards, features)``, ``[N, ...]``.

    ``obs`` is the egocentric observation of ``state`` (``env.observe``). The boards are
    ``obs`` with the food channel zeroed if the *real* game is constrictor (and
    ``hide_constrictor_food``), whatever mode the network is told. Training and play both
    go through here.
    """
    boards = modes.hide_food(obs, config) if cfg.hide_constrictor_food else obs
    return boards, policy_features(state, config, cfg, game_mode)


def group_inputs(
    cfg: ModesSettings, group: Group, states: State
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(boards, features, mask)`` of a batch of ``[E]`` games: ``[E, N, ...]`` each."""
    obs = jax.vmap(group.env.observe)(states)
    boards, feats = jax.vmap(lambda o, s: policy_inputs(o, s, group.game, cfg, group.name))(
        obs, states
    )
    return boards, feats, jax.vmap(group.env.action_mask)(states)


def told_mode(cfg: ModesSettings, config: GameConfig, game_mode: str | None) -> str | None:
    """The mode a network is told in games with ``config`` (None without features)."""
    if cfg.features == "none":
        return None  # no features: the mode never reaches the network
    mode = modes.mode_of(config) if game_mode is None else game_mode
    if mode not in modes.MODE_NAMES:
        raise ValueError(f"unknown game mode {mode!r}; expected one of {modes.MODE_NAMES}")
    return mode


# --- The torso ---------------------------------------------------------------------------


def _conv_im2col(x: jax.Array, w: jax.Array, stride: int) -> jax.Array:
    """3x3 "SAME" convolution as 9 strided slices of the padded input and one matmul."""
    n, h, wd, c = x.shape
    ho, wo = -(-h // stride), -(-wd // stride)
    ph, pw = max((ho - 1) * stride + 3 - h, 0), max((wo - 1) * stride + 3 - wd, 0)
    x = jnp.pad(x, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
    patches = [
        jax.lax.slice(
            x,
            (0, dy, dx, 0),
            (n, dy + (ho - 1) * stride + 1, dx + (wo - 1) * stride + 1, c),
            (1, stride, stride, 1),
        )
        for dy in range(3)
        for dx in range(3)
    ]
    return jnp.concatenate(patches, axis=-1) @ w.reshape(9 * c, -1)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def conv3x3(x: jax.Array, w: jax.Array, stride: int) -> jax.Array:
    """3x3 "SAME" convolution, ``x: [B, H, W, Cin]``, ``w: [3, 3, Cin, Cout]``.

    The forward pass is XLA's convolution; the backward pass differentiates the
    equivalent im2col form (:func:`_conv_im2col`), which XLA:CPU runs much faster inside
    a ``lax.scan`` (as in ``baselines/ppo.py``).
    """
    return jax.lax.conv_general_dilated(
        x, w, (stride, stride), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )


def _conv3x3_fwd(x: jax.Array, w: jax.Array, stride: int):
    return conv3x3(x, w, stride), (x, w)


def _conv3x3_bwd(stride: int, residuals: tuple[jax.Array, jax.Array], g: jax.Array):
    x, w = residuals
    _, vjp = jax.vjp(lambda x, w: _conv_im2col(x, w, stride), x, w)
    return vjp(g)


conv3x3.defvjp(_conv3x3_fwd, _conv3x3_bwd)


def dense(key: jax.Array, shape: tuple[int, ...], fan_in: int, gain: float = 2.0) -> dict:
    """A layer ``{"w": normal * sqrt(gain / fan_in), "b": zeros}`` (He-normal for gain 2)."""
    w = jax.random.normal(key, shape) * math.sqrt(gain / fan_in)
    return {"w": w, "b": jnp.zeros(shape[-1])}


def init_convs(key: jax.Array, cfg: ModesSettings) -> tuple[Params, int]:
    """``(params, flat size)``: He-normal convolutions ``conv<i>`` and the size of their
    flattened output, the hidden layer's input."""
    rows, cols, cin = obs_shape(cfg.game)
    keys = iter(jax.random.split(key, len(cfg.conv_channels)))
    params: Params = {}
    for i, (cout, stride) in enumerate(zip(cfg.conv_channels, cfg.conv_strides, strict=True)):
        params[f"conv{i}"] = dense(next(keys), (3, 3, cin, cout), 9 * cin)
        rows, cols, cin = -(-rows // stride), -(-cols // stride), cout  # "SAME" padding
    return params, rows * cols * cin


def init_feat(cfg: ModesSettings) -> Params:
    """``{"feat": zeros}`` (nothing for ``features="none"``): the network starts as if it had
    no features, and learns what they are worth."""
    if cfg.features == "none":
        return {}
    return {"feat": {"w": jnp.zeros((modes.NUM_FEATURES, cfg.hidden))}}


def init_torso(key: jax.Array, cfg: ModesSettings) -> Params:
    """The convolutions ``conv<i>``, the hidden layer and ``feat`` (all zeros; absent for
    ``features="none"``): ``ppo_modes.py``'s torso."""
    k_conv, k_hidden = jax.random.split(key)
    params, flat = init_convs(k_conv, cfg)
    params["hidden"] = dense(k_hidden, (flat, cfg.hidden), flat)
    return {**params, **init_feat(cfg)}


def conv_features(params: Params, boards: jax.Array, cfg: ModesSettings) -> jax.Array:
    """The flattened convolution output ``f32[B, flat]`` of boards ``[..., rows, cols, C]``.

    The convolutions take ``cfg.compute_dtype`` inputs (the float32 parameters cast at
    each call; their gradients come back float32); biases and activations stay float32,
    so the bias gradients, sums over every board cell, are summed in float32. With
    float32 the casts are no-ops.
    """
    dtype = jnp.dtype(cfg.compute_dtype)
    x = boards.reshape(-1, *boards.shape[-3:])
    for i, stride in enumerate(cfg.conv_strides):
        p = params[f"conv{i}"]
        y = conv3x3(x.astype(dtype), p["w"].astype(dtype), stride).astype(jnp.float32)
        x = jax.nn.relu(y + p["b"])
    return x.reshape(x.shape[0], -1)


def matmul(x: jax.Array, w: jax.Array, cfg: ModesSettings) -> jax.Array:
    """``x @ w`` with ``cfg.compute_dtype`` inputs and a float32 result."""
    dtype = jnp.dtype(cfg.compute_dtype)
    return (x.astype(dtype) @ w.astype(dtype)).astype(jnp.float32)


def feature_term(params: Params, feats: jax.Array) -> jax.Array | float:
    """``feats @ feat/w``, added to the hidden pre-activation (0 without a ``feat`` layer)."""
    if "feat" not in params:
        return 0.0
    return feats.reshape(-1, feats.shape[-1]) @ params["feat"]["w"]


def torso(params: Params, boards: jax.Array, feats: jax.Array, cfg: ModesSettings) -> jax.Array:
    """The hidden activations ``f32[B, hidden]`` of boards ``[..., rows, cols, C]``, flattened.

    ``feats`` is ``f32[..., NUM_FEATURES]`` (ignored without a ``feat`` layer). The
    convolutions and the hidden matmul run in ``cfg.compute_dtype``
    (:func:`conv_features`).
    """
    x = conv_features(params, boards, cfg)
    h = params["hidden"]
    return jax.nn.relu(matmul(x, h["w"], cfg) + h["b"] + feature_term(params, feats))


def count_params(params: Any) -> int:
    return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params))


# --- The masked policy -------------------------------------------------------------------


def masked_log_probs(logits: jax.Array, mask: jax.Array) -> jax.Array:
    """Log-probabilities ``f32[..., 4]`` of the policy restricted to the legal moves."""
    return jax.nn.log_softmax(jnp.where(mask, logits, MASKED_LOGIT), axis=-1)


def masked_entropy(log_probs: jax.Array, mask: jax.Array) -> jax.Array:
    """Entropy ``f32[...]`` of the masked policy (illegal moves contribute nothing)."""
    return -jnp.sum(jnp.where(mask, jnp.exp(log_probs) * log_probs, 0.0), axis=-1)


def sample_actions(key: jax.Array, logits: jax.Array, mask: jax.Array) -> jax.Array:
    """One legal move per row of ``logits``/``mask`` (``[..., 4]``), sampled from the policy."""
    return jax.random.categorical(key, jnp.where(mask, logits, MASKED_LOGIT)).astype(jnp.int32)


def greedy_actions(scores: jax.Array, mask: jax.Array) -> jax.Array:
    """The legal move with the highest score (logit or Q) per row."""
    return jnp.argmax(jnp.where(mask, scores, -jnp.inf), axis=-1).astype(jnp.int32)


# --- Learning rate --------------------------------------------------------------------


def lr_factor(env_steps: jax.Array, cfg: ModesSettings) -> jax.Array:
    """The learning rate's multiplier at ``env_steps`` (per mode): 1, or the cooldown's
    linear decay from 1 at ``cooldown_from`` to 0 at ``total_env_steps``."""
    if cfg.cooldown_from < 0:
        return jnp.ones((), jnp.float32)
    span = cfg.total_env_steps - cfg.cooldown_from
    left = (cfg.total_env_steps - env_steps).astype(jnp.float32) / span
    return jnp.clip(left, 0.0, 1.0)


def scale_updates(updates: Any, lr: jax.Array) -> Any:
    """``-lr * updates``: the step of an optimizer chain that ends before its learning rate."""
    return jax.tree.map(lambda u: -lr * u, updates)


# --- Game statistics -----------------------------------------------------------------------


def game_metrics(states: State, new: State, done: jax.Array, truncated: jax.Array) -> dict:
    """Per-turn counts of a batch of games ``[E]``: games ended, their turns, draws,
    truncations and the deaths by cause."""
    died = states.alive & ~new.alive  # [E, N]
    causes = jnp.arange(NUM_CAUSES) == new.elim_cause[..., None]  # [E, N, causes]
    return {
        "games": jnp.sum(done),
        "game_turns": jnp.sum(jnp.where(done, new.turn, 0)),
        "draws": jnp.sum(done & ~truncated & ~jnp.any(new.alive, axis=-1)),
        "truncated": jnp.sum(truncated),
        "deaths": jnp.sum(causes & died[..., None], axis=(0, 1)),
    }


def mode_game_stats(m: dict[str, Any], g: int) -> dict[str, Any]:
    """The game statistics of group ``g`` from summed :func:`game_metrics`, for the record."""
    games = int(m["games"][g])
    return {
        "games": games,
        "mean_game_len": float(m["game_turns"][g]) / games if games else None,
        "draws": int(m["draws"][g]),
        "truncated": int(m["truncated"][g]),
        "deaths": {k: int(m["deaths"][g][c]) for k, c in DEATH_CAUSES.items()},
    }


def _pct(x: float, total: float) -> str:
    return f"{100 * x / total:.0f}%" if total else "-"


def format_games(r: dict[str, Any]) -> str:
    """``games ... len ... draw ... | deaths ...`` of one mode's record block."""
    deaths = r["deaths"]
    n_dead = sum(deaths.values())
    shown = [k for k in DEATH_CAUSES if k != "hazard" or deaths[k]]
    length = f"{r['mean_game_len']:.0f}" if r["mean_game_len"] is not None else "-"
    return (
        f"games {r['games']:,} len {length} draw {_pct(r['draws'], r['games'])} "
        f"trunc {r['truncated']} | deaths "
        + " ".join(f"{k} {_pct(deaths[k], n_dead)}" for k in shown)
    )


def finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


# --- Evaluation ------------------------------------------------------------------------------


def evaluate_params(
    module: types.ModuleType,
    params: Params,
    cfg: ModesSettings,
    group: Group,
    key: jax.Array,
    num_games: int,
    opponent: str = "heuristic",
    greedy: bool = True,
) -> TableResult:
    """The network (``module.make_policy``) in one seat, ``opponent`` in all others.

    ``play_table`` in the group's mode with the line-up ``(0, 1, ..., 1)``, seats rotating
    from game to game and games stopped once decided. The parameters are baked into the
    compiled table, so every call compiles once more.
    """
    from slinky.agents import make_agent  # the registry imports the baselines in turn

    params = jax.tree.map(jnp.copy, params)  # the training loop donates the runner's arrays
    policy = module.make_policy(params, cfg, group.game, group.name, greedy)
    other = make_agent(opponent, group.game).policy
    lineup = (0,) + (1,) * (group.num_snakes - 1)
    return play_table(group.env, (policy, other), lineup, key, num_games, max_turns=cfg.max_turns)


def eval_record(
    r: TableResult, group: Group, opponent: str, mode: str, env_steps: int, seconds: float
) -> dict[str, Any]:
    """An eval record: the network's (policy 0's) share, placement and outcome counts."""
    return {
        "type": "eval",
        "env_steps": env_steps,
        "game_mode": group.name,
        "num_snakes": group.num_snakes,
        "opponent": opponent,
        "mode": mode,
        "score": r.share[0],
        "score_ci95": r.share_ci95[0],
        "score_wilson": list(r.share_wilson[0]),
        "placement": r.placement[0],
        "placement_ci95": r.placement_ci95[0],
        "wins": r.wins[0],
        "draws": r.ties[0],  # a share strictly between 0 and 1
        "losses": r.losses[0],
        "truncated": r.truncated,
        "num_games": r.num_games,
        "mean_turns": r.mean_turns,  # the turn the games were decided
        "deaths": r.deaths[0],
        "seconds": seconds,
    }


def format_eval(r: TableResult) -> str:
    n = r.num_games
    return (
        f"W {_pct(r.wins[0], n)} T {_pct(r.ties[0], n)} L {_pct(r.losses[0], n)} | "
        f"share {r.share[0]:.3f} +/- {r.share_ci95[0]:.3f} | "
        f"place {r.placement[0]:.2f} +/- {r.placement_ci95[0]:.2f} | "
        f"decided at {r.mean_turns:.0f} | truncated {r.truncated}"
    )


def run_evals(
    module: types.ModuleType,
    params: Params,
    cfg: ModesSettings,
    groups: tuple[Group, ...],
    key: jax.Array,
    num_games: int,
    opponents: tuple[str, ...],
    env_steps: int,
    write: Callable[[dict[str, Any]], None] = lambda record: None,
) -> list[dict[str, Any]]:
    """Every (game mode, opponent) table, greedy; prints and returns eval records."""
    records = []
    for group in groups:
        k = jax.random.fold_in(key, group.mode_id)
        for opponent in opponents:
            t = time.perf_counter()
            r = evaluate_params(module, params, cfg, group, k, num_games, opponent)
            seconds = time.perf_counter() - t
            record = eval_record(r, group, opponent, "greedy", env_steps, seconds)
            write(record)
            records.append(record)
            where = f"@ {env_steps:,} " if env_steps >= 0 else ""
            print(
                f"eval {where}{group.name:>11} vs {opponent}: {format_eval(r)} ({seconds:.0f}s)",
                flush=True,
            )
    return records


# --- Checkpoints --------------------------------------------------------------------------


def save_params(run_dir: str, params: Params, name: str = "params.npz") -> str:
    """Write ``name`` (flattened ``layer/name`` keys) atomically; returns its path."""
    flat = {
        "/".join(k.key for k in path): np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
    }
    path = os.path.join(run_dir, name)
    tmp = os.path.join(run_dir, name.removesuffix(".npz") + ".tmp.npz")
    np.savez(tmp, **flat)
    os.replace(tmp, path)
    return path


def snapshot_name(env_steps: int) -> str:
    """The file of the snapshot taken at ``env_steps`` (per mode), as ``ppo_modes.py`` names it."""
    return f"params-{env_steps}.npz"


def load_params_like(template: Params, path: str) -> Params:
    """The parameters in the ``.npz`` at ``path``, structured and checked like ``template``."""
    paths, treedef = jax.tree_util.tree_flatten_with_path(template)
    with np.load(path) as f:
        leaves = []
        for p, ref in paths:
            leaf = f["/".join(k.key for k in p)]
            if leaf.shape != ref.shape:
                raise ValueError(f"{p}: checkpoint shape {leaf.shape} != {ref.shape}")
            leaves.append(jnp.asarray(leaf))
    return jax.tree.unflatten(treedef, leaves)


def save_config(run_dir: str, algorithm: str, cfg: Any) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"algorithm": algorithm, **dataclasses.asdict(cfg)}, f, indent=2)


def load_config(run_dir: str, algorithm: str, cls: type) -> Any:
    """The ``cls`` config saved in ``run_dir``, which must be an ``algorithm`` run."""
    with open(os.path.join(run_dir, "config.json")) as f:
        d = json.load(f)
    if d.get("algorithm") != algorithm:
        raise ValueError(
            f"{run_dir} is not a {algorithm} run (config.json has no algorithm={algorithm})"
        )
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in fields}
    return cls(**kwargs)


def save_runner(run_dir: str, runner: Any, extra: dict[str, int]) -> None:
    """Write the whole training state to ``runner.npz`` atomically (keys stored as data)."""
    leaves = jax.tree.leaves(jax.tree.map(_key_data, runner, is_leaf=_is_key))
    arrays = {f"leaf{i}": np.asarray(x) for i, x in enumerate(leaves)}
    tmp = os.path.join(run_dir, "runner.tmp.npz")
    np.savez(tmp, **{f"extra_{k}": v for k, v in extra.items()}, **arrays)
    os.replace(tmp, os.path.join(run_dir, "runner.npz"))


def load_runner(run_dir: str, template: Any) -> tuple[Any, dict[str, int]]:
    """``(runner, extra)`` from ``runner.npz``, structured like ``template``."""
    data_template = jax.tree.map(_key_data, template, is_leaf=_is_key)
    refs, treedef = jax.tree.flatten(data_template)
    with np.load(os.path.join(run_dir, "runner.npz")) as f:
        extra = {k.removeprefix("extra_"): int(f[k]) for k in f.files if k.startswith("extra_")}
        if len(f.files) - len(extra) != len(refs):
            raise ValueError("runner.npz does not match the config (number of arrays)")
        leaves = [f[f"leaf{i}"] for i in range(len(refs))]
    for leaf, ref in zip(leaves, refs, strict=True):
        if leaf.shape != ref.shape or leaf.dtype != ref.dtype:
            raise ValueError(f"runner.npz does not match the config: {leaf.shape} vs {ref.shape}")
    restored = jax.tree.unflatten(treedef, [jnp.asarray(x) for x in leaves])
    runner = jax.tree.map(
        lambda t, x: jax.random.wrap_key_data(x) if _is_key(t) else x,
        template, restored, is_leaf=_is_key,
    )  # fmt: skip
    return runner, extra


def _is_key(x: Any) -> bool:
    return isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jax.dtypes.prng_key)


def _key_data(x: Any) -> Any:
    return jax.random.key_data(x) if _is_key(x) else x


def truncate_metrics(path: str, env_steps: int) -> None:
    """Drop records logged after the checkpoint we resume from (they will be redone)."""
    if not os.path.exists(path):
        return
    records = []
    with open(path) as f:
        for line in f:
            try:
                if json.loads(line)["env_steps"] <= env_steps:
                    records.append(line)
            except json.JSONDecodeError:  # a line cut short by a hard crash
                pass
    with open(path, "w") as f:
        f.writelines(records)


# --- The run loop ------------------------------------------------------------------------


def train(module: types.ModuleType, cfg: Any, resume: bool = False) -> tuple[str, Params]:
    """Run ``module``'s training with ``cfg``; returns ``(run_dir, final params)``.

    The loop runs ``module.make_train_chunk(cfg, groups)`` (jitted, the runner donated)
    until ``total_env_steps`` per mode, writing each chunk's ``module.chunk_record`` to
    ``metrics.jsonl``. It saves a snapshot every ``snapshot_every`` env steps per mode and
    at the end, ``runner.npz`` every ``checkpoint_every`` and at the end, and stops early,
    after a checkpoint, when the run directory has a ``STOP`` file (renamed ``STOPPED``).
    With ``resume`` it continues from ``runner.npz``; a new run in an old directory first
    deletes that run's files.
    """
    name = module.ALGORITHM
    run_dir = cfg.run_dir or os.path.join("runs", time.strftime(f"{name}-%Y%m%d-%H%M%S"))
    cfg = dataclasses.replace(cfg, run_dir=run_dir)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    runner_path = os.path.join(run_dir, "runner.npz")

    groups = make_groups(cfg)
    if cfg.eval_games > 0:  # fail now, not at the end, if an opponent can't be built
        from slinky.agents import make_agent

        for group in groups:
            for opponent in cfg.eval_opponents:
                make_agent(opponent, group.game)
    resuming = resume and os.path.exists(runner_path)
    k_eval, k_init = jax.random.split(jax.random.key(cfg.seed))
    runner = jax.jit(module.init_runner, static_argnums=(0, 1))(cfg, groups, k_init)
    os.makedirs(run_dir, exist_ok=True)
    if not resuming:  # a new run in an old directory: --resume must not find the old run
        names = ["metrics.jsonl", "runner.npz", "params.npz", STOP_FILE, "STOPPED"]
        names += [os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "params-*.npz"))]
        for n in names:
            if os.path.exists(path := os.path.join(run_dir, n)):
                os.remove(path)
    save_config(run_dir, name, cfg)
    extra = {"agent_samples": 0}
    if resuming:
        runner, extra = load_runner(run_dir, runner)
        extra.pop("env_steps", None)
        truncate_metrics(metrics_path, int(runner.env_steps))
        print(f"resuming from env step {int(runner.env_steps):,} per mode")

    n_params = count_params(runner.params)
    print(f"run dir: {run_dir}")
    print(f"config: {json.dumps(dataclasses.asdict(cfg))}")
    print(
        f"{name}: {n_params:,} parameters | modes "
        f"{', '.join(f'{g.name} ({g.num_snakes})' for g in groups)} | "
        f"{cfg.total_env_steps:,} env steps per mode"
    )
    t0 = time.perf_counter()
    train_chunk = (
        jax.jit(module.make_train_chunk(cfg, groups), donate_argnums=0).lower(runner).compile()
    )
    print(f"compiled train_chunk in {time.perf_counter() - t0:.1f}s", flush=True)

    start = time.perf_counter()
    stop_path = os.path.join(run_dir, STOP_FILE)
    with open(metrics_path, "a") as log:

        def write(record: dict[str, Any]) -> None:
            log.write(json.dumps(record) + "\n")
            log.flush()

        steps = int(runner.env_steps)
        while steps < cfg.total_env_steps:
            t = time.perf_counter()
            runner, m = train_chunk(runner)
            m = jax.device_get(m)
            seconds = time.perf_counter() - t
            before, steps = steps, int(runner.env_steps)
            record = module.chunk_record(m, seconds, steps - before, cfg)
            extra["agent_samples"] += record.get("agent_samples", 0)
            record["agent_samples_total"] = extra["agent_samples"]
            record["elapsed"] = time.perf_counter() - start
            write(record)
            print(module.format_record(record, cfg.total_env_steps), flush=True)

            last = steps >= cfg.total_env_steps
            stop = os.path.exists(stop_path)
            if last or stop or steps // cfg.snapshot_every > before // cfg.snapshot_every:
                save_params(run_dir, runner.params, snapshot_name(steps))
                save_params(run_dir, runner.params)
            if last or stop or steps // cfg.checkpoint_every > before // cfg.checkpoint_every:
                save_runner(run_dir, runner, {**extra, "env_steps": steps})
            if stop:
                os.replace(stop_path, os.path.join(run_dir, "STOPPED"))
                print(f"stopped at {steps:,} env steps per mode ({STOP_FILE} file)", flush=True)
                return run_dir, runner.params
        if cfg.eval_games > 0:
            run_evals(
                module, runner.params, cfg, groups, jax.random.fold_in(k_eval, steps),
                cfg.eval_games, cfg.eval_opponents, steps, write,
            )  # fmt: skip
    print(f"done in {time.perf_counter() - start:.0f}s; checkpoint in {run_dir}")
    return run_dir, runner.params


def cooldown_config(cfg: Any, env_steps: int, cooldown: int) -> Any:
    """``cfg`` for a cooldown of ``cooldown`` env steps per mode from ``env_steps``."""
    return dataclasses.replace(cfg, cooldown_from=env_steps, total_env_steps=env_steps + cooldown)


# --- Command line --------------------------------------------------------------------------


def _parse_bool(text: str) -> bool:
    if text.lower() in ("1", "true", "yes", "on"):
        return True
    if text.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def _parse_tuple(kind: type) -> Callable[[str], tuple[Any, ...]]:
    def parse(text: str) -> tuple[Any, ...]:
        return tuple(kind(t.strip()) for t in text.split(",") if t.strip())

    return parse


def _field_parser(annotation: Any) -> Callable[[str], Any]:
    """The parser of a config field, from its type annotation."""
    if annotation is bool:
        return _parse_bool
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is tuple:
        return _parse_tuple(args[0])
    if origin in (typing.Union, types.UnionType):  # X | None
        (kind,) = (a for a in args if a is not type(None))
        return kind
    return annotation


def main(module: types.ModuleType, argv: list[str] | None = None) -> None:
    """The command line of a multi-mode trainer: train, ``--resume``, ``--cooldown`` or
    ``--eval-only``; every config field is a flag, and ``--size S|M|L`` sets the network."""
    cls = module.Config
    p = argparse.ArgumentParser(description=module.__doc__.split("\n\n")[0])
    p.add_argument(
        "--eval-only", metavar="RUN_DIR",
        help="load RUN_DIR's checkpoint and play --eval-games games in each of its game modes "
        "against each of --eval-opponents instead of training",
    )  # fmt: skip
    p.add_argument(
        "--resume", metavar="RUN_DIR",
        help="continue an interrupted or stopped run from its last full checkpoint, with its "
        "saved config (other flags are ignored)",
    )  # fmt: skip
    p.add_argument(
        "--cooldown", type=int, metavar="STEPS",
        help="with --resume: continue for STEPS more env steps per mode, the learning rate "
        "decaying linearly to 0 (the end of a campaign run)",
    )  # fmt: skip
    p.add_argument("--size", choices=tuple(SIZES), help="the network: sets conv-channels, hidden")
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        default = f.default
        shown = ",".join(map(str, default)) if isinstance(default, tuple) else default
        p.add_argument(
            "--" + f.name.replace("_", "-"),
            type=_field_parser(hints[f.name]),
            default=default,
            help=f"(default: {shown})",
        )
    args = p.parse_args(argv)
    kwargs = {f.name: getattr(args, f.name) for f in dataclasses.fields(cls)}
    if args.size:
        kwargs["conv_channels"], kwargs["hidden"] = SIZES[args.size]
    if args.eval_only:
        cfg = module.load_config(args.eval_only)
        params = module.load_params(args.eval_only, cfg)
        records = run_evals(
            module, params, cfg, make_groups(cfg), jax.random.key(args.seed), args.eval_games,
            args.eval_opponents, -1,
        )  # fmt: skip
        for record in records:
            print(json.dumps({k: v for k, v in record.items() if k != "env_steps"}))
        return
    if args.resume:
        cfg = dataclasses.replace(module.load_config(args.resume), run_dir=args.resume)
        if args.cooldown:
            steps = runner_env_steps(args.resume)
            cfg = cooldown_config(cfg, steps, args.cooldown)
            print(f"cooldown: {args.cooldown:,} env steps per mode from {steps:,}")
        train(module, cfg, resume=True)
        return
    if args.cooldown:
        p.error("--cooldown needs --resume")
    train(module, cls(**kwargs))


def runner_env_steps(run_dir: str) -> int:
    """The env steps per mode of ``run_dir``'s ``runner.npz`` (stored beside the arrays)."""
    with np.load(os.path.join(run_dir, "runner.npz")) as f:
        return int(f["extra_env_steps"])
