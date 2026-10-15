"""AlphaZero for the four leaderboard modes: one network told its mode, placement values.

``baselines/az.py`` (self-play with decoupled-PUCT search guided by the network's priors
and ``tanh`` values, root Dirichlet noise, visit-share policy targets and game-outcome
value targets, compact replay with outcomes that arrive later, board-symmetry
augmentation, AdamW; the same hyperparameters) trained on every mode of
:mod:`slinky.modes` at once with the multi-mode design of ``baselines/ppo_modes.py``
(shared code in :mod:`slinky.unified`): one group of ``num_envs`` games per mode, the
mode features and hidden constrictor food, and every mode weighing the same in the loss.
Changes from ``az.py``:

* **Placement values.** In a 4-snake game the search and the value targets use
  :func:`slinky.mcts.placement_values` (``MCTSConfig(value="placement")``): every snake
  outlasted is worth ``+1/(N-1)``, every snake that outlasts it ``-1/(N-1)``, so places
  1 to 4 are worth +1, +1/3, -1/3 and -1, and a dead snake's value is exact inside the
  tree while the others play on. ``draw_value`` (-0.5, contempt) scales the tie of the
  snakes eliminated together on the final turn. In the duel it is ``az.py``'s
  win/loss value.
* **One search per mode.** Each turn searches every group's games, one group after
  another (a search is a ``vmap`` over games of one config), so a simulation's network
  call covers ``num_envs * N`` observations of one mode.
* **Replay per mode** (``buffer_capacity`` positions each), and every gradient step
  samples ``batch_size`` positions of each mode; one forward pass on all of them. The
  loss is the weighted mean over the modes (``mode_weights``) of each mode's
  cross-entropy plus ``value_coef`` times its value MSE.
* **Augmentation** by a random board symmetry, except in royale, whose hazards grow
  from one edge of the board (its map state is not symmetric).
* **The learning rate** is constant (``az.py``'s 1e-3) with the campaign's optional
  cooldown (:func:`slinky.unified.lr_factor`) instead of the decay to a tenth; no
  teacher phase or heuristic blend.

Played as an agent the network plays raw (``az_modes``, the legal argmax of its policy;
``az_modes-sample`` samples from it) or with ``:simulations=<n>`` of search
(:func:`agent_search`).

Commands::

    python baselines/az_modes.py --size M --run-dir runs/az-modes-M
    python baselines/az_modes.py --resume runs/az-modes-M --cooldown 2097152
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import os
import sys
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from slinky import rules, unified
from slinky.env import BattlesnakeEnv
from slinky.evaluate import Policy
from slinky.mcts import (
    Evaluator,
    MCTSConfig,
    SearchOutput,
    placement_values,
    root_noise_draws,
    search,
)
from slinky.types import ACTION_DELTAS, NUM_ACTIONS, GameConfig, State, TimeStep
from slinky.unified import Group, Params

ALGORITHM = "az_modes"


@dataclasses.dataclass(frozen=True)
class Config(unified.ModesSettings):
    """All hyperparameters (and :class:`slinky.unified.ModesSettings`'). Every field can be
    overridden on the command line. Steps are env steps per mode. One iteration plays
    ``num_steps`` turns in each of ``num_envs`` games of every mode, then takes
    ``updates_per_iteration`` gradient steps."""

    lr: float = 1e-3
    total_env_steps: int = 16_777_216  # the campaign's cap for AlphaZero
    mode_weights: tuple[float, ...] = ()  # loss weight of each game_modes entry; () = equal
    # Self-play, per mode.
    num_envs: int = 256
    num_steps: int = 32
    num_simulations: int = 64
    temperature_turns: int = 16  # sample moves from the root visits before this turn
    root_noise_frac: float = 0.25
    root_dirichlet_alpha: float = 0.5
    # Search constants (self-play and the agent).
    c_puct: float = 0.5
    fpu_reduction: float = 0.2
    draw_value: float = -0.5  # the final tie's contempt, in the search and the targets
    # Replay, per mode: positions (one State and every snake's targets each).
    buffer_capacity: int = 262_144  # rounded down to whole iterations
    learning_starts: int = 32_768  # positions in each buffer before the first update
    batch_size: int = 512  # positions of each mode per gradient step
    replay_ratio: float = 4.0  # expected times each stored position is sampled
    augment: bool = True  # a random board symmetry per sampled position (not in royale)
    # Optimization.
    weight_decay: float = 1e-4  # AdamW (decoupled, scaled by the learning rate)
    max_grad_norm: float = 1.0
    adam_eps: float = 1e-8
    value_coef: float = 1.0
    log_every: int = 32_768  # env steps per mode per jitted chunk, whole iterations

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode_weights and len(self.mode_weights) != len(self.game_modes):
            raise ValueError("mode_weights needs one weight per game mode")
        if any(w < 0 for w in self.mode_weights) or (
            self.mode_weights and sum(self.mode_weights) <= 0
        ):
            raise ValueError("mode_weights must be >= 0 with a positive sum")
        for name in ("num_envs", "num_steps", "num_simulations", "batch_size", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.temperature_turns < 0:
            raise ValueError("temperature_turns must be >= 0")
        if self.buffer_capacity < self.steps_per_iteration:
            raise ValueError("buffer_capacity must hold at least one iteration")
        if self.learning_starts > self.capacity:
            raise ValueError("learning_starts must not exceed the (rounded) buffer_capacity")
        if not 0.0 < self.replay_ratio < math.inf:
            raise ValueError("replay_ratio must be finite and > 0")
        selfplay_search(self)  # checks the search settings

    @property
    def steps_per_iteration(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def capacity(self) -> int:
        """Replay positions per mode: ``buffer_capacity`` rounded down to whole iterations."""
        return self.buffer_capacity // self.steps_per_iteration * self.steps_per_iteration

    @property
    def updates_per_iteration(self) -> int:
        return max(round(self.replay_ratio * self.steps_per_iteration / self.batch_size), 1)

    @property
    def chunk_iters(self) -> int:
        return max(self.log_every // self.steps_per_iteration, 1)

    @property
    def loss_weights(self) -> tuple[float, ...]:
        w = self.mode_weights or (1.0,) * len(self.game_modes)
        return tuple(x / sum(w) for x in w)


def selfplay_search(cfg: Config) -> MCTSConfig:
    """Self-play's search: decoupled PUCT, network priors and leaves, root noise, placement."""
    return MCTSConfig(
        num_simulations=cfg.num_simulations,
        selection="puct",
        prior="network",
        leaf="network",
        c_puct=cfg.c_puct,
        fpu_reduction=cfg.fpu_reduction,
        root_noise_frac=cfg.root_noise_frac,
        root_dirichlet_alpha=cfg.root_dirichlet_alpha,
        draw_value=cfg.draw_value,
        value="placement",
    )


def agent_search(cfg: Config, simulations: int) -> MCTSConfig:
    """The trained network's search as a player: no noise, the most visited move."""
    return MCTSConfig(
        num_simulations=simulations,
        selection="puct",
        prior="network",
        leaf="network",
        c_puct=cfg.c_puct,
        fpu_reduction=cfg.fpu_reduction,
        draw_value=cfg.draw_value,
        value="placement",
    )


# --- Network: the torso, a policy head and a tanh value head --------------------------


def init_network(key: jax.Array, cfg: Config) -> Params:
    """The torso, a near-zero policy head and a value head that starts near 0 (``az.py``'s)."""
    k_torso, k_pi, k_v = jax.random.split(key, 3)
    params = unified.init_torso(k_torso, cfg)
    params["policy"] = unified.dense(k_pi, (cfg.hidden, NUM_ACTIONS), cfg.hidden, gain=1e-4)
    params["value"] = unified.dense(k_v, (cfg.hidden, 1), cfg.hidden, gain=1e-2)
    return params


def network(
    params: Params, boards: jax.Array, feats: jax.Array, cfg: Config
) -> tuple[jax.Array, jax.Array]:
    """``(logits f32[..., 4], value f32[...] in (-1, 1))`` for boards ``[..., rows, cols, C]``."""
    lead = boards.shape[:-3]
    x = unified.torso(params, boards, feats, cfg)
    logits = x @ params["policy"]["w"] + params["policy"]["b"]
    value = jnp.tanh(x @ params["value"]["w"] + params["value"]["b"])
    return logits.reshape(*lead, NUM_ACTIONS), value.reshape(lead)


def make_evaluator(
    params: Params, cfg: Config, group: Group, game_mode: str | None = None
) -> Evaluator:
    """``slinky.mcts`` evaluator ``state -> (value f32[N], logits f32[N, 4])`` of one game,
    every snake's inputs built from the state (told ``game_mode``, default the group's)."""
    mode = group.name if game_mode is None else game_mode

    def evaluator(state: State) -> tuple[jax.Array, jax.Array]:
        boards, feats = unified.policy_inputs(
            group.env.observe(state), state, group.game, cfg, mode
        )
        logits, value = network(params, boards, feats, cfg)
        return value, logits

    return evaluator


def make_policy(
    params: Params,
    cfg: Config,
    game_config: GameConfig,
    game_mode: str | None = None,
    greedy: bool = True,
    simulations: int = 0,
) -> Policy:
    """An ``evaluate.Policy``: the raw network (``simulations == 0``: the legal argmax of its
    logits, or a sample with ``greedy=False``) or the move of its search (:func:`agent_search`).

    ``game_mode`` is the mode the network is told (default: the config's own). The
    parameters are baked into whatever traces the policy, so a new call compiles again.
    """
    mode = unified.told_mode(cfg, game_config, game_mode)
    group = unified.Group(
        name=mode or "none",
        mode_id=-1,
        game=game_config,
        env=BattlesnakeEnv(game_config, obs="egocentric"),
        sim=BattlesnakeEnv(game_config, obs=None),
    )
    if simulations == 0:

        def raw(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
            boards, feats = unified.policy_inputs(ts.obs, state, game_config, cfg, mode)
            logits = network(params, boards, feats, cfg)[0]
            if greedy:
                return unified.greedy_actions(logits, ts.action_mask)
            return unified.sample_actions(key, logits, ts.action_mask)

        return raw
    mcfg = agent_search(cfg, simulations)
    evaluator = make_evaluator(params, cfg, group, mode)

    def searched(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        return search(key, state, group.sim, mcfg, evaluator).action

    return searched


# --- Board symmetries (az.py's) ---------------------------------------------------------

SYMMETRIES = tuple(itertools.product((False, True), repeat=3))


def action_permutation(sym: int) -> np.ndarray:
    """int32[4]: ``perm[a]`` is the move that ``a`` becomes on the transformed board."""
    transpose, flip_x, flip_y = SYMMETRIES[sym]
    out = []
    for dx, dy in ACTION_DELTAS:
        if transpose:
            dx, dy = dy, dx
        out.append(ACTION_DELTAS.index((-dx if flip_x else dx, -dy if flip_y else dy)))
    return np.array(out, np.int32)


class SymmetryTables(NamedTuple):
    flags: np.ndarray  # bool[S, 3]: (transpose, flip_x, flip_y)
    perm: np.ndarray  # int32[S, 4]
    inverse: np.ndarray  # int32[S, 4]
    source: np.ndarray  # int32[S, H * W]: flat source cell of every flat target cell


@functools.cache
def symmetry_tables(width: int, height: int) -> SymmetryTables:
    """All 8 symmetries on a square board; only the 4 flips otherwise."""
    count = len(SYMMETRIES) if width == height else len(SYMMETRIES) // 2
    ys, xs = np.mgrid[:height, :width]
    source = np.zeros((count, height * width), np.int32)
    for s in range(count):
        transpose, flip_x, flip_y = SYMMETRIES[s]
        x, y = (ys, xs) if transpose else (xs, ys)
        x = width - 1 - x if flip_x else x
        y = height - 1 - y if flip_y else y
        source[s, (y * width + x).ravel()] = (ys * width + xs).ravel()
    perm = np.stack([action_permutation(s) for s in range(count)])
    return SymmetryTables(
        np.array(SYMMETRIES[:count], bool), perm, np.argsort(perm, axis=1).astype(np.int32), source
    )


def transform_state(state: State, sym: jax.Array | int, game: GameConfig) -> State:
    """The game ``state`` mirrored/rotated by symmetry ``sym`` (maps without state only)."""
    if jax.tree.leaves(state.map_state):
        raise ValueError("board symmetries are implemented for maps without state only")
    tab = symmetry_tables(game.width, game.height)
    sym = jnp.asarray(sym, jnp.int32)
    src = jnp.asarray(tab.source)[sym]

    def grid(a: jax.Array) -> jax.Array:
        flat = a.reshape(*a.shape[:-2], game.height * game.width)
        return jnp.take(flat, src, axis=-1).reshape(a.shape)

    transpose, flip_x, flip_y = (jnp.asarray(tab.flags)[sym, i] for i in range(3))
    x, y = state.head[..., 0], state.head[..., 1]
    x, y = jnp.where(transpose, y, x), jnp.where(transpose, x, y)
    x = jnp.where(flip_x, game.width - 1 - x, x)
    y = jnp.where(flip_y, game.height - 1 - y, y)
    perm = jnp.asarray(tab.perm)[sym]
    return state._replace(
        body=grid(state.body),
        food=grid(state.food),
        hazard=grid(state.hazard),
        head=jnp.stack([x, y], axis=-1).astype(state.head.dtype),
        last_move=perm[state.last_move.astype(jnp.int32)].astype(state.last_move.dtype),
    )


def transform_moves(x: jax.Array, sym: jax.Array | int, game: GameConfig) -> jax.Array:
    """A per-move array ``x[..., 4]`` of a position, moved with the position."""
    inverse = jnp.asarray(symmetry_tables(game.width, game.height).inverse)[sym]
    return jnp.take(x, inverse, axis=-1)


def augments(cfg: Config, group: Group) -> bool:
    """Whether the group's positions are augmented (not with map state: royale)."""
    return cfg.augment and group.game.map != "royale"


# --- Replay with outcomes that arrive later (az.py's) -------------------------------------


class Replay(NamedTuple):
    """One mode's positions and the outcomes of finished games (see ``az.py``'s ``Replay``).

    A position is a valid sample iff its game's outcome row holds its own uid.
    """

    state: State  # [C]
    policy: jax.Array  # f32[C, N, 4]
    uid: jax.Array  # int32[C]
    outcome_uid: jax.Array  # int32[C] (-1: none yet)
    outcome: jax.Array  # f32[C, N]
    ptr: jax.Array
    size: jax.Array


def replay_init(example: State, capacity: int) -> Replay:
    n = example.alive.shape[0]

    def zeros(*shape: int, dtype=jnp.float32) -> jax.Array:
        return jnp.zeros((capacity, *shape), dtype)

    return Replay(
        state=jax.tree.map(lambda x: zeros(*x.shape, dtype=x.dtype), example),
        policy=zeros(n, NUM_ACTIONS),
        uid=zeros(dtype=jnp.int32),
        outcome_uid=jnp.full((capacity,), -1, jnp.int32),
        outcome=zeros(n),
        ptr=jnp.zeros((), jnp.int32),
        size=jnp.zeros((), jnp.int32),
    )


def replay_add(replay: Replay, state: State, policy: jax.Array, uid: jax.Array) -> Replay:
    k, capacity = uid.shape[0], replay.uid.shape[0]
    if capacity % k:
        raise ValueError(f"capacity {capacity} is not a multiple of the batch size {k}")

    def put(buf: jax.Array, x: jax.Array) -> jax.Array:
        return jax.lax.dynamic_update_slice_in_dim(buf, x.astype(buf.dtype), replay.ptr, axis=0)

    return replay._replace(
        state=jax.tree.map(put, replay.state, state),
        policy=put(replay.policy, policy),
        uid=put(replay.uid, uid),
        ptr=(replay.ptr + k) % capacity,
        size=jnp.minimum(replay.size + k, capacity),
    )


def replay_record_outcomes(
    replay: Replay, uid: jax.Array, done: jax.Array, outcome: jax.Array
) -> Replay:
    capacity = replay.uid.shape[0]
    rows = jnp.where(done, uid % capacity, capacity)  # rows of running games are dropped
    return replay._replace(
        outcome_uid=replay.outcome_uid.at[rows].set(uid, mode="drop"),
        outcome=replay.outcome.at[rows].set(outcome, mode="drop"),
    )


def replay_lookup(replay: Replay, idx: jax.Array) -> tuple[jax.Array, jax.Array]:
    uid = replay.uid[idx]
    row = uid % replay.uid.shape[0]
    return replay.outcome_uid[row] == uid, replay.outcome[row]


class Batch(NamedTuple):
    """One mode's minibatch of ``B`` positions; per-snake fields are ``[B, N, ...]``."""

    boards: jax.Array
    feats: jax.Array
    legal: jax.Array  # bool[B, N, 4]
    target: jax.Array  # f32[B, N, 4] root visit share
    z: jax.Array  # f32[B, N] placement outcome
    w_policy: jax.Array  # bool[B, N]
    w_value: jax.Array  # bool[B, N]
    finished: jax.Array  # bool[B]


def sample_batch(replay: Replay, key: jax.Array, cfg: Config, group: Group) -> Batch:
    """``cfg.batch_size`` positions of one mode, drawn uniformly from the filled slots.

    The value loss counts the snakes alive at the position, the policy loss those of
    them not certain to starve; positions of running games count in neither.
    """
    k_idx, k_sym = jax.random.split(key)
    game = group.game
    idx = jax.random.randint(k_idx, (cfg.batch_size,), 0, jnp.maximum(replay.size, 1))
    state = jax.tree.map(lambda x: x[idx], replay.state)
    target = replay.policy[idx]
    finished, z = replay_lookup(replay, idx)
    if augments(cfg, group):
        count = symmetry_tables(game.width, game.height).perm.shape[0]
        sym = jax.random.randint(k_sym, (cfg.batch_size,), 0, count)
        state = jax.vmap(lambda s, g: transform_state(s, g, game))(state, sym)
        target = jax.vmap(lambda t, g: transform_moves(t, g, game))(target, sym)
    boards, feats, legal = unified.group_inputs(cfg, group, state)
    starving = jax.vmap(lambda s: rules.certainly_starving(s, game))(state)
    w_value = finished[:, None] & state.alive
    return Batch(boards, feats, legal, target, z, w_value & ~starving, w_value, finished)


# --- Loss ---------------------------------------------------------------------------------

LOSS_METRICS = (
    "policy_loss", "value_loss", "target_entropy", "net_entropy", "value_abs_mean",
    "value_target_var", "masked_frac",
)  # fmt: skip


def modes_loss(
    params: Params, batches: tuple[Batch, ...], cfg: Config, groups: tuple[Group, ...]
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The weighted mean over the modes of ``az.py``'s loss (cross-entropy to the visit
    shares plus ``value_coef`` times the value MSE), one forward pass on every batch.
    Returns ``(loss, parts)``, each part ``f32[G]``."""
    sizes = tuple(b.target.shape[0] for b in batches)
    boards = unified.join([b.boards for b in batches])
    feats = unified.join([b.feats for b in batches])
    logits_all, values_all = network(params, boards, feats, cfg)
    logits_g = unified.split(logits_all, groups, sizes)
    values_g = unified.split(values_all, groups, sizes)
    loss, parts = jnp.zeros(()), []
    for g, (b, logits, value) in enumerate(zip(batches, logits_g, values_g, strict=True)):
        log_probs = unified.masked_log_probs(logits, b.legal)
        target = jnp.where(b.legal, b.target, 0.0)
        cross_entropy = -jnp.sum(target * log_probs, axis=-1)
        wp, wv = b.w_policy.astype(jnp.float32), b.w_value.astype(jnp.float32)

        def mean(x: jax.Array, w: jax.Array) -> jax.Array:
            return jnp.sum(w * x) / jnp.maximum(jnp.sum(w), 1.0)

        policy_loss = mean(cross_entropy, wp)
        value_loss = mean((value - b.z) ** 2, wv)
        loss = loss + cfg.loss_weights[g] * (policy_loss + cfg.value_coef * value_loss)
        target_log = jnp.log(jnp.where(target > 0, target, 1.0))
        z_mean = mean(b.z, wv)
        parts.append(
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "target_entropy": mean(-jnp.sum(target * target_log, axis=-1), wp),
                "net_entropy": mean(unified.masked_entropy(log_probs, b.legal), wp),
                "value_abs_mean": mean(jnp.abs(value), wv),
                "value_target_var": mean((b.z - z_mean) ** 2, wv),
                "masked_frac": 1.0 - jnp.mean(b.finished.astype(jnp.float32)),
            }
        )
    return loss, {k: jnp.stack([p[k] for p in parts]) for k in LOSS_METRICS}


# --- Training ------------------------------------------------------------------------------


class RunnerState(NamedTuple):
    """Everything the jitted training loop carries between chunks."""

    params: Params
    opt_state: Any
    env_states: tuple[State, ...]  # per group, [num_envs] running games (never done)
    uids: tuple[jax.Array, ...]  # per group, int32[num_envs] game ids (see Replay)
    replays: tuple[Replay, ...]
    key: jax.Array
    env_steps: jax.Array  # int32[] env steps taken, per mode
    updates: jax.Array  # int32[] gradient steps taken


class Positions(NamedTuple):
    """What a self-play turn records per game of one group (leaves ``[E, ...]``)."""

    state: State
    policy: jax.Array  # f32[E, N, 4]
    uid: jax.Array  # int32[E]
    done: jax.Array  # bool[E] the game ended at t + 1
    outcome: jax.Array  # f32[E, N] placement values of s_{t+1} (used where done)


def make_optimizer(cfg: Config) -> optax.GradientTransformation:
    """AdamW without its learning rate (applied by the update): clipping, Adam's scaling,
    decoupled weight decay."""
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.scale_by_adam(eps=cfg.adam_eps),
        optax.add_decayed_weights(cfg.weight_decay),
    )


def init_runner(cfg: Config, groups: tuple[Group, ...], key: jax.Array) -> RunnerState:
    k_net, k_env, k_run = jax.random.split(key, 3)
    params = init_network(k_net, cfg)
    env_states = tuple(
        jax.vmap(g.sim.init_state)(jax.random.split(unified.group_key(k_env, g), cfg.num_envs))
        for g in groups
    )
    return RunnerState(
        params=params,
        opt_state=make_optimizer(cfg).init(params),
        env_states=env_states,
        uids=tuple(jnp.arange(cfg.num_envs, dtype=jnp.int32) for _ in groups),
        replays=tuple(
            replay_init(jax.tree.map(lambda x: x[0], s), cfg.capacity) for s in env_states
        ),
        key=k_run,
        env_steps=jnp.zeros((), jnp.int32),
        updates=jnp.zeros((), jnp.int32),
    )


def make_selfplay(cfg: Config, groups: tuple[Group, ...]) -> Callable[..., Any]:
    """``selfplay(params, states, uids, key) -> ((states, uids), positions, metrics)``:
    ``num_steps`` turns of every group's games (see :func:`make_train_chunk`)."""
    mcfg = selfplay_search(cfg)
    num_envs = cfg.num_envs

    def turn_keys(key, group):
        """(search keys [E], move, step, reset keys) of a turn of one mode."""
        k_search, k_move, k_step, k_reset = jax.random.split(unified.group_key(key, group), 4)
        return jax.random.split(k_search, num_envs), k_move, k_step, k_reset

    def turn_group(params, group, states, uids, key, draws):
        keys, k_move, k_step, k_reset = turn_keys(key, group)
        evaluator = make_evaluator(params, cfg, group)
        out: SearchOutput = jax.vmap(
            lambda k, s, d: search(k, s, group.sim, mcfg, evaluator, root_draws=d)
        )(keys, states, draws)
        log_policy = jnp.where(out.policy > 0, jnp.log(jnp.maximum(out.policy, 1e-30)), -jnp.inf)
        sampled = jax.random.categorical(k_move, log_policy).astype(jnp.int32)
        explore = (states.turn < cfg.temperature_turns)[:, None]
        actions = jnp.where(explore, sampled, out.action)
        new, ts = jax.vmap(group.sim.step)(jax.random.split(k_step, num_envs), states, actions)
        fresh = jax.vmap(group.sim.init_state)(jax.random.split(k_reset, num_envs))
        next_states = unified.tree_where(ts.done, fresh, new)
        next_uids = jnp.where(ts.done, uids + num_envs, uids)
        starving = jax.vmap(lambda s: rules.certainly_starving(s, group.game))(states)
        metrics = {
            "agent_samples": jnp.sum(states.alive & ~starving),
            **unified.game_metrics(states, new, ts.done, ts.truncated),
            "search_depth": jnp.sum(out.depth),
        }
        outcome = jax.vmap(lambda s: placement_values(s, group.game, cfg.draw_value))(new)
        record = Positions(states, out.policy, uids, ts.done, outcome)
        return (next_states, next_uids), record, metrics

    def turn(params, carry, xs):
        states, uids = carry
        key, draws = xs
        out = [
            turn_group(params, g, s, u, key, d)
            for g, s, u, d in zip(groups, states, uids, draws, strict=True)
        ]
        next_states = tuple(o[0][0] for o in out)
        next_uids = tuple(o[0][1] for o in out)
        records = tuple(o[1] for o in out)
        metrics = {k: jnp.stack([o[2][k] for o in out]) for k in out[0][2]}
        return (next_states, next_uids), (records, metrics)

    def selfplay(params, states, uids, key):
        keys = jax.random.split(key, cfg.num_steps)
        # The root noise's draws only depend on the keys: one batched call per mode for the
        # whole chunk (the rejection loops of ``loggamma`` cost a host sync per iteration on
        # a GPU, and under the scan they would run once per turn and game batch).

        def chunk_draws(g):  # [T, E, N, 4]
            def turn_draws(k):
                return jax.vmap(lambda kk: root_noise_draws(kk, g.num_snakes, mcfg))(
                    turn_keys(k, g)[0]
                )

            return jax.vmap(turn_draws)(keys) if mcfg.root_noise_frac > 0 else None

        draws = tuple(chunk_draws(g) for g in groups)
        carry, (traj, metrics) = jax.lax.scan(
            lambda c, x: turn(params, c, x), (states, uids), (keys, draws)
        )
        return carry, traj, jax.tree.map(lambda x: jnp.sum(x, axis=0), metrics)

    return selfplay


def make_train_chunk(
    cfg: Config, groups: tuple[Group, ...]
) -> Callable[[RunnerState], tuple[RunnerState, dict[str, jax.Array]]]:
    """``train_chunk(runner) -> (runner, metrics)``: ``chunk_iters`` iterations in one scan.

    An iteration plays ``num_steps`` turns of self-play in every mode, stores the
    positions and the outcomes of the games that ended in each mode's replay, then takes
    ``updates_per_iteration`` gradient steps on one minibatch of each mode (none until
    every buffer holds ``learning_starts`` positions).
    """
    optimizer = make_optimizer(cfg)
    selfplay = make_selfplay(cfg, groups)
    loss_and_grad = jax.value_and_grad(
        functools.partial(modes_loss, cfg=cfg, groups=groups), has_aux=True
    )
    num_updates = cfg.updates_per_iteration
    k = cfg.steps_per_iteration
    num_groups = len(groups)

    def gradient_step(replays, lr, carry, key):
        params, opt_state = carry
        keys = jax.random.split(key, num_groups)
        batches = tuple(
            sample_batch(r, kk, cfg, g) for r, kk, g in zip(replays, keys, groups, strict=True)
        )
        (loss, parts), grads = loss_and_grad(params, batches)
        step, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, unified.scale_updates(step, lr))
        return (params, opt_state), {**parts, "loss": loss, "grad_norm": optax.tree.norm(grads)}

    def learn(replays, lr, carry, key):
        keys = jax.random.split(key, num_updates)
        carry, parts = jax.lax.scan(lambda c, kk: gradient_step(replays, lr, c, kk), carry, keys)
        return carry, jax.tree.map(lambda x: jnp.sum(x, axis=0), parts)

    def iteration(runner: RunnerState, _):
        key, k_play, k_learn = jax.random.split(runner.key, 3)
        (env_states, uids), traj, metrics = selfplay(
            runner.params, runner.env_states, runner.uids, k_play
        )

        def flat(x: jax.Array) -> jax.Array:  # [T, E, ...] -> [T * E, ...]
            return x.reshape(k, *x.shape[2:])

        replays = []
        for replay, t in zip(runner.replays, traj, strict=True):
            replay = replay_add(replay, jax.tree.map(flat, t.state), flat(t.policy), flat(t.uid))
            replays.append(
                replay_record_outcomes(replay, flat(t.uid), flat(t.done), flat(t.outcome))
            )
        replays = tuple(replays)
        carry = (runner.params, runner.opt_state)
        learning = jnp.all(jnp.stack([r.size >= cfg.learning_starts for r in replays]))
        lr = cfg.lr * unified.lr_factor(runner.env_steps, cfg)
        skipped = {
            **{name: jnp.zeros(num_groups, jnp.float32) for name in LOSS_METRICS},
            "loss": jnp.zeros((), jnp.float32),
            "grad_norm": jnp.zeros((), jnp.float32),
        }
        (params, opt_state), parts = jax.lax.cond(
            learning, lambda: learn(replays, lr, carry, k_learn), lambda: (carry, skipped)
        )
        updates = jnp.where(learning, num_updates, 0).astype(jnp.int32)
        metrics.update(parts)
        metrics["updates"] = updates
        runner = RunnerState(
            params=params,
            opt_state=opt_state,
            env_states=env_states,
            uids=uids,
            replays=replays,
            key=key,
            env_steps=runner.env_steps + k,
            updates=runner.updates + updates,
        )
        return runner, metrics

    def train_chunk(runner: RunnerState) -> tuple[RunnerState, dict[str, jax.Array]]:
        runner, metrics = jax.lax.scan(iteration, runner, None, length=cfg.chunk_iters)
        metrics = jax.tree.map(lambda x: jnp.sum(x, axis=0), metrics)
        metrics["env_steps"] = runner.env_steps
        metrics["total_updates"] = runner.updates
        metrics["buffer_size"] = runner.replays[0].size
        metrics["lr"] = cfg.lr * unified.lr_factor(runner.env_steps, cfg)
        return runner, metrics

    return train_chunk


# --- Records ----------------------------------------------------------------------------


def chunk_record(m: dict[str, Any], seconds: float, chunk_steps: int, cfg: Config) -> dict:
    """A JSON-friendly summary of one chunk: ``az.py``'s fields at the top level (the losses
    weighted means over the modes, the game statistics the first mode's) and a
    ``game_modes`` block with every mode's."""
    n_upd = max(int(m["updates"]), 1)
    weights = cfg.loss_weights
    per_mode = {}
    for g, (name, n) in enumerate(cfg.snakes.items()):
        turns = chunk_steps
        losses = {k: float(m[k][g]) / n_upd for k in LOSS_METRICS}
        var = losses["value_target_var"]
        per_mode[name] = {
            "num_snakes": n,
            **unified.mode_game_stats(m, g),
            "agent_samples": int(m["agent_samples"][g]),
            **losses,
            "explained_var": unified.finite(1.0 - losses["value_loss"] / var) if var > 0 else None,
            "search_depth": float(m["search_depth"][g]) / max(turns, 1),
        }
    head = per_mode[cfg.game_modes[0]]
    modes_ = list(per_mode.values())
    return {
        "type": "train",
        "env_steps": int(m["env_steps"]),
        "env_steps_all": int(m["env_steps"]) * len(cfg.game_modes),
        "agent_samples": sum(p["agent_samples"] for p in modes_),
        "updates": int(m["updates"]),
        "total_updates": int(m["total_updates"]),
        "lr": float(m["lr"]),
        "loss": float(m["loss"]) / n_upd,
        **{
            k: sum(w * p[k] for w, p in zip(weights, modes_, strict=True))
            for k in ("policy_loss", "value_loss", "net_entropy", "target_entropy")
        },
        "grad_norm": float(m["grad_norm"]) / n_upd,
        "buffer_size": int(m["buffer_size"]),
        **{k: head[k] for k in ("games", "mean_game_len", "draws", "truncated", "deaths")},
        "env_steps_per_s": chunk_steps / seconds,
        "seconds": seconds,
        "game_modes": per_mode,
    }


def format_record(r: dict[str, Any], total_steps: int) -> str:
    head = (
        f"steps {r['env_steps']:>10,} ({100 * r['env_steps'] / total_steps:3.0f}%) per mode | "
        f"loss {r['loss']:.4f} pi {r['policy_loss']:.4f} v {r['value_loss']:.4f} "
        f"ent {r['net_entropy']:.3f}/{r['target_entropy']:.3f} lr {r['lr']:.2e} "
        f"buffer {r['buffer_size']:,} | {r['env_steps_per_s']:,.0f} steps/s per mode"
    )
    lines = [head]
    for name, p in r["game_modes"].items():
        ev = f"{p['explained_var']:.2f}" if p["explained_var"] is not None else "-"
        lines.append(
            f"  {name:>11} | pi {p['policy_loss']:.4f} v {p['value_loss']:.4f} ev {ev} "
            f"ent {p['net_entropy']:.3f}/{p['target_entropy']:.3f} "
            f"masked {p['masked_frac']:.2f} | {unified.format_games(p)}"
        )
    return "\n".join(lines)


# --- Checkpoints ----------------------------------------------------------------------


def load_config(run_dir: str) -> Config:
    return unified.load_config(run_dir, ALGORITHM, Config)


def load_params(run_dir: str, cfg: Config | None = None, name: str = "params.npz") -> Params:
    """The parameters saved in ``run_dir/<name>`` (a snapshot name works too)."""
    cfg = cfg or load_config(run_dir)
    template = init_network(jax.random.key(0), cfg)
    return unified.load_params_like(template, os.path.join(run_dir, name))


if __name__ == "__main__":
    unified.main(sys.modules[__name__])
