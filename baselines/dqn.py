"""Self-play DQN baseline for the 1v1 duel (single file, cleanRL / PureJaxRL style).

* **Self-play with parameter sharing.** One Q-network acts for both snakes,
  each from its own egocentric observation, epsilon-greedily *among its legal
  moves* (``env.action_mask``).
* **Vectorized.** ``num_envs`` games run in parallel; a finished game is
  replaced by a fresh one right away.
* **Compact replay.** The circular buffer stores game-level transitions as
  pairs of :class:`~slinky.types.State` (under 1 KB each) instead of
  observations (23 KB per agent); observations, and the action masks of the
  next states, are recomputed at sample time. A sampled game transition gives
  one agent-transition per snake.
* **Learning.** Double DQN targets (argmax over legal moves), Huber loss, Adam
  with gradient clipping and a target network (hard copy or Polyak averaging).

Everything between two log lines (acting, stepping, storing and the gradient
updates) is one jitted ``lax.scan`` (:func:`make_train_chunk`). The Python loop
only logs, evaluates the greedy policy against ``random_legal`` and saves
checkpoints::

    python baselines/dqn.py                                   # default run
    python baselines/dqn.py --total-env-steps 100000 --lr 5e-4
    python baselines/dqn.py --eval-only runs/dqn-20261009-120000

An *env step* is one game advancing one turn. It gives one *agent-transition*
per snake, so two in a duel.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import os
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from slinky.env import BattlesnakeEnv
from slinky.evaluate import MatchResult, greedy_from_q, play_match, random_legal
from slinky.observations import obs_shape
from slinky.types import NUM_ACTIONS, Cause, GameConfig, State

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
_NUM_CAUSES = len(Cause)


@dataclasses.dataclass(frozen=True)
class DQNConfig:
    """All hyperparameters. Every field can be overridden on the command line.

    Steps are *env steps* (games advanced one turn); the buffer holds game
    transitions, each worth ``num_snakes`` (2) agent-transitions.
    """

    # Game: the 11x11 standard duel, truncated after max_turns turns.
    max_turns: int = 1000
    # Collection.
    num_envs: int = 32  # games stepped in parallel
    total_env_steps: int = 1_280_000  # rounded up to whole log chunks
    # Replay. The capacity is rounded down to a multiple of num_envs.
    buffer_capacity: int = 100_000  # game transitions (2x agent-transitions)
    learning_starts: int = 10_000  # buffer size (game transitions) before learning
    batch_size: int = 128  # game transitions per update (2x agent samples)
    updates_per_step: int = 1  # gradient updates per vectorized step of num_envs games
    # Optimization.
    lr: float = 5e-4
    max_grad_norm: float = 10.0
    gamma: float = 0.99
    target_update_period: int = 250  # gradient updates between target updates
    tau: float = 1.0  # target <- tau * online + (1 - tau) * target; 1.0 is a hard copy
    # Exploration: linear decay over the first eps_decay_fraction of training.
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_fraction: float = 0.3
    # Network: 3x3 convs (ReLU), a dense hidden layer (ReLU), then a linear Q head.
    conv_channels: tuple[int, ...] = (32, 64, 64)
    conv_strides: tuple[int, ...] = (2, 2, 1)
    hidden: int = 256
    dueling: bool = False
    # Logging, evaluation and checkpoints.
    seed: int = 0
    log_every: int = 16_000  # env steps per jitted chunk (one log line)
    eval_every: int = 256_000  # env steps between evaluations (and checkpoints)
    eval_games: int = 1000  # greedy DQN vs random_legal
    run_dir: str | None = None  # default: runs/dqn-<timestamp>

    def __post_init__(self) -> None:
        if len(self.conv_channels) != len(self.conv_strides):
            raise ValueError("conv_channels and conv_strides must have the same length")
        if self.buffer_capacity < self.num_envs:
            raise ValueError("buffer_capacity must be at least num_envs")
        positive = (
            "max_turns", "num_envs", "total_env_steps", "batch_size", "updates_per_step",
            "target_update_period", "log_every", "eval_every",
        )  # fmt: skip
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")

    @property
    def game(self) -> GameConfig:
        return GameConfig(max_turns=self.max_turns)


# --- Q-network -------------------------------------------------------------------


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

    The forward pass is XLA's convolution. The backward pass differentiates the
    equivalent im2col form (:func:`_conv_im2col`) instead: on XLA:CPU (jax
    0.11) convolution *gradients* inside a ``lax.scan`` run 3-50x slower than
    outside one, which made a training chunk ~20x slower; matmuls don't suffer.
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


def init_network(key: jax.Array, cfg: DQNConfig) -> Params:
    """He-normal weights and zero biases for :func:`q_network`."""
    rows, cols, cin = obs_shape(cfg.game)
    keys = iter(jax.random.split(key, len(cfg.conv_channels) + 3))

    def layer(shape: tuple[int, ...], fan_in: int, gain: float = 2.0) -> dict[str, jax.Array]:
        w = jax.random.normal(next(keys), shape) * math.sqrt(gain / fan_in)
        return {"w": w, "b": jnp.zeros(shape[-1])}

    params: Params = {}
    for i, (cout, stride) in enumerate(zip(cfg.conv_channels, cfg.conv_strides, strict=True)):
        params[f"conv{i}"] = layer((3, 3, cin, cout), 9 * cin)
        rows, cols, cin = -(-rows // stride), -(-cols // stride), cout  # "SAME" padding
    params["hidden"] = layer((rows * cols * cin, cfg.hidden), rows * cols * cin)
    if cfg.dueling:
        params["value"] = layer((cfg.hidden, 1), cfg.hidden, gain=1.0)
        params["advantage"] = layer((cfg.hidden, NUM_ACTIONS), cfg.hidden, gain=1.0)
    else:
        params["q"] = layer((cfg.hidden, NUM_ACTIONS), cfg.hidden, gain=1.0)
    return params


def q_network(params: Params, obs: jax.Array, cfg: DQNConfig) -> jax.Array:
    """Q-values ``f32[..., 4]`` for observations ``f32[..., rows, cols, C]``."""
    lead = obs.shape[:-3]
    x = obs.reshape(-1, *obs.shape[-3:])
    for i, stride in enumerate(cfg.conv_strides):
        p = params[f"conv{i}"]
        x = jax.nn.relu(conv3x3(x, p["w"], stride) + p["b"])
    x = x.reshape(x.shape[0], -1)
    x = jax.nn.relu(x @ params["hidden"]["w"] + params["hidden"]["b"])
    if cfg.dueling:
        v = x @ params["value"]["w"] + params["value"]["b"]
        a = x @ params["advantage"]["w"] + params["advantage"]["b"]
        q = v + a - a.mean(axis=-1, keepdims=True)
    else:
        q = x @ params["q"]["w"] + params["q"]["b"]
    return q.reshape(*lead, NUM_ACTIONS)


def epsilon_greedy(key: jax.Array, q: jax.Array, mask: jax.Array, eps: jax.Array) -> jax.Array:
    """Per agent: a uniformly random legal action with probability ``eps``, else the legal argmax.

    ``q`` and ``mask`` are ``[..., 4]``; mask rows are never all-False.
    """
    k_explore, k_random = jax.random.split(key)
    greedy = jnp.argmax(jnp.where(mask, q, -jnp.inf), axis=-1)
    random = jax.random.categorical(k_random, jnp.where(mask, 0.0, -jnp.inf), axis=-1)
    explore = jax.random.uniform(k_explore, mask.shape[:-1]) < eps
    return jnp.where(explore, random, greedy).astype(jnp.int32)


def epsilon(cfg: DQNConfig, env_steps: jax.Array) -> jax.Array:
    """Linear decay from ``eps_start`` to ``eps_end`` over ``eps_decay_fraction`` of training."""
    decay_steps = max(cfg.eps_decay_fraction * cfg.total_env_steps, 1.0)
    frac = jnp.clip(env_steps / decay_steps, 0.0, 1.0)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


# --- Replay buffer ---------------------------------------------------------------


class Transition(NamedTuple):
    """One game transition (all snakes). Batched leaves have a leading axis."""

    state: State  # s_t; never done, since finished games are reset right away
    actions: jax.Array  # int32[N]
    rewards: jax.Array  # float32[N]
    next_state: State  # s_{t+1} as reached, before any autoreset
    truncated: jax.Array  # bool[] the game was cut off by max_turns at t+1


class Buffer(NamedTuple):
    """Fixed-capacity circular buffer over a pytree of items (leaves ``[capacity, ...]``)."""

    data: Any
    ptr: jax.Array  # int32[] next write position
    size: jax.Array  # int32[] number of filled slots


def buffer_init(example: Any, capacity: int) -> Buffer:
    """An empty buffer for items shaped like ``example`` (one item, no leading axis)."""
    data = jax.tree.map(lambda x: jnp.zeros((capacity, *x.shape), x.dtype), example)
    return Buffer(data, jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32))


def buffer_add(buf: Buffer, items: Any) -> Buffer:
    """Write a batch of ``B`` items at ``ptr``, overwriting the oldest ones once full.

    The capacity must be a multiple of ``B`` (always adding ``B`` items), so a
    batch never straddles the end and is one contiguous in-place update.
    """
    batch = jax.tree.leaves(items)[0].shape[0]
    capacity = jax.tree.leaves(buf.data)[0].shape[0]
    if capacity % batch:
        raise ValueError(f"capacity {capacity} is not a multiple of the batch size {batch}")
    data = jax.tree.map(
        lambda d, x: jax.lax.dynamic_update_slice_in_dim(d, x.astype(d.dtype), buf.ptr, axis=0),
        buf.data,
        items,
    )
    return Buffer(data, (buf.ptr + batch) % capacity, jnp.minimum(buf.size + batch, capacity))


def buffer_sample(buf: Buffer, key: jax.Array, batch_size: int) -> Any:
    """``batch_size`` items drawn uniformly with replacement from the filled slots."""
    idx = jax.random.randint(key, (batch_size,), 0, jnp.maximum(buf.size, 1))
    return jax.tree.map(lambda d: d[idx], buf.data)


# --- Targets and loss --------------------------------------------------------------


def agent_flags(
    state: State, next_state: State, truncated: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Per-agent ``(valid, terminal)`` flags, ``bool[..., N]``, of game transitions.

    * ``valid``: the agent was alive in a running game at ``t``, so it acted and
      the transition is its own. Others are left out of the loss.
    * ``terminal``: no bootstrapping from ``s_{t+1}``. The agent died, or the
      rules ended the game: the winner's +1 and a draw's 0 are final. Agents
      still alive when ``max_turns`` cuts the game off bootstrap.
    """
    valid = state.alive & ~state.done[..., None]
    ended = next_state.done & ~truncated
    terminal = ~next_state.alive | ended[..., None]
    return valid, terminal


def td_targets(
    tr: Transition,
    q_next_online: jax.Array,
    q_next_target: jax.Array,
    next_mask: jax.Array,
    gamma: float,
) -> tuple[jax.Array, jax.Array]:
    """Double DQN targets and validity flags, ``f32[..., N]`` and ``bool[..., N]``.

    ``a* = argmax`` of the online Q over the legal moves at ``s_{t+1}``;
    ``target = r + gamma * Q_target(s_{t+1}, a*)``, without the bootstrap term
    for terminal agents.
    """
    valid, terminal = agent_flags(tr.state, tr.next_state, tr.truncated)
    a_star = jnp.argmax(jnp.where(next_mask, q_next_online, -jnp.inf), axis=-1)
    q_next = jnp.take_along_axis(q_next_target, a_star[..., None], axis=-1)[..., 0]
    return tr.rewards + gamma * jnp.where(terminal, 0.0, q_next), valid


def make_loss_fn(
    env: BattlesnakeEnv, cfg: DQNConfig
) -> Callable[[Params, Params, Transition], tuple[jax.Array, tuple[jax.Array, jax.Array]]]:
    """``loss(params, target_params, batch) -> (loss, (grads, mean_q))`` on game transitions."""
    observe = jax.vmap(env.observe)  # [B] states -> [B, N, rows, cols, C]
    masks = jax.vmap(env.action_mask)  # [B] states -> [B, N, 4]

    def td_loss(params, obs, actions, target, valid):
        q = q_network(params, obs, cfg)
        q_taken = jnp.take_along_axis(q, actions[..., None], axis=-1)[..., 0]
        w = valid.astype(jnp.float32)
        count = jnp.maximum(w.sum(), 1.0)
        loss = jnp.sum(w * optax.huber_loss(q_taken, target)) / count
        return loss, jnp.sum(w * q_taken) / count

    def loss_and_grad(params, target_params, tr):
        next_obs = observe(tr.next_state)
        target, valid = td_targets(
            tr,
            q_network(params, next_obs, cfg),
            q_network(target_params, next_obs, cfg),
            masks(tr.next_state),
            cfg.gamma,
        )
        target = jax.lax.stop_gradient(target)
        (loss, mean_q), grads = jax.value_and_grad(td_loss, has_aux=True)(
            params, observe(tr.state), tr.actions, target, valid
        )
        return loss, (grads, mean_q)

    return loss_and_grad


# --- Training ----------------------------------------------------------------------


class RunnerState(NamedTuple):
    """Everything the jitted training loop carries between chunks."""

    params: Params
    target_params: Params
    opt_state: Any
    env_states: State  # [num_envs] running games (never done)
    buffer: Buffer
    key: jax.Array
    env_steps: jax.Array  # int32[] env steps taken
    updates: jax.Array  # int32[] gradient updates taken


def make_envs(cfg: DQNConfig) -> tuple[BattlesnakeEnv, BattlesnakeEnv]:
    """``(env, sim)``: ``env`` computes egocentric observations, ``sim`` steps without them."""
    return BattlesnakeEnv(cfg.game, obs="egocentric"), BattlesnakeEnv(cfg.game, obs=None)


def make_optimizer(cfg: DQNConfig) -> optax.GradientTransformation:
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))


def init_runner(cfg: DQNConfig, sim: BattlesnakeEnv, key: jax.Array) -> RunnerState:
    k_net, k_env, k_run = jax.random.split(key, 3)
    params = init_network(k_net, cfg)
    env_states = jax.vmap(sim.init_state)(jax.random.split(k_env, cfg.num_envs))
    one = jax.tree.map(lambda x: x[0], env_states)
    n = sim.num_agents
    example = Transition(
        state=one,
        actions=jnp.zeros((n,), jnp.int32),
        rewards=jnp.zeros((n,), jnp.float32),
        next_state=one,
        truncated=jnp.zeros((), bool),
    )
    capacity = cfg.buffer_capacity // cfg.num_envs * cfg.num_envs
    return RunnerState(
        params=params,
        # A copy: the runner is donated, and one buffer can't be donated twice.
        target_params=jax.tree.map(jnp.copy, params),
        opt_state=make_optimizer(cfg).init(params),
        env_states=env_states,
        buffer=buffer_init(example, capacity),
        key=k_run,
        env_steps=jnp.zeros((), jnp.int32),
        updates=jnp.zeros((), jnp.int32),
    )


def make_train_chunk(
    cfg: DQNConfig, env: BattlesnakeEnv, sim: BattlesnakeEnv, num_iters: int
) -> Callable[[RunnerState], tuple[RunnerState, dict[str, jax.Array]]]:
    """``train_chunk(runner) -> (runner, metrics)``: ``num_iters`` iterations in one ``lax.scan``.

    Each iteration acts in and steps all ``num_envs`` games, stores the
    transitions, and once the buffer holds ``learning_starts`` transitions takes
    ``updates_per_step`` gradient steps. Metrics are summed over the chunk.
    """
    optimizer = make_optimizer(cfg)
    loss_and_grad = make_loss_fn(env, cfg)
    num_envs = cfg.num_envs
    start = max(cfg.learning_starts, 1)

    def gradient_step(buffer, carry, key):
        params, target_params, opt_state, updates = carry
        batch = buffer_sample(buffer, key, cfg.batch_size)
        loss, (grads, mean_q) = loss_and_grad(params, target_params, batch)
        step, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, step)
        updates = updates + 1
        sync = updates % cfg.target_update_period == 0
        target_params = jax.tree.map(
            lambda t, p: jnp.where(sync, optax.incremental_update(p, t, cfg.tau), t),
            target_params,
            params,
        )
        return (params, target_params, opt_state, updates), (loss, mean_q)

    def learn(buffer, carry, key):
        keys = jax.random.split(key, cfg.updates_per_step)
        carry, (loss, mean_q) = jax.lax.scan(lambda c, k: gradient_step(buffer, c, k), carry, keys)
        return carry, (loss.sum(), mean_q.sum(), jnp.int32(cfg.updates_per_step))

    def skip(buffer, carry, key):
        return carry, (jnp.float32(0), jnp.float32(0), jnp.int32(0))

    def iteration(runner: RunnerState, _):
        key, k_act, k_step, k_reset, k_learn = jax.random.split(runner.key, 5)
        states = runner.env_states

        # Act: every snake from its own egocentric view, one shared network.
        q = q_network(runner.params, jax.vmap(env.observe)(states), cfg)
        mask = jax.vmap(env.action_mask)(states)
        actions = epsilon_greedy(k_act, q, mask, epsilon(cfg, runner.env_steps))

        # Step, store the transition to the reached state, then reset finished games.
        new, ts = jax.vmap(sim.step)(jax.random.split(k_step, num_envs), states, actions)
        transition = Transition(states, actions, ts.reward, new, ts.truncated)
        buffer = buffer_add(runner.buffer, transition)
        fresh = jax.vmap(sim.init_state)(jax.random.split(k_reset, num_envs))
        env_states = jax.vmap(_tree_where)(ts.done, fresh, new)

        carry = (runner.params, runner.target_params, runner.opt_state, runner.updates)
        carry, (loss, mean_q, n_updates) = jax.lax.cond(
            buffer.size >= start, learn, skip, buffer, carry, k_learn
        )
        params, target_params, opt_state, updates = carry

        died = states.alive & ~new.alive  # [E, N]
        causes = jnp.arange(_NUM_CAUSES) == new.elim_cause[..., None]  # [E, N, causes]
        metrics = {
            "agent_transitions": jnp.sum(states.alive & ~states.done[:, None]),
            "games": jnp.sum(ts.done),
            "game_turns": jnp.sum(jnp.where(ts.done, new.turn, 0)),
            "draws": jnp.sum(ts.done & ~ts.truncated & ~jnp.any(new.alive, axis=-1)),
            "truncated": jnp.sum(ts.truncated),
            "deaths": jnp.sum(causes & died[..., None], axis=(0, 1)),
            "loss": loss,
            "mean_q": mean_q,
            "updates": n_updates,
        }
        runner = RunnerState(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            env_states=env_states,
            buffer=buffer,
            key=key,
            env_steps=runner.env_steps + num_envs,
            updates=updates,
        )
        return runner, metrics

    def train_chunk(runner: RunnerState) -> tuple[RunnerState, dict[str, jax.Array]]:
        runner, metrics = jax.lax.scan(iteration, runner, None, length=num_iters)
        metrics = jax.tree.map(lambda x: jnp.sum(x, axis=0), metrics)
        metrics["eps"] = epsilon(cfg, runner.env_steps)
        metrics["env_steps"] = runner.env_steps
        metrics["total_updates"] = runner.updates
        metrics["buffer_size"] = runner.buffer.size
        return runner, metrics

    return train_chunk


def _tree_where(cond: jax.Array, a: Any, b: Any) -> Any:
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


# --- Evaluation and checkpoints ---------------------------------------------------


def evaluate_params(
    params: Params, cfg: DQNConfig, env: BattlesnakeEnv, key: jax.Array, num_games: int
) -> MatchResult:
    """Greedy DQN (legal argmax) vs ``random_legal``, seats balanced.

    The network parameters are baked into the compiled match (see
    ``slinky.evaluate``), so every call compiles once more (a few seconds).
    """

    # A private copy: the training loop donates (deletes) the runner's arrays.
    params = jax.tree.map(jnp.copy, params)

    def q_fn(obs: jax.Array) -> jax.Array:
        return q_network(params, obs, cfg)

    return play_match(
        env, greedy_from_q(q_fn), random_legal(env), key, num_games, max_turns=cfg.max_turns
    )


def save_checkpoint(run_dir: str, params: Params) -> str:
    """Write ``params.npz`` (flattened ``layer/name`` keys) atomically; returns its path."""
    flat = {
        "/".join(k.key for k in path): np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
    }
    path = os.path.join(run_dir, "params.npz")
    tmp = os.path.join(run_dir, "params.tmp.npz")
    np.savez(tmp, **flat)
    os.replace(tmp, path)
    return path


def save_config(run_dir: str, cfg: DQNConfig) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)


def load_config(run_dir: str) -> DQNConfig:
    with open(os.path.join(run_dir, "config.json")) as f:
        d = json.load(f)
    fields = {f.name: f for f in dataclasses.fields(DQNConfig)}
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in fields}
    return DQNConfig(**kwargs)


def load_params(run_dir: str, cfg: DQNConfig | None = None) -> Params:
    """The parameters saved in ``run_dir/params.npz`` (structure from ``run_dir/config.json``)."""
    cfg = cfg or load_config(run_dir)
    template = init_network(jax.random.key(0), cfg)
    paths, treedef = jax.tree_util.tree_flatten_with_path(template)
    with np.load(os.path.join(run_dir, "params.npz")) as f:
        leaves = []
        for path, ref in paths:
            leaf = f["/".join(k.key for k in path)]
            if leaf.shape != ref.shape:
                raise ValueError(f"{path}: checkpoint shape {leaf.shape} != {ref.shape}")
            leaves.append(jnp.asarray(leaf))
    return jax.tree.unflatten(treedef, leaves)


def evaluate_run(run_dir: str, num_games: int, seed: int = 0) -> MatchResult:
    """Load ``run_dir``'s checkpoint and play it against ``random_legal``."""
    cfg = load_config(run_dir)
    env, _ = make_envs(cfg)
    return evaluate_params(load_params(run_dir, cfg), cfg, env, jax.random.key(seed), num_games)


# --- Logging -------------------------------------------------------------------------


def _pct(x: float, total: float) -> str:
    return f"{100 * x / total:.0f}%" if total else "-"


def format_eval(r: MatchResult) -> str:
    n = r.num_games
    return (
        f"W {_pct(r.wins, n)} D {_pct(r.draws, n)} L {_pct(r.losses, n)} | "
        f"score {r.score:.3f} +/- {r.score_ci95:.3f} | len {r.mean_turns:.0f} "
        f"| truncated {r.truncated}"
    )


def chunk_record(m: dict[str, Any], seconds: float, chunk_steps: int) -> dict[str, Any]:
    """A JSON-friendly summary of one chunk's summed metrics."""
    games, n_upd = int(m["games"]), int(m["updates"])
    deaths = {name: int(m["deaths"][c]) for name, c in DEATH_CAUSES.items()}
    return {
        "type": "train",
        "env_steps": int(m["env_steps"]),
        "agent_transitions": int(m["agent_transitions"]),
        "total_updates": int(m["total_updates"]),
        "buffer_size": int(m["buffer_size"]),
        "eps": float(m["eps"]),
        "loss": float(m["loss"]) / n_upd if n_upd else None,
        "mean_q": float(m["mean_q"]) / n_upd if n_upd else None,
        "games": games,
        "mean_game_len": float(m["game_turns"]) / games if games else None,
        "draws": int(m["draws"]),
        "truncated": int(m["truncated"]),
        "deaths": deaths,
        "updates_per_s": n_upd / seconds,
        "env_steps_per_s": chunk_steps / seconds,
        "seconds": seconds,
    }


def format_record(r: dict[str, Any], total_steps: int, agent_total: int) -> str:
    deaths = r["deaths"]
    n_dead = sum(deaths.values())
    shown = [k for k in DEATH_CAUSES if k != "hazard" or deaths[k]]
    loss = f"{r['loss']:.4f}" if r["loss"] is not None else "-"
    q = f"{r['mean_q']:+.3f}" if r["mean_q"] is not None else "-"
    length = f"{r['mean_game_len']:.0f}" if r["mean_game_len"] is not None else "-"
    return (
        f"steps {r['env_steps']:>9,} ({100 * r['env_steps'] / total_steps:3.0f}%) "
        f"agent-tr {agent_total:>10,} | eps {r['eps']:.2f} | loss {loss} Q {q} | "
        f"games {r['games']:,} len {length} draw {_pct(r['draws'], r['games'])} | deaths "
        + " ".join(f"{k} {_pct(deaths[k], n_dead)}" for k in shown)
        + f" | {r['updates_per_s']:.0f} upd/s {r['env_steps_per_s']:,.0f} steps/s"
    )


# --- Main loop -----------------------------------------------------------------------


def train(cfg: DQNConfig) -> tuple[str, Params]:
    """Run a full training; returns ``(run_dir, final params)``."""
    run_dir = cfg.run_dir or os.path.join("runs", time.strftime("dqn-%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    cfg = dataclasses.replace(cfg, run_dir=run_dir)
    save_config(run_dir, cfg)

    env, sim = make_envs(cfg)
    key, k_init = jax.random.split(jax.random.key(cfg.seed))
    runner = init_runner(cfg, sim, k_init)

    chunk_iters = max(cfg.log_every // cfg.num_envs, 1)
    chunk_steps = chunk_iters * cfg.num_envs
    num_chunks = -(-cfg.total_env_steps // chunk_steps)
    total_steps = num_chunks * chunk_steps
    n_params = sum(x.size for x in jax.tree.leaves(runner.params))
    state_bytes = sum(x.nbytes for x in jax.tree.leaves(runner.buffer.data))
    print(f"run dir: {run_dir}")
    print(f"config: {json.dumps(dataclasses.asdict(cfg))}")
    print(
        f"{n_params:,} parameters | replay buffer {state_bytes / 2**20:,.0f} MiB | "
        f"{num_chunks} chunks of {chunk_steps:,} env steps = {total_steps:,} env steps"
    )

    t0 = time.perf_counter()
    train_chunk = (
        jax.jit(make_train_chunk(cfg, env, sim, chunk_iters), donate_argnums=0)
        .lower(runner)
        .compile()
    )
    print(f"compiled train_chunk in {time.perf_counter() - t0:.1f}s", flush=True)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    agent_total = 0
    start = time.perf_counter()
    with open(metrics_path, "a") as log:

        def write(record: dict[str, Any]) -> None:
            log.write(json.dumps(record) + "\n")
            log.flush()

        for chunk in range(num_chunks):
            t = time.perf_counter()
            runner, m = train_chunk(runner)
            m = jax.device_get(m)
            seconds = time.perf_counter() - t
            record = chunk_record(m, seconds, chunk_steps)
            agent_total += record["agent_transitions"]
            record["agent_transitions_total"] = agent_total
            record["elapsed"] = time.perf_counter() - start
            write(record)
            print(format_record(record, total_steps, agent_total), flush=True)

            steps = record["env_steps"]
            last = chunk == num_chunks - 1
            if last or steps // cfg.eval_every > (steps - chunk_steps) // cfg.eval_every:
                save_checkpoint(run_dir, runner.params)
                if cfg.eval_games > 0:
                    key, k_eval = jax.random.split(key)
                    t = time.perf_counter()
                    r = evaluate_params(runner.params, cfg, env, k_eval, cfg.eval_games)
                    seconds = time.perf_counter() - t
                    write(
                        {
                            "type": "eval",
                            "env_steps": steps,
                            "opponent": "random_legal",
                            **r._asdict(),
                            "seconds": seconds,
                        }
                    )
                    print(f"eval @ {steps:,} vs random_legal: {format_eval(r)} ({seconds:.0f}s)")
    print(f"done in {time.perf_counter() - start:.0f}s; checkpoint in {run_dir}")
    return run_dir, runner.params


def _parse_bool(text: str) -> bool:
    if text.lower() in ("1", "true", "yes", "on"):
        return True
    if text.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in text.split(",") if t.strip())


def parse_args(argv: list[str] | None = None) -> tuple[DQNConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--eval-only",
        metavar="RUN_DIR",
        help="load RUN_DIR's checkpoint and play --eval-games games (with --seed) vs "
        "random_legal instead of training",
    )
    for f in dataclasses.fields(DQNConfig):
        default = f.default
        if isinstance(default, bool):
            parse: Callable[[str], Any] = _parse_bool
        elif isinstance(default, tuple):
            parse = _parse_ints
        elif default is None:
            parse = str
        else:
            parse = type(default)
        shown = ",".join(map(str, default)) if isinstance(default, tuple) else default
        p.add_argument(
            "--" + f.name.replace("_", "-"), type=parse, default=default, help=f"(default: {shown})"
        )
    args = p.parse_args(argv)
    cfg = DQNConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(DQNConfig)})
    return cfg, args


def main(argv: list[str] | None = None) -> None:
    cfg, args = parse_args(argv)
    if args.eval_only:
        t = time.perf_counter()
        r = evaluate_run(args.eval_only, cfg.eval_games, cfg.seed)
        seconds = time.perf_counter() - t
        print(f"{args.eval_only} vs random_legal: {format_eval(r)} ({seconds:.0f}s)")
        print(json.dumps(r._asdict()))
        return
    train(cfg)


if __name__ == "__main__":
    main()
