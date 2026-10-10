"""Self-play Rainbow DQN baseline for the 1v1 duel (single file, cleanRL / PureJaxRL style).

Rainbow (Hessel et al., 2018, https://arxiv.org/abs/1710.02298) combines six
extensions of DQN. All six are here, and every one but the distributional head
can be switched off from the command line for ablations:

* **Distributional (C51) value head.** The network predicts, for each move, a
  categorical distribution over returns on ``num_atoms`` fixed atoms in
  ``[v_min, v_max]``. A snake's return in this game always lies in ``[-1, 1]``
  (-1 when it dies, +1 when it wins, 0 otherwise, every reward is final), so
  the default support is exactly that range and nothing is ever clipped.
* **Double Q-learning** (``--double``): the online network picks the bootstrap
  move, the target network's distribution for it is the target.
* **Dueling heads** (``--dueling``): a value stream and a mean-centred
  advantage stream, per atom.
* **Multi-step returns** (``--n-step``): ``n``-step discounted returns,
  truncated at the end of a game.
* **Prioritized replay** (``--per-alpha``, ``--per-beta-*``): proportional
  prioritization on the per-transition KL loss, with importance-sampling
  weights annealed to 1. ``--per-alpha 0`` is uniform replay.
* **Noisy nets** (``--noisy``): factorized Gaussian noise on the dense layers
  replaces epsilon-greedy exploration (``--noisy false`` restores the epsilon
  schedule).

As in ``dqn.py``: one network plays both snakes (parameter-sharing self-play),
each from its own egocentric observation and only among its legal moves; the
replay buffer stores compact game states, not observations; everything between
two log lines is one jitted ``lax.scan`` (:func:`make_train_chunk`)::

    python baselines/rainbow.py                                   # default run
    python baselines/rainbow.py --n-step 1 --per-alpha 0          # ablate two components
    python baselines/rainbow.py --eval-only runs/rainbow-20261009-120000
    python baselines/rainbow.py --resume runs/rainbow-20261009-120000

**Multi-step transitions in self-play.** The last ``n_step`` one-step
transitions of every game slot are kept in a sliding window
(:class:`RunnerState` ``window``). Once the window is full, each iteration
turns its oldest step into one ``n``-step game transition
(:func:`nstep_transition`) and stores it:

* the window stops early at the end of a game, since a finished game is
  replaced by a new one in the same slot;
* each snake's return is its discounted reward sum over the kept steps (a snake
  gets no reward after its own elimination);
* the bootstrap state is the state the last kept step reached. A snake's
  discount is ``gamma ** steps``, or 0 when it is dead there or the rules ended
  the game. A game cut off by ``max_turns`` still bootstraps, as in ``dqn.py``.

The last ``n_step - 1`` steps of a run are never stored.

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
class RainbowConfig:
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
    lr: float = 2.5e-4
    adam_eps: float = 1.5e-4  # the Rainbow paper's (Hessel et al., 2018, table 1)
    max_grad_norm: float = 10.0
    gamma: float = 0.99
    target_update_period: int = 250  # gradient updates between target updates
    tau: float = 1.0  # target <- tau * online + (1 - tau) * target; 1.0 is a hard copy
    # Rainbow components.
    n_step: int = 3  # 1 is one-step TD
    num_atoms: int = 51
    v_min: float = -1.0  # returns lie in [-1, 1] in this game
    v_max: float = 1.0
    double: bool = True
    dueling: bool = True
    noisy: bool = True  # noisy dense layers instead of epsilon-greedy
    noisy_sigma0: float = 0.5  # initial noise scale: sigma = sigma0 / sqrt(fan_in)
    per_alpha: float = 0.5  # priority exponent; 0 is uniform replay
    per_beta_start: float = 0.4  # importance-sampling exponent, annealed linearly
    per_beta_end: float = 1.0  # ... reached at the end of training
    per_eps: float = 1e-6  # added to every priority, so every transition can be replayed
    # Epsilon-greedy, used only with --noisy false: linear decay over the first
    # eps_decay_fraction of training.
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_fraction: float = 0.3
    # Network: 3x3 convs (ReLU), a dense hidden layer (ReLU), then the
    # distributional head (dueling or not); dense layers are noisy with --noisy.
    conv_channels: tuple[int, ...] = (32, 64, 64)
    conv_strides: tuple[int, ...] = (2, 2, 1)
    hidden: int = 256
    # Logging, evaluation and checkpoints.
    seed: int = 0
    log_every: int = 16_000  # env steps per jitted chunk (one log line)
    eval_every: int = 256_000  # env steps between evaluations (and checkpoints)
    eval_games: int = 1000  # greedy Rainbow (noise off) vs random_legal
    checkpoint_every: int = 128_000  # env steps between full checkpoints (for --resume)
    run_dir: str | None = None  # default: runs/rainbow-<timestamp>

    def __post_init__(self) -> None:
        if len(self.conv_channels) != len(self.conv_strides):
            raise ValueError("conv_channels and conv_strides must have the same length")
        if self.buffer_capacity < self.num_envs:
            raise ValueError("buffer_capacity must be at least num_envs")
        if self.learning_starts > self.buffer_capacity // self.num_envs * self.num_envs:
            # The buffer never fills that far, so learning would never start.
            raise ValueError("learning_starts must not exceed the (rounded) buffer_capacity")
        positive = (
            "max_turns", "num_envs", "total_env_steps", "batch_size", "updates_per_step",
            "target_update_period", "log_every", "eval_every", "checkpoint_every", "n_step",
        )  # fmt: skip
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.num_atoms < 2:
            raise ValueError("num_atoms must be >= 2")
        if not (self.v_min <= -1.0 and self.v_max >= 1.0):
            # Returns lie in [-1, 1]; a narrower support would clip every win or loss.
            raise ValueError("the support [v_min, v_max] must cover the returns [-1, 1]")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.per_alpha < 0.0:
            raise ValueError("per_alpha must be >= 0")
        if not (0.0 <= self.per_beta_start <= 1.0 and 0.0 <= self.per_beta_end <= 1.0):
            raise ValueError("per_beta_start and per_beta_end must be in [0, 1]")
        if self.per_eps <= 0.0:
            raise ValueError("per_eps must be > 0")

    @property
    def game(self) -> GameConfig:
        return GameConfig(max_turns=self.max_turns)


def support(cfg: RainbowConfig) -> jax.Array:
    """The return atoms ``f32[num_atoms]``, evenly spaced from ``v_min`` to ``v_max``."""
    return jnp.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms)


# --- Network -----------------------------------------------------------------------


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

    The forward pass is XLA's convolution, the backward pass differentiates the
    equivalent im2col form: on XLA:CPU, convolution gradients inside a
    ``lax.scan`` are many times slower than matmuls (see ``dqn.py``).
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


def _scaled_noise(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """``f(e) = sign(e) * sqrt(|e|)`` of standard normal ``e``, as in factorized NoisyNets."""
    e = jax.random.normal(key, shape)
    return jnp.sign(e) * jnp.sqrt(jnp.abs(e))


def dense(p: dict[str, jax.Array], x: jax.Array, noise: jax.Array | None, per_sample: bool):
    """``x @ W + b`` for ``x: [B, in]``, with factorized Gaussian noise if ``p`` is noisy.

    A noisy layer holds means ``w``, ``b`` and scales ``w_sigma``, ``b_sigma``; its
    weights are ``w + w_sigma * outer(f(e_in), f(e_out))`` and its bias
    ``b + b_sigma * f(e_out)`` (Fortunato et al., 2018). The noisy term is computed
    as ``f(e_out) * ((x * f(e_in)) @ w_sigma + b_sigma)``, so drawing a separate
    noise sample per row (``per_sample``) costs no more than one for the batch.

    ``noise=None`` (or a layer without scales) uses the means only.
    """
    y = x @ p["w"] + p["b"]
    if noise is None or "w_sigma" not in p:
        return y
    k_in, k_out = jax.random.split(noise)
    lead = x.shape[:1] if per_sample else ()
    e_in = _scaled_noise(k_in, (*lead, x.shape[-1]))
    e_out = _scaled_noise(k_out, (*lead, y.shape[-1]))
    return y + e_out * ((x * e_in) @ p["w_sigma"] + p["b_sigma"])


def init_network(key: jax.Array, cfg: RainbowConfig) -> Params:
    """Parameters for :func:`network`.

    Convolutions get He-normal weights and zero biases. Dense layers get the
    NoisyNet initialization when ``cfg.noisy`` (means uniform in
    ``+-1/sqrt(fan_in)``, scales ``sigma0/sqrt(fan_in)``), else He-normal weights
    as in ``dqn.py``.
    """
    rows, cols, cin = obs_shape(cfg.game)
    keys = iter(jax.random.split(key, len(cfg.conv_channels) + 3))

    def conv(shape: tuple[int, ...], fan_in: int) -> dict[str, jax.Array]:
        w = jax.random.normal(next(keys), shape) * math.sqrt(2.0 / fan_in)
        return {"w": w, "b": jnp.zeros(shape[-1])}

    def linear(fan_in: int, fan_out: int, gain: float) -> dict[str, jax.Array]:
        if not cfg.noisy:
            w = jax.random.normal(next(keys), (fan_in, fan_out)) * math.sqrt(gain / fan_in)
            return {"w": w, "b": jnp.zeros(fan_out)}
        k_w, k_b = jax.random.split(next(keys))
        bound = 1.0 / math.sqrt(fan_in)
        sigma = cfg.noisy_sigma0 / math.sqrt(fan_in)
        return {
            "w": jax.random.uniform(k_w, (fan_in, fan_out), minval=-bound, maxval=bound),
            "b": jax.random.uniform(k_b, (fan_out,), minval=-bound, maxval=bound),
            "w_sigma": jnp.full((fan_in, fan_out), sigma),
            "b_sigma": jnp.full((fan_out,), sigma),
        }

    params: Params = {}
    for i, (cout, stride) in enumerate(zip(cfg.conv_channels, cfg.conv_strides, strict=True)):
        params[f"conv{i}"] = conv((3, 3, cin, cout), 9 * cin)
        rows, cols, cin = -(-rows // stride), -(-cols // stride), cout  # "SAME" padding
    params["hidden"] = linear(rows * cols * cin, cfg.hidden, gain=2.0)
    if cfg.dueling:
        params["value"] = linear(cfg.hidden, cfg.num_atoms, gain=1.0)
        params["advantage"] = linear(cfg.hidden, NUM_ACTIONS * cfg.num_atoms, gain=1.0)
    else:
        params["logits"] = linear(cfg.hidden, NUM_ACTIONS * cfg.num_atoms, gain=1.0)
    return params


def network(
    params: Params,
    obs: jax.Array,
    cfg: RainbowConfig,
    noise: jax.Array | None = None,
    per_sample: bool = False,
) -> jax.Array:
    """Return-distribution logits ``f32[..., 4, num_atoms]`` for ``obs: f32[..., rows, cols, C]``.

    Args:
      noise: a PRNG key for the noisy layers, or None for their means (the
        deterministic network used for evaluation and play).
      per_sample: draw separate noise for every observation instead of one
        sample shared by the batch.
    """
    lead = obs.shape[:-3]
    x = obs.reshape(-1, *obs.shape[-3:])
    for i, stride in enumerate(cfg.conv_strides):
        p = params[f"conv{i}"]
        x = jax.nn.relu(conv3x3(x, p["w"], stride) + p["b"])
    x = x.reshape(x.shape[0], -1)
    keys = [None] * 3 if noise is None else list(jax.random.split(noise, 3))
    x = jax.nn.relu(dense(params["hidden"], x, keys[0], per_sample))
    shape = (x.shape[0], NUM_ACTIONS, cfg.num_atoms)
    if cfg.dueling:
        v = dense(params["value"], x, keys[1], per_sample)[:, None, :]
        a = dense(params["advantage"], x, keys[2], per_sample).reshape(shape)
        logits = v + a - a.mean(axis=1, keepdims=True)
    else:
        logits = dense(params["logits"], x, keys[1], per_sample).reshape(shape)
    return logits.reshape(*lead, NUM_ACTIONS, cfg.num_atoms)


def q_values(logits: jax.Array, cfg: RainbowConfig) -> jax.Array:
    """Expected returns ``f32[..., 4]`` of return-distribution logits ``[..., 4, num_atoms]``."""
    return jnp.sum(jax.nn.softmax(logits, axis=-1) * support(cfg), axis=-1)


def epsilon_greedy(key: jax.Array, q: jax.Array, mask: jax.Array, eps: jax.Array) -> jax.Array:
    """Per agent: a uniformly random legal action with probability ``eps``, else the legal argmax.

    ``q`` and ``mask`` are ``[..., 4]``; mask rows are never all-False.
    """
    k_explore, k_random = jax.random.split(key)
    greedy = jnp.argmax(jnp.where(mask, q, -jnp.inf), axis=-1)
    random = jax.random.categorical(k_random, jnp.where(mask, 0.0, -jnp.inf), axis=-1)
    explore = jax.random.uniform(k_explore, mask.shape[:-1]) < eps
    return jnp.where(explore, random, greedy).astype(jnp.int32)


def _linear_schedule(start: float, end: float, steps: float, env_steps: jax.Array) -> jax.Array:
    frac = jnp.clip(env_steps / max(steps, 1.0), 0.0, 1.0)
    return start + frac * (end - start)


def epsilon(cfg: RainbowConfig, env_steps: jax.Array) -> jax.Array:
    """The exploration rate: 0 with noisy nets, else the linear epsilon schedule."""
    if cfg.noisy:
        return jnp.zeros((), jnp.float32)
    decay = cfg.eps_decay_fraction * cfg.total_env_steps
    return _linear_schedule(cfg.eps_start, cfg.eps_end, decay, env_steps)


def per_beta(cfg: RainbowConfig, env_steps: jax.Array) -> jax.Array:
    """Importance-sampling exponent, linear from ``per_beta_start`` to ``per_beta_end``."""
    return _linear_schedule(cfg.per_beta_start, cfg.per_beta_end, cfg.total_env_steps, env_steps)


# --- Multi-step transitions ----------------------------------------------------------


class Step(NamedTuple):
    """One-step game transitions of the ``num_envs`` slots (leaves ``[num_envs, ...]``)."""

    state: State  # s_t; never done, since finished games are reset right away
    actions: jax.Array  # int32[E, N]
    rewards: jax.Array  # float32[E, N]
    next_state: State  # s_{t+1} as reached, before any autoreset
    truncated: jax.Array  # bool[E] the game was cut off by max_turns at t+1


class Transition(NamedTuple):
    """One stored ``n``-step game transition (all snakes). Batched leaves have a leading axis."""

    state: State  # s_t (never done)
    actions: jax.Array  # int32[N] the moves taken at s_t
    returns: jax.Array  # float32[N] discounted reward sum over the kept steps
    next_state: State  # bootstrap state: what the last kept step reached
    discount: jax.Array  # float32[N] gamma ** steps, or 0 for no bootstrap


def nstep_transition(window: Step, gamma: float) -> Transition:
    """``n``-step transitions starting at the oldest step of ``window`` (leaves ``[n, E, ...]``).

    Steps are kept up to and including the first one that ends the game (the
    slot holds a new game after it). See the module docstring.
    """
    n, num_envs = window.truncated.shape
    done = window.next_state.done  # [n, E]
    # kept[k]: no step before k ended the oldest step's game.
    before = jnp.concatenate([jnp.ones((1, num_envs), bool), ~done[:-1]], axis=0)
    kept = jnp.cumprod(before, axis=0).astype(bool)
    steps = kept.sum(axis=0)  # [E] in 1..n
    powers = gamma ** jnp.arange(n, dtype=jnp.float32)
    returns = jnp.sum(jnp.where(kept[..., None], powers[:, None, None] * window.rewards, 0.0), 0)

    last, env = steps - 1, jnp.arange(num_envs)
    next_state = jax.tree.map(lambda x: x[last, env], window.next_state)
    truncated = window.truncated[last, env]
    ended = next_state.done & ~truncated  # the rules ended the game: rewards are final
    bootstrap = next_state.alive & ~ended[:, None]
    discount = jnp.where(bootstrap, (gamma**steps)[:, None], 0.0).astype(jnp.float32)
    first = jax.tree.map(lambda x: x[0], window)
    return Transition(first.state, first.actions, returns, next_state, discount)


def window_push(window: Step, step: Step) -> Step:
    """Drop the oldest step of ``window`` and append ``step`` as the newest."""
    return jax.tree.map(lambda w, s: jnp.concatenate([w[1:], s[None]], axis=0), window, step)


# --- Prioritized replay buffer --------------------------------------------------------


class Buffer(NamedTuple):
    """Circular buffer with a replay priority per item (leaves ``[capacity, ...]``)."""

    data: Any
    priority: jax.Array  # f32[capacity] raw priorities (|loss| + per_eps), 0 when empty
    max_priority: jax.Array  # f32[] the largest priority so far; new items get it
    ptr: jax.Array  # int32[] next write position
    size: jax.Array  # int32[] number of filled slots


def buffer_init(example: Any, capacity: int) -> Buffer:
    """An empty buffer for items shaped like ``example`` (one item, no leading axis)."""
    data = jax.tree.map(lambda x: jnp.zeros((capacity, *x.shape), x.dtype), example)
    return Buffer(
        data,
        jnp.zeros((capacity,), jnp.float32),
        jnp.ones((), jnp.float32),
        jnp.zeros((), jnp.int32),
        jnp.zeros((), jnp.int32),
    )


def buffer_add(buf: Buffer, items: Any) -> Buffer:
    """Write a batch of ``B`` items at ``ptr`` with the maximum priority so far.

    The capacity must be a multiple of ``B`` (always adding ``B`` items), so a
    batch never straddles the end and is one contiguous in-place update.
    """
    batch = jax.tree.leaves(items)[0].shape[0]
    capacity = buf.priority.shape[0]
    if capacity % batch:
        raise ValueError(f"capacity {capacity} is not a multiple of the batch size {batch}")
    data = jax.tree.map(
        lambda d, x: jax.lax.dynamic_update_slice_in_dim(d, x.astype(d.dtype), buf.ptr, axis=0),
        buf.data,
        items,
    )
    priority = jax.lax.dynamic_update_slice_in_dim(
        buf.priority, jnp.full((batch,), buf.max_priority), buf.ptr, axis=0
    )
    return Buffer(
        data,
        priority,
        buf.max_priority,
        (buf.ptr + batch) % capacity,
        jnp.minimum(buf.size + batch, capacity),
    )


def buffer_sample(
    priority: jax.Array, size: jax.Array, key: jax.Array, batch_size: int, alpha: float, beta
) -> tuple[jax.Array, jax.Array]:
    """``(indices, weights)`` of ``batch_size`` items drawn with probability ``p ** alpha / sum``.

    Stratified: one draw from each of ``batch_size`` equal slices of the
    cumulative priority mass (a cumulative sum and a binary search; for this
    buffer size that is cheaper than maintaining a sum tree). Only filled slots
    can be drawn. ``weights`` are the importance-sampling corrections
    ``(size * P(i)) ** -beta``, divided by their maximum over the batch. ``P(i)``
    is read off the same float32 cumulative sum the draw used, so the weights
    match the actual draw probabilities even where rounding over ~1e5 items
    shifts a tiny priority's share.
    """
    filled = jnp.arange(priority.shape[0]) < size
    scaled = jnp.where(filled, priority**alpha, 0.0)
    cdf = jnp.cumsum(scaled)
    total = cdf[-1]
    u = (jnp.arange(batch_size) + jax.random.uniform(key, (batch_size,))) / batch_size * total
    idx = jnp.minimum(jnp.searchsorted(cdf, u, side="right"), jnp.maximum(size - 1, 0))
    below = jnp.where(idx > 0, cdf[jnp.maximum(idx - 1, 0)], 0.0)
    mass = cdf[idx] - below
    prob = jnp.where(mass > 0, mass, scaled[idx]) / total  # 0 only via the clip above
    weights = (size * prob) ** -beta
    return idx.astype(jnp.int32), weights / jnp.max(weights)


def update_priorities(
    priority: jax.Array, max_priority: jax.Array, idx: jax.Array, new: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Set the priorities of the sampled ``idx`` (a repeated index keeps one of its values)."""
    return priority.at[idx].set(new), jnp.maximum(max_priority, jnp.max(new))


# --- Distributional targets and loss --------------------------------------------------


def project(returns: jax.Array, discount: jax.Array, probs: jax.Array, atoms: jax.Array):
    """Project the distribution of ``returns + discount * Z`` onto ``atoms`` (C51's Phi).

    ``Z`` has probabilities ``probs: [..., A]`` on ``atoms``; ``returns`` and
    ``discount`` are ``[...]``. Each shifted atom, clipped to the support, splits
    its mass between its two neighbouring atoms in proportion to closeness: the
    triangular kernel ``max(0, 1 - |Tz_j - z_i| / dz)``, which is exact for evenly
    spaced atoms and keeps the total mass.
    """
    dz = atoms[1] - atoms[0]
    tz = jnp.clip(returns[..., None] + discount[..., None] * atoms, atoms[0], atoms[-1])
    kernel = jnp.clip(1.0 - jnp.abs(tz[..., :, None] - atoms) / dz, 0.0, 1.0)  # [..., A_j, A_i]
    return jnp.sum(probs[..., :, None] * kernel, axis=-2)


def target_distribution(
    tr: Transition,
    next_logits_select: jax.Array,
    next_logits_target: jax.Array,
    next_mask: jax.Array,
    cfg: RainbowConfig,
) -> jax.Array:
    """Projected target distributions ``f32[..., N, num_atoms]`` for ``n``-step transitions.

    The bootstrap move ``a*`` is the legal argmax of the expected return under
    ``next_logits_select`` (the online network for double Q-learning, else the
    target network); the target network's distribution for ``a*`` is shifted by
    the ``n``-step return and discounted (a point mass at the return when the
    discount is 0).
    """
    q_select = q_values(next_logits_select, cfg)
    a_star = jnp.argmax(jnp.where(next_mask, q_select, -jnp.inf), axis=-1)
    probs = jax.nn.softmax(next_logits_target, axis=-1)
    p_next = jnp.take_along_axis(probs, a_star[..., None, None], axis=-2)[..., 0, :]
    return project(tr.returns, tr.discount, p_next, support(cfg))


def valid_agents(state: State) -> jax.Array:
    """``bool[..., N]``: the agent was alive in a running game at ``t``, so it acted."""
    return state.alive & ~state.done[..., None]


class LossOut(NamedTuple):
    grads: Params
    priorities: jax.Array  # f32[B] mean KL loss over the valid agents of each transition
    mean_q: jax.Array  # f32[] mean expected return of the taken moves (valid agents)


def make_loss_fn(
    env: BattlesnakeEnv, cfg: RainbowConfig
) -> Callable[[Params, Params, Transition, jax.Array, jax.Array], tuple[jax.Array, LossOut]]:
    """``loss(params, target_params, batch, weights, key) -> (loss, LossOut)``.

    The loss is the importance-weighted KL divergence from the projected target
    to the online distribution of the taken move, averaged over valid agents.
    Its gradient is the cross-entropy's; unlike the cross-entropy it is 0 for a
    perfect prediction, so it also serves as the replay priority (as in the
    paper) without favouring transitions whose targets are spread out. Each
    network evaluation draws its own noise (one sample per batch).
    """
    observe = jax.vmap(env.observe)  # [B] states -> [B, N, rows, cols, C]
    masks = jax.vmap(env.action_mask)  # [B] states -> [B, N, 4]
    atoms = support(cfg)

    def noise(key: jax.Array) -> jax.Array | None:
        return key if cfg.noisy else None

    def kl_loss(params, obs, actions, target, valid, weights, key):
        logits = network(params, obs, cfg, noise(key))  # [B, N, 4, A]
        taken = jnp.take_along_axis(logits, actions[..., None, None], axis=-2)[..., 0, :]
        log_p = jax.nn.log_softmax(taken, axis=-1)
        kl = jnp.sum(jax.scipy.special.xlogy(target, target) - target * log_p, axis=-1)  # [B, N]
        v = valid.astype(jnp.float32)
        count = jnp.maximum(v.sum(), 1.0)
        loss = jnp.sum(weights[:, None] * v * kl) / count
        per_item = jnp.sum(v * kl, axis=-1) / jnp.maximum(v.sum(axis=-1), 1.0)
        mean_q = jnp.sum(v * jnp.sum(jnp.exp(log_p) * atoms, axis=-1)) / count
        return loss, (jnp.maximum(per_item, 0.0), mean_q)  # KL >= 0 up to rounding

    def loss_and_grad(params, target_params, tr, weights, key):
        k_online, k_select, k_target = jax.random.split(key, 3)
        next_obs = observe(tr.next_state)
        next_target = network(target_params, next_obs, cfg, noise(k_target))
        # Double Q-learning: the online network (own noise) picks the bootstrap move.
        next_select = network(params, next_obs, cfg, noise(k_select)) if cfg.double else next_target
        target = target_distribution(tr, next_select, next_target, masks(tr.next_state), cfg)
        target = jax.lax.stop_gradient(target)
        (loss, (per_item, mean_q)), grads = jax.value_and_grad(kl_loss, has_aux=True)(
            params, observe(tr.state), tr.actions, target, valid_agents(tr.state), weights, k_online
        )
        return loss, LossOut(grads, jax.lax.stop_gradient(per_item), mean_q)

    return loss_and_grad


# --- Training ----------------------------------------------------------------------


class RunnerState(NamedTuple):
    """Everything the jitted training loop carries between chunks."""

    params: Params
    target_params: Params
    opt_state: Any
    env_states: State  # [num_envs] running games (never done)
    window: Step  # [n_step, num_envs] the latest one-step transitions, oldest first
    buffer: Buffer
    key: jax.Array
    env_steps: jax.Array  # int32[] env steps taken
    updates: jax.Array  # int32[] gradient updates taken


def make_envs(cfg: RainbowConfig) -> tuple[BattlesnakeEnv, BattlesnakeEnv]:
    """``(env, sim)``: ``env`` computes egocentric observations, ``sim`` steps without them."""
    return BattlesnakeEnv(cfg.game, obs="egocentric"), BattlesnakeEnv(cfg.game, obs=None)


def make_optimizer(cfg: RainbowConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=cfg.adam_eps)
    )


def init_runner(cfg: RainbowConfig, sim: BattlesnakeEnv, key: jax.Array) -> RunnerState:
    k_net, k_env, k_run = jax.random.split(key, 3)
    params = init_network(k_net, cfg)
    env_states = jax.vmap(sim.init_state)(jax.random.split(k_env, cfg.num_envs))
    e, n = cfg.num_envs, sim.num_agents
    step = Step(
        state=env_states,
        actions=jnp.zeros((e, n), jnp.int32),
        rewards=jnp.zeros((e, n), jnp.float32),
        next_state=env_states,
        truncated=jnp.zeros((e,), bool),
    )
    one = jax.tree.map(lambda x: x[0], env_states)
    example = Transition(
        state=one,
        actions=jnp.zeros((n,), jnp.int32),
        returns=jnp.zeros((n,), jnp.float32),
        next_state=one,
        discount=jnp.zeros((n,), jnp.float32),
    )
    capacity = cfg.buffer_capacity // cfg.num_envs * cfg.num_envs
    return RunnerState(
        params=params,
        # A copy: the runner is donated, and one buffer can't be donated twice.
        target_params=jax.tree.map(jnp.copy, params),
        opt_state=make_optimizer(cfg).init(params),
        env_states=env_states,
        # Placeholder steps; none is stored before n_step real ones have been pushed.
        window=jax.tree.map(lambda x: jnp.stack([x] * cfg.n_step), step),
        buffer=buffer_init(example, capacity),
        key=k_run,
        env_steps=jnp.zeros((), jnp.int32),
        updates=jnp.zeros((), jnp.int32),
    )


def make_train_chunk(
    cfg: RainbowConfig, env: BattlesnakeEnv, sim: BattlesnakeEnv, num_iters: int
) -> Callable[[RunnerState], tuple[RunnerState, dict[str, jax.Array]]]:
    """``train_chunk(runner) -> (runner, metrics)``: ``num_iters`` iterations in one ``lax.scan``.

    Each iteration acts in and steps all ``num_envs`` games, pushes the step into
    the ``n``-step window and stores the completed ``n``-step transitions, and
    once the buffer holds ``learning_starts`` transitions takes
    ``updates_per_step`` gradient steps. Metrics are summed over the chunk.
    """
    optimizer = make_optimizer(cfg)
    loss_and_grad = make_loss_fn(env, cfg)
    num_envs = cfg.num_envs
    start = max(cfg.learning_starts, 1)

    def gradient_step(data, beta, carry, key):
        params, target_params, opt_state, updates, priority, max_priority, size = carry
        k_sample, k_loss = jax.random.split(key)
        idx, weights = buffer_sample(priority, size, k_sample, cfg.batch_size, cfg.per_alpha, beta)
        batch = jax.tree.map(lambda d: d[idx], data)
        loss, out = loss_and_grad(params, target_params, batch, weights, k_loss)
        step, opt_state = optimizer.update(out.grads, opt_state, params)
        params = optax.apply_updates(params, step)
        priority, max_priority = update_priorities(
            priority, max_priority, idx, out.priorities + cfg.per_eps
        )
        updates = updates + 1
        sync = updates % cfg.target_update_period == 0
        target_params = jax.tree.map(
            lambda t, p: jnp.where(sync, optax.incremental_update(p, t, cfg.tau), t),
            target_params,
            params,
        )
        carry = (params, target_params, opt_state, updates, priority, max_priority, size)
        return carry, (loss, out.mean_q)

    def learn(data, beta, carry, key):
        keys = jax.random.split(key, cfg.updates_per_step)
        carry, (loss, mean_q) = jax.lax.scan(
            lambda c, k: gradient_step(data, beta, c, k), carry, keys
        )
        return carry, (loss.sum(), mean_q.sum(), jnp.int32(cfg.updates_per_step))

    def skip(data, beta, carry, key):
        return carry, (jnp.float32(0), jnp.float32(0), jnp.int32(0))

    def iteration(runner: RunnerState, _):
        key, k_noise, k_act, k_step, k_reset, k_learn = jax.random.split(runner.key, 6)
        states = runner.env_states

        # Act: every snake from its own egocentric view, one shared network; with
        # noisy nets each snake draws its own noise.
        obs = jax.vmap(env.observe)(states)
        logits = network(runner.params, obs, cfg, k_noise if cfg.noisy else None, per_sample=True)
        mask = jax.vmap(env.action_mask)(states)
        actions = epsilon_greedy(k_act, q_values(logits, cfg), mask, epsilon(cfg, runner.env_steps))

        # Step, slide the window, store the n-step transition of its oldest step
        # once it holds n real steps, then reset finished games. Until then the
        # write is not committed (ptr and size stay), so the next one overwrites it;
        # this avoids a lax.cond around the whole buffer.
        new, ts = jax.vmap(sim.step)(jax.random.split(k_step, num_envs), states, actions)
        window = window_push(runner.window, Step(states, actions, ts.reward, new, ts.truncated))
        ready = runner.env_steps // num_envs >= cfg.n_step - 1
        buffer = buffer_add(runner.buffer, nstep_transition(window, cfg.gamma))
        buffer = buffer._replace(
            ptr=jnp.where(ready, buffer.ptr, runner.buffer.ptr),
            size=jnp.where(ready, buffer.size, runner.buffer.size),
        )
        fresh = jax.vmap(sim.init_state)(jax.random.split(k_reset, num_envs))
        env_states = jax.vmap(_tree_where)(ts.done, fresh, new)

        beta = per_beta(cfg, runner.env_steps)
        carry = (
            runner.params, runner.target_params, runner.opt_state, runner.updates,
            buffer.priority, buffer.max_priority, buffer.size,
        )  # fmt: skip
        carry, (loss, mean_q, n_updates) = jax.lax.cond(
            buffer.size >= start, learn, skip, buffer.data, beta, carry, k_learn
        )
        params, target_params, opt_state, updates, priority, max_priority, _ = carry
        buffer = buffer._replace(priority=priority, max_priority=max_priority)

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
            window=window,
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
        metrics["beta"] = per_beta(cfg, runner.env_steps)
        metrics["max_priority"] = runner.buffer.max_priority
        metrics["noise_sigma"] = (
            jnp.mean(jnp.abs(runner.params["hidden"]["w_sigma"])) if cfg.noisy else jnp.float32(0)
        )
        metrics["env_steps"] = runner.env_steps
        metrics["total_updates"] = runner.updates
        metrics["buffer_size"] = runner.buffer.size
        return runner, metrics

    return train_chunk


def _tree_where(cond: jax.Array, a: Any, b: Any) -> Any:
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


# --- Evaluation and checkpoints ---------------------------------------------------


def q_function(params: Params, cfg: RainbowConfig) -> Callable[[jax.Array], jax.Array]:
    """``obs -> expected returns [..., 4]`` of the noise-free network, for greedy play."""

    def q_fn(obs: jax.Array) -> jax.Array:
        return q_values(network(params, obs, cfg), cfg)

    return q_fn


def evaluate_params(
    params: Params, cfg: RainbowConfig, env: BattlesnakeEnv, key: jax.Array, num_games: int
) -> MatchResult:
    """Greedy Rainbow (legal argmax of the expected return, noise off) vs ``random_legal``.

    The network parameters are baked into the compiled match (see
    ``slinky.evaluate``), so every call compiles once more (a few seconds).
    """
    # A private copy: the training loop donates (deletes) the runner's arrays.
    params = jax.tree.map(jnp.copy, params)
    return play_match(
        env,
        greedy_from_q(q_function(params, cfg)),
        random_legal(env),
        key,
        num_games,
        max_turns=cfg.max_turns,
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


def save_config(run_dir: str, cfg: RainbowConfig) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"algorithm": "rainbow", **dataclasses.asdict(cfg)}, f, indent=2)


def load_config(run_dir: str) -> RainbowConfig:
    with open(os.path.join(run_dir, "config.json")) as f:
        d = json.load(f)
    if d.get("algorithm") != "rainbow":
        raise ValueError(f"{run_dir} is not a Rainbow run (config.json has no algorithm=rainbow)")
    fields = {f.name: f for f in dataclasses.fields(RainbowConfig)}
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in fields}
    return RainbowConfig(**kwargs)


def load_params(run_dir: str, cfg: RainbowConfig | None = None) -> Params:
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


def save_runner(run_dir: str, runner: RunnerState, agent_transitions: int) -> None:
    """Write the whole training state (replay buffer included) to ``runner.npz`` atomically."""
    runner = runner._replace(key=jax.random.key_data(runner.key))
    leaves = {f"leaf{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(runner))}
    tmp = os.path.join(run_dir, "runner.tmp.npz")
    np.savez(tmp, agent_transitions=agent_transitions, **leaves)
    os.replace(tmp, os.path.join(run_dir, "runner.npz"))


def load_runner(run_dir: str, template: RunnerState) -> tuple[RunnerState, int]:
    """``(runner, agent_transitions)`` from ``runner.npz``, structured like ``template``."""
    template = template._replace(key=jax.random.key_data(template.key))
    refs, treedef = jax.tree.flatten(template)
    with np.load(os.path.join(run_dir, "runner.npz")) as f:
        leaves = [f[f"leaf{i}"] for i in range(len(refs))]
        agent_transitions = int(f["agent_transitions"])
    for leaf, ref in zip(leaves, refs, strict=True):
        if leaf.shape != ref.shape or leaf.dtype != ref.dtype:
            raise ValueError(f"runner.npz does not match the config: {leaf.shape} vs {ref.shape}")
    runner = jax.tree.unflatten(treedef, [jnp.asarray(x) for x in leaves])
    return runner._replace(key=jax.random.wrap_key_data(runner.key)), agent_transitions


def _truncate_metrics(path: str, env_steps: int) -> None:
    """Drop records logged after the checkpoint we resume from (they will be redone)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        records = [line for line in f if json.loads(line)["env_steps"] <= env_steps]
    with open(path, "w") as f:
        f.writelines(records)


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
        "beta": float(m["beta"]),
        "max_priority": float(m["max_priority"]),
        "noise_sigma": float(m["noise_sigma"]),
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


def format_record(r: dict[str, Any], total_steps: int, agent_total: int, noisy: bool) -> str:
    deaths = r["deaths"]
    n_dead = sum(deaths.values())
    shown = [k for k in DEATH_CAUSES if k != "hazard" or deaths[k]]
    loss = f"{r['loss']:.4f}" if r["loss"] is not None else "-"
    q = f"{r['mean_q']:+.3f}" if r["mean_q"] is not None else "-"
    length = f"{r['mean_game_len']:.0f}" if r["mean_game_len"] is not None else "-"
    explore = f"sigma {r['noise_sigma']:.4f}" if noisy else f"eps {r['eps']:.2f}"
    return (
        f"steps {r['env_steps']:>9,} ({100 * r['env_steps'] / total_steps:3.0f}%) "
        f"agent-tr {agent_total:>10,} | {explore} beta {r['beta']:.2f} | "
        f"loss {loss} Q {q} | "
        f"games {r['games']:,} len {length} draw {_pct(r['draws'], r['games'])} | deaths "
        + " ".join(f"{k} {_pct(deaths[k], n_dead)}" for k in shown)
        + f" | {r['updates_per_s']:.0f} upd/s {r['env_steps_per_s']:,.0f} steps/s"
    )


# --- Main loop -----------------------------------------------------------------------


def train(cfg: RainbowConfig, resume: bool = False) -> tuple[str, Params]:
    """Run a full training; returns ``(run_dir, final params)``.

    With ``resume``, continue from ``run_dir/runner.npz`` (written every
    ``checkpoint_every`` env steps) if it exists. Resuming is exact: the run
    continues as if it had never stopped.
    """
    run_dir = cfg.run_dir or os.path.join("runs", time.strftime("rainbow-%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    cfg = dataclasses.replace(cfg, run_dir=run_dir)
    save_config(run_dir, cfg)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    env, sim = make_envs(cfg)
    k_eval, k_init = jax.random.split(jax.random.key(cfg.seed))
    runner = jax.jit(init_runner, static_argnums=(0, 1))(cfg, sim, k_init)
    agent_total = 0
    if resume and os.path.exists(os.path.join(run_dir, "runner.npz")):
        runner, agent_total = load_runner(run_dir, runner)
        _truncate_metrics(metrics_path, int(runner.env_steps))
        print(f"resuming from env step {int(runner.env_steps):,}")
    elif os.path.exists(metrics_path):
        os.remove(metrics_path)  # a fresh run in an old directory

    chunk_iters = max(cfg.log_every // cfg.num_envs, 1)
    chunk_steps = chunk_iters * cfg.num_envs
    num_chunks = -(-cfg.total_env_steps // chunk_steps)
    total_steps = num_chunks * chunk_steps
    stored = total_steps - (cfg.n_step - 1) * cfg.num_envs  # the last n-1 steps are not stored
    if cfg.learning_starts > stored:
        raise ValueError(
            f"learning_starts ({cfg.learning_starts:,}) exceeds the {stored:,} transitions this "
            f"run stores ({total_steps:,} env steps less the last n_step - 1 steps): it would "
            f"never learn"
        )
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

    start = time.perf_counter()
    with open(metrics_path, "a") as log:

        def write(record: dict[str, Any]) -> None:
            log.write(json.dumps(record) + "\n")
            log.flush()

        def crossed(every: int, steps: int) -> bool:
            return steps // every > (steps - chunk_steps) // every

        for chunk in range(int(runner.env_steps) // chunk_steps, num_chunks):
            t = time.perf_counter()
            runner, m = train_chunk(runner)
            m = jax.device_get(m)
            seconds = time.perf_counter() - t
            record = chunk_record(m, seconds, chunk_steps)
            agent_total += record["agent_transitions"]
            record["agent_transitions_total"] = agent_total
            record["elapsed"] = time.perf_counter() - start
            write(record)
            print(format_record(record, total_steps, agent_total, cfg.noisy), flush=True)

            steps = record["env_steps"]
            last = chunk == num_chunks - 1
            if last or crossed(cfg.eval_every, steps):
                save_checkpoint(run_dir, runner.params)
                if cfg.eval_games > 0:
                    t = time.perf_counter()
                    k = jax.random.fold_in(k_eval, steps)
                    r = evaluate_params(runner.params, cfg, env, k, cfg.eval_games)
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
            # After the eval, so an interrupted eval is redone on resume.
            if last or crossed(cfg.checkpoint_every, steps):
                save_runner(run_dir, runner, agent_total)
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


def parse_args(argv: list[str] | None = None) -> tuple[RainbowConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--eval-only",
        metavar="RUN_DIR",
        help="load RUN_DIR's checkpoint and play --eval-games games (with --seed) vs "
        "random_legal instead of training",
    )
    p.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help="continue an interrupted run from its last full checkpoint, with its saved "
        "config (other flags are ignored)",
    )
    for f in dataclasses.fields(RainbowConfig):
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
    cfg = RainbowConfig(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(RainbowConfig)}
    )
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
    if args.resume:
        train(dataclasses.replace(load_config(args.resume), run_dir=args.resume), resume=True)
        return
    train(cfg)


if __name__ == "__main__":
    main()
