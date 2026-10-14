"""Self-play Rainbow DQN for the four leaderboard modes, one network told its mode.

``baselines/rainbow.py`` (all six Rainbow components: C51 head on ``[-1, 1]``, double
Q-learning, dueling heads, ``n``-step returns, proportional prioritized replay and noisy
dense layers; the same hyperparameters) trained on every mode of :mod:`slinky.modes` at
once with the multi-mode design of ``baselines/ppo_modes.py`` (shared code in
:mod:`slinky.unified`): one group of ``num_envs`` games per mode, one network call per
turn on every snake of every group, the mode features (added to the noisy hidden layer's
pre-activation through the zero-initialised ``feat/w``) and hidden constrictor food, and
the placement reward. A snake's placement return lies in ``[-1, 1]`` in every mode (its
gains and its loss each sum to at most 1 in magnitude, so any discounted partial sum
does too), so the support needs no change. Changes from ``rainbow.py``:

* **Per mode:** an ``n``-step window, a prioritized buffer of ``buffer_capacity``
  game transitions, and ``batch_size`` transitions sampled from it at every update,
  with importance weights normalised within the mode's batch. The loss is the mean
  over the modes of each mode's weighted KL loss (``mode_weights``).
* **Schedules** in env steps per mode are fixed (``per_beta_steps``; epsilon, used
  only without noisy nets, over ``eps_decay_steps``), not fractions of the run, whose
  length the plateau decides.
* **The learning rate** is constant (``rainbow.py``'s 2.5e-4) with the campaign's
  optional cooldown (:func:`slinky.unified.lr_factor`).

Played as an agent (``rainbow_modes``) the network is noise-free and greedy (the legal
argmax of the expected return); ``rainbow_modes-sample`` plays epsilon-greedy with
``EVAL_EPSILON``.

Commands::

    python baselines/rainbow_modes.py --size M --run-dir runs/rainbow-modes-M
    python baselines/rainbow_modes.py --resume runs/rainbow-modes-M --cooldown 2097152
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from slinky import unified
from slinky.evaluate import Policy
from slinky.types import NUM_ACTIONS, GameConfig, State, TimeStep
from slinky.unified import Group, Params

ALGORITHM = "rainbow_modes"
# The exploration of a sampled (``rainbow_modes-sample``) agent.
EVAL_EPSILON = 0.05


@dataclasses.dataclass(frozen=True)
class Config(unified.ModesSettings):
    """All hyperparameters (and :class:`slinky.unified.ModesSettings`'). Every field can be
    overridden on the command line. Steps are env steps per mode; buffers hold game
    transitions, each worth one agent-transition per snake."""

    lr: float = 2.5e-4
    mode_weights: tuple[float, ...] = ()  # loss weight of each game_modes entry; () = equal
    # Collection and replay, per mode (rainbow.py's).
    num_envs: int = 32
    buffer_capacity: int = 100_000  # rounded down to a multiple of num_envs
    learning_starts: int = 10_000
    batch_size: int = 128
    updates_per_step: int = 1
    # Optimization.
    adam_eps: float = 1.5e-4
    max_grad_norm: float = 10.0
    gamma: float = 0.99
    target_update_period: int = 250
    tau: float = 1.0
    # Rainbow components.
    n_step: int = 3
    num_atoms: int = 51
    v_min: float = -1.0
    v_max: float = 1.0
    double: bool = True
    dueling: bool = True
    noisy: bool = True
    noisy_sigma0: float = 0.5
    per_alpha: float = 0.5
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 5_242_880  # env steps per mode to reach per_beta_end
    per_eps: float = 1e-6
    # Epsilon-greedy, used only with --noisy false.
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 1_572_864
    log_every: int = 32_768  # env steps per mode per jitted chunk (one log line)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode_weights and len(self.mode_weights) != len(self.game_modes):
            raise ValueError("mode_weights needs one weight per game mode")
        if any(w < 0 for w in self.mode_weights) or (
            self.mode_weights and sum(self.mode_weights) <= 0
        ):
            raise ValueError("mode_weights must be >= 0 with a positive sum")
        if self.buffer_capacity < self.num_envs:
            raise ValueError("buffer_capacity must be at least num_envs")
        if self.learning_starts > self.capacity:
            raise ValueError("learning_starts must not exceed the (rounded) buffer_capacity")
        positive = (
            "num_envs", "batch_size", "updates_per_step", "target_update_period", "n_step",
            "per_beta_steps", "eps_decay_steps", "log_every",
        )  # fmt: skip
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.num_atoms < 2:
            raise ValueError("num_atoms must be >= 2")
        if not (self.v_min <= -1.0 and self.v_max >= 1.0):
            raise ValueError("the support [v_min, v_max] must cover the returns [-1, 1]")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.per_alpha < 0.0 or self.per_eps <= 0.0:
            raise ValueError("per_alpha must be >= 0 and per_eps > 0")
        if not (0.0 <= self.per_beta_start <= 1.0 and 0.0 <= self.per_beta_end <= 1.0):
            raise ValueError("per_beta_start and per_beta_end must be in [0, 1]")

    @property
    def capacity(self) -> int:
        return self.buffer_capacity // self.num_envs * self.num_envs

    @property
    def loss_weights(self) -> tuple[float, ...]:
        w = self.mode_weights or (1.0,) * len(self.game_modes)
        return tuple(x / sum(w) for x in w)

    @property
    def chunk_iters(self) -> int:
        return max(self.log_every // self.num_envs, 1)


def support(cfg: Config) -> jax.Array:
    """The return atoms ``f32[num_atoms]``, evenly spaced from ``v_min`` to ``v_max``."""
    return jnp.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms)


# --- Network: the convolutions, a noisy hidden layer and noisy C51 heads ---------------


def _scaled_noise(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """``f(e) = sign(e) * sqrt(|e|)`` of standard normal ``e``, as in factorized NoisyNets."""
    e = jax.random.normal(key, shape)
    return jnp.sign(e) * jnp.sqrt(jnp.abs(e))


def noisy_dense(
    p: dict[str, jax.Array], x: jax.Array, noise: jax.Array | None, per_sample: bool, cfg: Config
) -> jax.Array:
    """``x @ W + b`` for ``x: [B, in]``, with factorized Gaussian noise if ``p`` is noisy
    (``rainbow.py``'s ``dense``; the mean matmul in ``compute_dtype``)."""
    y = unified.matmul(x, p["w"], cfg) + p["b"]
    if noise is None or "w_sigma" not in p:
        return y
    k_in, k_out = jax.random.split(noise)
    lead = x.shape[:1] if per_sample else ()
    e_in = _scaled_noise(k_in, (*lead, x.shape[-1]))
    e_out = _scaled_noise(k_out, (*lead, y.shape[-1]))
    return y + e_out * ((x * e_in) @ p["w_sigma"] + p["b_sigma"])


def init_network(key: jax.Array, cfg: Config) -> Params:
    """He-normal convolutions; dense layers with the NoisyNet initialization when
    ``cfg.noisy`` (means uniform in ``+-1/sqrt(fan_in)``, scales ``sigma0/sqrt(fan_in)``),
    else He-normal; ``feat/w`` zero."""
    k_conv, k_dense = jax.random.split(key)
    params, flat = unified.init_convs(k_conv, cfg)
    keys = iter(jax.random.split(k_dense, 3))

    def linear(fan_in: int, fan_out: int, gain: float) -> dict[str, jax.Array]:
        if not cfg.noisy:
            return unified.dense(next(keys), (fan_in, fan_out), fan_in, gain)
        k_w, k_b = jax.random.split(next(keys))
        bound = 1.0 / math.sqrt(fan_in)
        sigma = cfg.noisy_sigma0 / math.sqrt(fan_in)
        return {
            "w": jax.random.uniform(k_w, (fan_in, fan_out), minval=-bound, maxval=bound),
            "b": jax.random.uniform(k_b, (fan_out,), minval=-bound, maxval=bound),
            "w_sigma": jnp.full((fan_in, fan_out), sigma),
            "b_sigma": jnp.full((fan_out,), sigma),
        }

    params["hidden"] = linear(flat, cfg.hidden, gain=2.0)
    if cfg.dueling:
        params["value"] = linear(cfg.hidden, cfg.num_atoms, gain=1.0)
        params["advantage"] = linear(cfg.hidden, NUM_ACTIONS * cfg.num_atoms, gain=1.0)
    else:
        params["logits"] = linear(cfg.hidden, NUM_ACTIONS * cfg.num_atoms, gain=1.0)
    return {**params, **unified.init_feat(cfg)}


def network(
    params: Params,
    boards: jax.Array,
    feats: jax.Array,
    cfg: Config,
    noise: jax.Array | None = None,
    per_sample: bool = False,
) -> jax.Array:
    """Return-distribution logits ``f32[..., 4, num_atoms]`` for boards ``[..., rows, cols, C]``.

    ``noise`` is a PRNG key for the noisy layers, or None for their means (the network
    used for play); ``per_sample`` draws separate noise for every row.
    """
    lead = boards.shape[:-3]
    x = unified.conv_features(params, boards, cfg)
    keys = [None] * 3 if noise is None else list(jax.random.split(noise, 3))
    pre = noisy_dense(params["hidden"], x, keys[0], per_sample, cfg)
    x = jax.nn.relu(pre + unified.feature_term(params, feats))
    shape = (x.shape[0], NUM_ACTIONS, cfg.num_atoms)
    if cfg.dueling:
        v = noisy_dense(params["value"], x, keys[1], per_sample, cfg)[:, None, :]
        a = noisy_dense(params["advantage"], x, keys[2], per_sample, cfg).reshape(shape)
        logits = v + a - a.mean(axis=1, keepdims=True)
    else:
        logits = noisy_dense(params["logits"], x, keys[1], per_sample, cfg).reshape(shape)
    return logits.reshape(*lead, NUM_ACTIONS, cfg.num_atoms)


def q_values(logits: jax.Array, cfg: Config) -> jax.Array:
    """Expected returns ``f32[..., 4]`` of return-distribution logits ``[..., 4, num_atoms]``."""
    return jnp.sum(jax.nn.softmax(logits, axis=-1) * support(cfg), axis=-1)


def epsilon_greedy(key: jax.Array, q: jax.Array, mask: jax.Array, eps: jax.Array) -> jax.Array:
    """Per agent: a random legal action with probability ``eps``, else the legal argmax."""
    k_explore, k_random = jax.random.split(key)
    greedy = jnp.argmax(jnp.where(mask, q, -jnp.inf), axis=-1)
    random = jax.random.categorical(k_random, jnp.where(mask, 0.0, -jnp.inf), axis=-1)
    explore = jax.random.uniform(k_explore, mask.shape[:-1]) < eps
    return jnp.where(explore, random, greedy).astype(jnp.int32)


def _linear_schedule(start: float, end: float, steps: int, env_steps: jax.Array) -> jax.Array:
    frac = jnp.clip(env_steps / max(steps, 1), 0.0, 1.0)
    return start + frac * (end - start)


def epsilon(cfg: Config, env_steps: jax.Array) -> jax.Array:
    """The exploration rate: 0 with noisy nets, else the linear epsilon schedule."""
    if cfg.noisy:
        return jnp.zeros((), jnp.float32)
    return _linear_schedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps, env_steps)


def per_beta(cfg: Config, env_steps: jax.Array) -> jax.Array:
    """Importance-sampling exponent, linear from ``per_beta_start`` to ``per_beta_end``."""
    return _linear_schedule(cfg.per_beta_start, cfg.per_beta_end, cfg.per_beta_steps, env_steps)


def make_policy(
    params: Params,
    cfg: Config,
    game_config: GameConfig,
    game_mode: str | None = None,
    greedy: bool = True,
) -> Policy:
    """An ``evaluate.Policy``: the legal argmax of the noise-free network's expected return,
    or epsilon-greedy with ``EVAL_EPSILON`` (``greedy=False``)."""
    mode = unified.told_mode(cfg, game_config, game_mode)

    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        boards, feats = unified.policy_inputs(ts.obs, state, game_config, cfg, mode)
        q = q_values(network(params, boards, feats, cfg), cfg)
        if greedy:
            return unified.greedy_actions(q, ts.action_mask)
        return epsilon_greedy(key, q, ts.action_mask, jnp.float32(EVAL_EPSILON))

    return policy


# --- Multi-step transitions (rainbow.py's) ---------------------------------------------


class Step(NamedTuple):
    """One-step game transitions of a group's slots (leaves ``[num_envs, ...]``)."""

    state: State
    actions: jax.Array  # int32[E, N]
    rewards: jax.Array  # float32[E, N]
    next_state: State  # s_{t+1} as reached, before any autoreset
    truncated: jax.Array  # bool[E]


class Transition(NamedTuple):
    """One stored ``n``-step game transition (all snakes)."""

    state: State
    actions: jax.Array  # int32[N]
    returns: jax.Array  # float32[N] discounted reward sum over the kept steps
    next_state: State  # bootstrap state: what the last kept step reached
    discount: jax.Array  # float32[N] gamma ** steps, or 0 for no bootstrap


def nstep_transition(window: Step, gamma: float) -> Transition:
    """``n``-step transitions starting at the oldest step of ``window`` (leaves ``[n, E, ...]``)."""
    n, num_envs = window.truncated.shape
    done = window.next_state.done  # [n, E]
    before = jnp.concatenate([jnp.ones((1, num_envs), bool), ~done[:-1]], axis=0)
    kept = jnp.cumprod(before, axis=0).astype(bool)
    steps = kept.sum(axis=0)  # [E] in 1..n
    powers = gamma ** jnp.arange(n, dtype=jnp.float32)
    returns = jnp.sum(jnp.where(kept[..., None], powers[:, None, None] * window.rewards, 0.0), 0)
    last, env = steps - 1, jnp.arange(num_envs)
    next_state = jax.tree.map(lambda x: x[last, env], window.next_state)
    truncated = window.truncated[last, env]
    ended = next_state.done & ~truncated
    bootstrap = next_state.alive & ~ended[:, None]
    discount = jnp.where(bootstrap, (gamma**steps)[:, None], 0.0).astype(jnp.float32)
    first = jax.tree.map(lambda x: x[0], window)
    return Transition(first.state, first.actions, returns, next_state, discount)


def window_push(window: Step, step: Step) -> Step:
    """Drop the oldest step of ``window`` and append ``step`` as the newest."""
    return jax.tree.map(lambda w, s: jnp.concatenate([w[1:], s[None]], axis=0), window, step)


# --- Prioritized replay (rainbow.py's) -------------------------------------------------


class Buffer(NamedTuple):
    """Circular buffer with a replay priority per item (leaves ``[capacity, ...]``)."""

    data: Any
    priority: jax.Array  # f32[capacity] raw priorities, 0 when empty
    max_priority: jax.Array  # f32[] the largest priority so far; new items get it
    ptr: jax.Array
    size: jax.Array


def buffer_init(example: Any, capacity: int) -> Buffer:
    data = jax.tree.map(lambda x: jnp.zeros((capacity, *x.shape), x.dtype), example)
    zero = jnp.zeros((), jnp.int32)
    return Buffer(data, jnp.zeros((capacity,), jnp.float32), jnp.ones((), jnp.float32), zero, zero)


def buffer_add(buf: Buffer, items: Any) -> Buffer:
    """Write a batch of ``B`` items at ``ptr`` with the maximum priority so far."""
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
        data, priority, buf.max_priority, (buf.ptr + batch) % capacity,
        jnp.minimum(buf.size + batch, capacity),
    )  # fmt: skip


def buffer_sample(
    priority: jax.Array, size: jax.Array, key: jax.Array, batch_size: int, alpha: float, beta
) -> tuple[jax.Array, jax.Array]:
    """``(indices, weights)``: a stratified proportional draw and its importance weights
    ``(size * P(i)) ** -beta``, divided by their maximum over the batch."""
    filled = jnp.arange(priority.shape[0]) < size
    scaled = jnp.where(filled, priority**alpha, 0.0)
    cdf = jnp.cumsum(scaled)
    total = cdf[-1]
    u = (jnp.arange(batch_size) + jax.random.uniform(key, (batch_size,))) / batch_size * total
    idx = jnp.minimum(jnp.searchsorted(cdf, u, side="right"), jnp.maximum(size - 1, 0))
    below = jnp.where(idx > 0, cdf[jnp.maximum(idx - 1, 0)], 0.0)
    mass = cdf[idx] - below
    prob = jnp.where(mass > 0, mass, scaled[idx]) / total
    weights = (size * prob) ** -beta
    return idx.astype(jnp.int32), weights / jnp.max(weights)


# --- Distributional targets and loss --------------------------------------------------


def project(returns: jax.Array, discount: jax.Array, probs: jax.Array, atoms: jax.Array):
    """Project the distribution of ``returns + discount * Z`` onto ``atoms`` (C51's Phi)."""
    dz = atoms[1] - atoms[0]
    tz = jnp.clip(returns[..., None] + discount[..., None] * atoms, atoms[0], atoms[-1])
    kernel = jnp.clip(1.0 - jnp.abs(tz[..., :, None] - atoms) / dz, 0.0, 1.0)
    return jnp.sum(probs[..., :, None] * kernel, axis=-2)


def target_distribution(
    tr: Transition, next_select: jax.Array, next_target: jax.Array, next_mask: jax.Array,
    cfg: Config,
) -> jax.Array:  # fmt: skip
    """Projected target distributions ``f32[..., N, num_atoms]`` for ``n``-step transitions."""
    a_star = jnp.argmax(jnp.where(next_mask, q_values(next_select, cfg), -jnp.inf), axis=-1)
    probs = jax.nn.softmax(next_target, axis=-1)
    p_next = jnp.take_along_axis(probs, a_star[..., None, None], axis=-2)[..., 0, :]
    return project(tr.returns, tr.discount, p_next, support(cfg))


def valid_agents(state: State) -> jax.Array:
    """``bool[..., N]``: the agent was alive in a running game at ``t``, so it acted."""
    return state.alive & ~state.done[..., None]


def joined_inputs(
    cfg: Config, groups: tuple[Group, ...], states: tuple[State, ...]
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
    """``(boards, feats, masks)``: every group's inputs, joined (the masks stay per group)."""
    inputs = [unified.group_inputs(cfg, g, s) for g, s in zip(groups, states, strict=True)]
    boards = unified.join([x[0] for x in inputs])
    feats = unified.join([x[1] for x in inputs])
    return boards, feats, tuple(x[2] for x in inputs)


def make_loss_fn(cfg: Config, groups: tuple[Group, ...]) -> Callable[..., Any]:
    """``loss(params, target_params, batches, weights, key) -> (loss, (grads, aux))``.

    Per group: the importance-weighted KL divergence from the projected target to the
    online distribution of the taken move, averaged over valid agents; the loss is their
    weighted mean. ``aux`` holds per group the new priorities ``f32[B]`` (mean KL of each
    transition), the loss and the mean Q.
    """
    sizes = (cfg.batch_size,) * len(groups)
    atoms = support(cfg)

    def noise(key: jax.Array) -> jax.Array | None:
        return key if cfg.noisy else None

    def kl_loss(params, boards, feats, actions, targets, valids, weights, key):
        logits_all = unified.split(network(params, boards, feats, cfg, noise(key)), groups, sizes)
        loss, out = jnp.zeros(()), []
        for g, (logits, a, target, valid, w) in enumerate(
            zip(logits_all, actions, targets, valids, weights, strict=True)
        ):
            taken = jnp.take_along_axis(logits, a[..., None, None], axis=-2)[..., 0, :]
            log_p = jax.nn.log_softmax(taken, axis=-1)
            kl = jnp.sum(jax.scipy.special.xlogy(target, target) - target * log_p, axis=-1)
            v = valid.astype(jnp.float32)
            count = jnp.maximum(v.sum(), 1.0)
            loss_g = jnp.sum(w[:, None] * v * kl) / count
            per_item = jnp.sum(v * kl, axis=-1) / jnp.maximum(v.sum(axis=-1), 1.0)
            mean_q = jnp.sum(v * jnp.sum(jnp.exp(log_p) * atoms, axis=-1)) / count
            loss = loss + cfg.loss_weights[g] * loss_g
            out.append((jnp.maximum(per_item, 0.0), loss_g, mean_q))
        return loss, tuple(out)

    def loss_and_grad(params, target_params, batches, weights, key):
        k_online, k_select, k_target = jax.random.split(key, 3)
        next_boards, next_feats, next_masks = joined_inputs(
            cfg, groups, tuple(tr.next_state for tr in batches)
        )
        next_target = network(target_params, next_boards, next_feats, cfg, noise(k_target))
        next_select = (
            network(params, next_boards, next_feats, cfg, noise(k_select))
            if cfg.double
            else next_target
        )
        targets = []
        for tr, sel, tgt, m in zip(
            batches, unified.split(next_select, groups, sizes),
            unified.split(next_target, groups, sizes), next_masks, strict=True,
        ):  # fmt: skip
            targets.append(jax.lax.stop_gradient(target_distribution(tr, sel, tgt, m, cfg)))
        boards, feats, _ = joined_inputs(cfg, groups, tuple(tr.state for tr in batches))
        actions = tuple(tr.actions for tr in batches)
        valids = tuple(valid_agents(tr.state) for tr in batches)
        (loss, aux), grads = jax.value_and_grad(kl_loss, has_aux=True)(
            params, boards, feats, actions, tuple(targets), valids, weights, k_online
        )
        return loss, (grads, jax.lax.stop_gradient(aux))

    return loss_and_grad


# --- Training ----------------------------------------------------------------------


class RunnerState(NamedTuple):
    """Everything the jitted training loop carries between chunks."""

    params: Params
    target_params: Params
    opt_state: Any
    env_states: tuple[State, ...]  # per group, [num_envs] running games (never done)
    windows: tuple[Step, ...]  # per group, [n_step, num_envs], oldest first
    buffers: tuple[Buffer, ...]  # per group
    key: jax.Array
    env_steps: jax.Array  # int32[] env steps taken, per mode
    updates: jax.Array  # int32[] gradient updates taken


def make_optimizer(cfg: Config) -> optax.GradientTransformation:
    """Clipping and Adam's scaling; the learning rate is applied by the update."""
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm), optax.scale_by_adam(eps=cfg.adam_eps)
    )


def init_runner(cfg: Config, groups: tuple[Group, ...], key: jax.Array) -> RunnerState:
    k_net, k_env, k_run = jax.random.split(key, 3)
    params = init_network(k_net, cfg)
    env_states, windows, buffers = [], [], []
    for g in groups:
        states = jax.vmap(g.sim.init_state)(
            jax.random.split(unified.group_key(k_env, g), cfg.num_envs)
        )
        e, n = cfg.num_envs, g.num_snakes
        step = Step(
            states, jnp.zeros((e, n), jnp.int32), jnp.zeros((e, n), jnp.float32), states,
            jnp.zeros((e,), bool),
        )  # fmt: skip
        one = jax.tree.map(lambda x: x[0], states)
        example = Transition(
            one, jnp.zeros((n,), jnp.int32), jnp.zeros((n,), jnp.float32), one,
            jnp.zeros((n,), jnp.float32),
        )  # fmt: skip
        env_states.append(states)
        # Placeholder steps; none is stored before n_step real ones have been pushed.
        windows.append(jax.tree.map(lambda x: jnp.stack([x] * cfg.n_step), step))
        buffers.append(buffer_init(example, cfg.capacity))
    return RunnerState(
        params=params,
        target_params=jax.tree.map(jnp.copy, params),
        opt_state=make_optimizer(cfg).init(params),
        env_states=tuple(env_states),
        windows=tuple(windows),
        buffers=tuple(buffers),
        key=k_run,
        env_steps=jnp.zeros((), jnp.int32),
        updates=jnp.zeros((), jnp.int32),
    )


def make_train_chunk(
    cfg: Config, groups: tuple[Group, ...]
) -> Callable[[RunnerState], tuple[RunnerState, dict[str, jax.Array]]]:
    """``train_chunk(runner) -> (runner, metrics)``: ``chunk_iters`` iterations in one scan.

    Each iteration acts in and steps every mode's games, pushes the steps into the
    windows and stores the completed ``n``-step transitions, and once the buffers hold
    ``learning_starts`` transitions takes ``updates_per_step`` gradient steps.
    """
    optimizer = make_optimizer(cfg)
    loss_and_grad = make_loss_fn(cfg, groups)
    envs = (cfg.num_envs,) * len(groups)
    start = max(cfg.learning_starts, 1)
    num_groups = len(groups)

    def gradient_step(datas, beta, env_steps, carry, key):
        params, target_params, opt_state, updates, priorities, max_priorities, sizes = carry
        k_sample, k_loss = jax.random.split(key)
        sample_keys = jax.random.split(k_sample, num_groups)
        draws = [
            buffer_sample(p, s, k, cfg.batch_size, cfg.per_alpha, beta)
            for p, s, k in zip(priorities, sizes, sample_keys, strict=True)
        ]
        batches = tuple(
            jax.tree.map(lambda d, i=i: d[i], data)
            for data, (i, _) in zip(datas, draws, strict=True)
        )
        weights = tuple(w for _, w in draws)
        loss, (grads, aux) = loss_and_grad(params, target_params, batches, weights, k_loss)
        step, opt_state = optimizer.update(grads, opt_state, params)
        lr = cfg.lr * unified.lr_factor(env_steps, cfg)
        params = optax.apply_updates(params, unified.scale_updates(step, lr))
        new_p, new_max = [], []
        for p, m, (idx, _), (per_item, _, _) in zip(
            priorities, max_priorities, draws, aux, strict=True
        ):
            new = per_item + cfg.per_eps
            new_p.append(p.at[idx].set(new))
            new_max.append(jnp.maximum(m, jnp.max(new)))
        updates = updates + 1
        sync = updates % cfg.target_update_period == 0
        target_params = jax.tree.map(
            lambda t, p: jnp.where(sync, optax.incremental_update(p, t, cfg.tau), t),
            target_params,
            params,
        )
        carry = (params, target_params, opt_state, updates, tuple(new_p), tuple(new_max), sizes)
        mode_loss = jnp.stack([a[1] for a in aux])
        mean_q = jnp.stack([a[2] for a in aux])
        return carry, (loss, mode_loss, mean_q)

    def learn(datas, beta, env_steps, carry, key):
        keys = jax.random.split(key, cfg.updates_per_step)
        carry, out = jax.lax.scan(
            lambda c, k: gradient_step(datas, beta, env_steps, c, k), carry, keys
        )
        return carry, (*(x.sum(0) for x in out), jnp.int32(cfg.updates_per_step))

    def skip(datas, beta, env_steps, carry, key):
        zeros = jnp.zeros(num_groups, jnp.float32)
        return carry, (jnp.float32(0), zeros, zeros, jnp.int32(0))

    def iteration(runner: RunnerState, _):
        key, k_noise, k_act, k_step, k_reset, k_learn = jax.random.split(runner.key, 6)
        boards, feats, masks = joined_inputs(cfg, groups, runner.env_states)
        noise = k_noise if cfg.noisy else None
        logits = network(runner.params, boards, feats, cfg, noise, per_sample=True)
        mask = jnp.concatenate([m.reshape(-1, NUM_ACTIONS) for m in masks])
        eps = epsilon(cfg, runner.env_steps)
        actions = epsilon_greedy(k_act, q_values(logits, cfg), mask, eps)
        actions = unified.split(actions, groups, envs)
        ready = runner.env_steps // cfg.num_envs >= cfg.n_step - 1

        env_states, windows, buffers, per_group = [], [], [], []
        for g, states, a, window, buf in zip(
            groups, runner.env_states, actions, runner.windows, runner.buffers, strict=True
        ):
            keys = jax.random.split(unified.group_key(k_step, g), cfg.num_envs)
            new, ts = jax.vmap(g.sim.step)(keys, states, a)
            window = window_push(window, Step(states, a, ts.reward, new, ts.truncated))
            added = buffer_add(buf, nstep_transition(window, cfg.gamma))
            # Until the window holds n real steps the write is not committed (ptr and size
            # stay), so the next one overwrites it.
            added = added._replace(
                ptr=jnp.where(ready, added.ptr, buf.ptr),
                size=jnp.where(ready, added.size, buf.size),
            )
            keys = jax.random.split(unified.group_key(k_reset, g), cfg.num_envs)
            fresh = jax.vmap(g.sim.init_state)(keys)
            env_states.append(unified.tree_where(ts.done, fresh, new))
            windows.append(window)
            buffers.append(added)
            per_group.append(
                {
                    "agent_transitions": jnp.sum(states.alive & ~states.done[:, None]),
                    **unified.game_metrics(states, new, ts.done, ts.truncated),
                }
            )

        beta = per_beta(cfg, runner.env_steps)
        carry = (
            runner.params, runner.target_params, runner.opt_state, runner.updates,
            tuple(b.priority for b in buffers), tuple(b.max_priority for b in buffers),
            tuple(b.size for b in buffers),
        )  # fmt: skip
        ready_to_learn = jnp.all(jnp.stack([b.size >= start for b in buffers]))
        datas = tuple(b.data for b in buffers)
        carry, (loss, mode_loss, mean_q, n_updates) = jax.lax.cond(
            ready_to_learn, learn, skip, datas, beta, runner.env_steps, carry, k_learn
        )
        params, target_params, opt_state, updates, priorities, max_priorities, _ = carry
        buffers = tuple(
            b._replace(priority=p, max_priority=m)
            for b, p, m in zip(buffers, priorities, max_priorities, strict=True)
        )
        metrics = {k: jnp.stack([p[k] for p in per_group]) for k in per_group[0]}
        metrics.update(loss=loss, mode_loss=mode_loss, mean_q=mean_q, updates=n_updates)
        runner = RunnerState(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            env_states=tuple(env_states),
            windows=tuple(windows),
            buffers=buffers,
            key=key,
            env_steps=runner.env_steps + cfg.num_envs,
            updates=updates,
        )
        return runner, metrics

    def train_chunk(runner: RunnerState) -> tuple[RunnerState, dict[str, jax.Array]]:
        runner, metrics = jax.lax.scan(iteration, runner, None, length=cfg.chunk_iters)
        metrics = jax.tree.map(lambda x: jnp.sum(x, axis=0), metrics)
        metrics["eps"] = epsilon(cfg, runner.env_steps)
        metrics["beta"] = per_beta(cfg, runner.env_steps)
        metrics["lr"] = cfg.lr * unified.lr_factor(runner.env_steps, cfg)
        metrics["noise_sigma"] = (
            jnp.mean(jnp.abs(runner.params["hidden"]["w_sigma"])) if cfg.noisy else jnp.float32(0)
        )
        metrics["env_steps"] = runner.env_steps
        metrics["total_updates"] = runner.updates
        metrics["buffer_size"] = runner.buffers[0].size
        return runner, metrics

    return train_chunk


# --- Records ----------------------------------------------------------------------------


def chunk_record(m: dict[str, Any], seconds: float, chunk_steps: int, cfg: Config) -> dict:
    """A JSON-friendly summary of one chunk: ``rainbow.py``'s fields at the top level
    (``loss`` the weighted mean over the modes, the game statistics and ``mean_q`` the
    first mode's) and a ``game_modes`` block with every mode's."""
    n_upd = max(int(m["updates"]), 1)
    per_mode = {}
    for g, (name, n) in enumerate(cfg.snakes.items()):
        per_mode[name] = {
            "num_snakes": n,
            **unified.mode_game_stats(m, g),
            "agent_transitions": int(m["agent_transitions"][g]),
            "loss": float(m["mode_loss"][g]) / n_upd,
            "mean_q": float(m["mean_q"][g]) / n_upd,
        }
    head = per_mode[cfg.game_modes[0]]
    return {
        "type": "train",
        "env_steps": int(m["env_steps"]),
        "env_steps_all": int(m["env_steps"]) * len(cfg.game_modes),
        "agent_samples": sum(p["agent_transitions"] for p in per_mode.values()),
        "updates": int(m["updates"]),
        "total_updates": int(m["total_updates"]),
        "loss": float(m["loss"]) / n_upd,
        "mean_q": head["mean_q"],
        "eps": float(m["eps"]),
        "beta": float(m["beta"]),
        "lr": float(m["lr"]),
        "noise_sigma": float(m["noise_sigma"]),
        "buffer_size": int(m["buffer_size"]),
        **{k: head[k] for k in ("games", "mean_game_len", "draws", "truncated", "deaths")},
        "env_steps_per_s": chunk_steps / seconds,
        "seconds": seconds,
        "game_modes": per_mode,
    }


def format_record(r: dict[str, Any], total_steps: int) -> str:
    head = (
        f"steps {r['env_steps']:>10,} ({100 * r['env_steps'] / total_steps:3.0f}%) per mode | "
        f"loss {r['loss']:.4f} sigma {r['noise_sigma']:.4f} beta {r['beta']:.2f} "
        f"lr {r['lr']:.2e} buffer {r['buffer_size']:,} | "
        f"{r['env_steps_per_s']:,.0f} steps/s per mode"
    )
    lines = [head]
    for name, p in r["game_modes"].items():
        lines.append(
            f"  {name:>11} | loss {p['loss']:.4f} Q {p['mean_q']:+.3f} | {unified.format_games(p)}"
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
