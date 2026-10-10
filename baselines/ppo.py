"""Self-play PPO baseline for the 1v1 duel (single file, cleanRL / PureJaxRL style).

* **Self-play with parameter sharing.** One actor-critic network acts for both
  snakes, each from its own egocentric observation. Illegal moves
  (``env.action_mask``) are masked out of the policy everywhere: in sampling, in
  the log-probabilities and in the entropy.
* **Vectorized.** ``num_envs`` games run in parallel and a finished game is
  replaced by a fresh one right away. A rollout of ``num_steps`` turns is one
  jitted ``lax.scan``; every turn gives one sample per living snake.
* **Per-agent episodes.** A snake's trajectory ends when it dies (-1), wins
  (+1) or draws (0, both snakes die on the same turn): no bootstrapping. A game
  cut off by ``max_turns`` is truncated, not terminated, so the snakes still in
  it bootstrap from the value of the state reached. GAE never crosses an
  episode boundary (:func:`compute_gae`): after a snake's last step its slot
  either belongs to a dead snake (left out of the loss) or to the next game.
* **Learning.** PPO-clip: GAE(lambda), clipped surrogate, (optionally clipped)
  value loss, entropy bonus, per-minibatch advantage normalisation, several
  epochs over shuffled minibatches, Adam with gradient clipping and a linear
  learning-rate decay.

The reward is the environment's sparse win/loss reward (no shaping), as in the
DQN baseline. Everything between two log lines (rollouts and updates) is one
jitted ``lax.scan`` (:func:`make_train_chunk`). The Python loop only logs,
evaluates (greedy and sampled play against ``random_legal``, the heuristic and
the DQN baseline) and saves checkpoints::

    python baselines/ppo.py                                   # default run
    python baselines/ppo.py --total-env-steps 1000000 --lr 3e-4
    python baselines/ppo.py --eval-only runs/ppo-20261010-120000 --eval-games 1000
    python baselines/ppo.py --resume runs/ppo-20261010-120000   # after an interruption

An *env step* is one game advancing one turn. It gives one *agent sample* per
living snake, so two in a duel. The file is self-contained (it shares no code
with ``baselines/dqn.py``; the network and checkpoint helpers follow it).
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
from slinky.evaluate import MatchResult, Policy, greedy_from_q, play_match
from slinky.observations import obs_shape
from slinky.types import NUM_ACTIONS, Cause, GameConfig, State, TimeStep

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
EVAL_MODES = ("greedy", "sample")
# Logit of an illegal move: exp() underflows to exactly 0, but unlike -inf it keeps
# p * log p and every gradient finite.
MASKED_LOGIT = -1e9


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """All hyperparameters. Every field can be overridden on the command line.

    Steps are *env steps* (games advanced one turn). One iteration collects
    ``num_envs * num_steps`` env steps, i.e. twice as many agent samples in a
    duel, then takes ``update_epochs * num_minibatches`` gradient steps.
    """

    # Game: the 11x11 standard duel, truncated after max_turns turns.
    max_turns: int = 1000
    # Collection.
    num_envs: int = 64  # games stepped in parallel
    num_steps: int = 128  # turns per rollout (per iteration)
    total_env_steps: int = 5_242_880  # rounded up to whole log chunks
    # PPO.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 8  # per epoch; agent samples must divide evenly
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    norm_adv: bool = True
    # Optimization.
    lr: float = 1e-3
    anneal_lr: bool = True  # linear decay to 0 over the run
    max_grad_norm: float = 0.5
    adam_eps: float = 1e-5
    # Network: 3x3 convs (ReLU), a dense hidden layer (ReLU), then linear policy and value heads.
    # Half the DQN's channels: twice the env steps per second, as strong per minute in pilots.
    conv_channels: tuple[int, ...] = (16, 32, 32)
    conv_strides: tuple[int, ...] = (2, 2, 1)
    hidden: int = 256
    # Logging, evaluation and checkpoints.
    seed: int = 0
    log_every: int = 32_768  # env steps per jitted chunk (one log line), whole iterations
    eval_every: int = 1_048_576  # env steps between evaluations (and checkpoints)
    eval_games: int = 256  # per opponent and mode
    eval_opponents: tuple[str, ...] = ("random_legal", "heuristic", "dqn")  # slinky.agents
    eval_modes: tuple[str, ...] = EVAL_MODES  # greedy (masked argmax) and/or sample
    checkpoint_every: int = 262_144  # env steps between full checkpoints (for --resume)
    run_dir: str | None = None  # default: runs/ppo-<timestamp>

    def __post_init__(self) -> None:
        if len(self.conv_channels) != len(self.conv_strides):
            raise ValueError("conv_channels and conv_strides must have the same length")
        positive = (
            "max_turns", "num_envs", "num_steps", "total_env_steps", "update_epochs",
            "num_minibatches", "hidden", "log_every", "eval_every", "checkpoint_every",
        )  # fmt: skip
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        samples = self.num_envs * self.num_steps * self.game.num_snakes
        if samples % self.num_minibatches:
            raise ValueError(
                f"num_minibatches ({self.num_minibatches}) must divide the agent samples per "
                f"iteration (num_envs * num_steps * {self.game.num_snakes} = {samples})"
            )
        if bad := set(self.eval_modes) - set(EVAL_MODES):
            raise ValueError(f"unknown eval_modes {sorted(bad)}; expected {EVAL_MODES}")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")

    @property
    def game(self) -> GameConfig:
        return GameConfig(max_turns=self.max_turns)

    @property
    def steps_per_iteration(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.steps_per_iteration * self.game.num_snakes // self.num_minibatches


class Schedule(NamedTuple):
    """How a run's env steps split into iterations and log chunks."""

    chunk_iters: int  # PPO iterations per jitted chunk
    chunk_steps: int  # env steps per chunk
    num_chunks: int
    total_steps: int  # env steps of the whole run (total_env_steps rounded up)
    total_updates: int  # gradient steps of the whole run (the learning-rate decay)


def schedule(cfg: PPOConfig) -> Schedule:
    chunk_iters = max(cfg.log_every // cfg.steps_per_iteration, 1)
    chunk_steps = chunk_iters * cfg.steps_per_iteration
    num_chunks = -(-cfg.total_env_steps // chunk_steps)
    updates = num_chunks * chunk_iters * cfg.update_epochs * cfg.num_minibatches
    return Schedule(chunk_iters, chunk_steps, num_chunks, num_chunks * chunk_steps, updates)


# --- Actor-critic network ----------------------------------------------------------


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
    outside one; matmuls don't suffer (the same trick as ``baselines/dqn.py``).
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


def init_network(key: jax.Array, cfg: PPOConfig) -> Params:
    """He-normal torso, a near-zero policy head (so play starts uniform over legal moves)."""
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
    params["policy"] = layer((cfg.hidden, NUM_ACTIONS), cfg.hidden, gain=1e-4)  # std 0.01/sqrt(n)
    params["value"] = layer((cfg.hidden, 1), cfg.hidden, gain=1.0)
    return params


def network(params: Params, obs: jax.Array, cfg: PPOConfig) -> tuple[jax.Array, jax.Array]:
    """``(logits f32[..., 4], value f32[...])`` for observations ``f32[..., rows, cols, C]``.

    The logits are unmasked; see :func:`masked_log_probs`.
    """
    lead = obs.shape[:-3]
    x = obs.reshape(-1, *obs.shape[-3:])
    for i, stride in enumerate(cfg.conv_strides):
        p = params[f"conv{i}"]
        x = jax.nn.relu(conv3x3(x, p["w"], stride) + p["b"])
    x = x.reshape(x.shape[0], -1)
    x = jax.nn.relu(x @ params["hidden"]["w"] + params["hidden"]["b"])
    logits = x @ params["policy"]["w"] + params["policy"]["b"]
    value = x @ params["value"]["w"] + params["value"]["b"]
    return logits.reshape(*lead, NUM_ACTIONS), value.reshape(lead)


# --- The masked policy --------------------------------------------------------------


def masked_log_probs(logits: jax.Array, mask: jax.Array) -> jax.Array:
    """Log-probabilities ``f32[..., 4]`` of the policy restricted to the legal moves.

    Illegal moves get probability exactly 0 (log-probability about -1e9).
    Mask rows are never all-False, so the distribution is always defined.
    """
    return jax.nn.log_softmax(jnp.where(mask, logits, MASKED_LOGIT), axis=-1)


def masked_entropy(log_probs: jax.Array, mask: jax.Array) -> jax.Array:
    """Entropy ``f32[...]`` of the masked policy (illegal moves contribute nothing)."""
    return -jnp.sum(jnp.where(mask, jnp.exp(log_probs) * log_probs, 0.0), axis=-1)


def sample_actions(key: jax.Array, logits: jax.Array, mask: jax.Array) -> jax.Array:
    """One legal move per row of ``logits``/``mask`` (``[..., 4]``), sampled from the policy."""
    return jax.random.categorical(key, jnp.where(mask, logits, MASKED_LOGIT)).astype(jnp.int32)


def make_policy(params: Params, cfg: PPOConfig, greedy: bool) -> Policy:
    """An ``evaluate.Policy`` for the network: greedy (masked argmax) or sampled.

    The parameters are baked into whatever traces the policy (e.g. a compiled
    ``play_match``), so a new call compiles again.
    """

    def logits_fn(obs: jax.Array) -> jax.Array:
        return network(params, obs, cfg)[0]

    return greedy_from_q(logits_fn) if greedy else sample_from_logits(logits_fn)


def sample_from_logits(logits_fn: Callable[[Any], jax.Array]) -> Policy:
    """Policy sampling each snake's move from ``softmax(logits_fn(obs))`` over its legal moves."""
    return _sample_from_logits(logits_fn)


@functools.lru_cache(maxsize=16)
def _sample_from_logits(logits_fn: Callable[[Any], jax.Array]) -> Policy:
    # Cached like evaluate.greedy_from_q, so the same logits_fn hits play_match's jit cache.
    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        return sample_actions(key, logits_fn(ts.obs), ts.action_mask)

    return policy


# --- Rollouts and GAE ---------------------------------------------------------------


class Rollout(NamedTuple):
    """One rollout. Leaves are ``[T, E, N, ...]`` (``done``: ``[T, E]``).

    Step ``t`` of env ``e`` goes from ``s_t`` (never a finished game) to
    ``s_{t+1}`` as reached, before any reset.
    """

    obs: jax.Array  # f32[T, E, N, rows, cols, C] observation of s_t
    mask: jax.Array  # bool[T, E, N, 4] legal moves at s_t
    actions: jax.Array  # int32[T, E, N]
    log_probs: jax.Array  # f32[T, E, N] log pi_old(a_t | s_t)
    values: jax.Array  # f32[T, E, N] V_old(s_t)
    rewards: jax.Array  # f32[T, E, N]
    valid: jax.Array  # bool[T, E, N] alive at s_t: the agent acted, the sample is its own
    terminal: jax.Array  # bool[T, E, N] no bootstrap: died, or the rules ended the game
    done: jax.Array  # bool[T, E] the game ended at t+1 (rules or max_turns): s_{t+1} is reset
    boot_values: jax.Array  # f32[T, E, N] V(s_{t+1}) if the game was truncated at t+1, else 0


def agent_flags(state: State, next_state: State, truncated: jax.Array) -> tuple[jax.Array, ...]:
    """Per-agent ``(valid, terminal)`` flags, ``bool[..., N]``, of game transitions.

    * ``valid``: the agent was alive in a running game at ``t``, so it acted.
    * ``terminal``: no bootstrapping from ``s_{t+1}``. The agent died, or the
      rules ended the game: the winner's +1 and a draw's 0 are final. Agents
      still alive when ``max_turns`` cuts the game off bootstrap.
    """
    valid = state.alive & ~state.done[..., None]
    ended = next_state.done & ~truncated
    terminal = ~next_state.alive | ended[..., None]
    return valid, terminal


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    valid: jax.Array,
    terminal: jax.Array,
    done: jax.Array,
    boot_values: jax.Array,
    last_values: jax.Array,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """Per-agent GAE(lambda): ``(advantages, returns)``, ``f32[T, E, N]``, 0 where not valid.

    Arrays are ``[T, E, N]`` as in :class:`Rollout`, ``done`` is ``[T, E]`` and
    ``last_values`` ``[E, N]`` is ``V(s_T)``, the value of the states the next
    rollout starts from. The value of ``s_{t+1}`` as agent ``i`` of step ``t``
    sees it is:

    * nothing (0) if the step is terminal for it (it died, won or drew);
    * ``boot_values[t]`` if the game was truncated at ``t+1`` (the slot was reset,
      so ``values[t+1]`` belongs to the next game);
    * ``values[t+1]`` otherwise (``last_values`` for the last step).

    The recursion ``A_t = delta_t + gamma * lam * A_{t+1}`` only continues while
    the agent's next step is in the same game and it is still alive.
    """
    done = done[..., None]
    next_values = jnp.concatenate([values[1:], last_values[None]], axis=0)
    next_values = jnp.where(done, boot_values, next_values)
    deltas = rewards + gamma * jnp.where(terminal, 0.0, next_values) - values
    cont = ~terminal & ~done

    def step(next_adv, x):
        delta, c = x
        adv = delta + gamma * lam * jnp.where(c, next_adv, 0.0)
        return adv, adv

    _, adv = jax.lax.scan(step, jnp.zeros_like(last_values), (deltas, cont), reverse=True)
    adv = jnp.where(valid, adv, 0.0)
    return adv, jnp.where(valid, adv + values, 0.0)


# --- Loss --------------------------------------------------------------------------


class Batch(NamedTuple):
    """Flat agent samples for the update (leading axis ``B``)."""

    obs: jax.Array
    mask: jax.Array
    actions: jax.Array
    log_probs: jax.Array
    values: jax.Array
    advantages: jax.Array
    returns: jax.Array
    valid: jax.Array


LOSS_METRICS = ("loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac")


def ppo_loss(params: Params, mb: Batch, cfg: PPOConfig) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The PPO-clip loss of a minibatch, and its parts (means over the valid samples)."""
    logits, values = network(params, mb.obs, cfg)
    log_probs_all = masked_log_probs(logits, mb.mask)
    log_probs = jnp.take_along_axis(log_probs_all, mb.actions[..., None], axis=-1)[..., 0]
    w = mb.valid.astype(jnp.float32)
    count = jnp.maximum(w.sum(), 1.0)

    def mean(x: jax.Array) -> jax.Array:
        return jnp.sum(w * x) / count

    adv = mb.advantages
    if cfg.norm_adv:
        mu = mean(adv)
        adv = (adv - mu) / (jnp.sqrt(mean((adv - mu) ** 2)) + 1e-8)
    log_ratio = log_probs - mb.log_probs
    ratio = jnp.exp(log_ratio)
    clipped = jnp.clip(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
    pg_loss = mean(jnp.maximum(-adv * ratio, -adv * clipped))
    v_err = (values - mb.returns) ** 2
    if cfg.clip_vloss:
        v_clipped = mb.values + jnp.clip(values - mb.values, -cfg.clip_coef, cfg.clip_coef)
        v_err = jnp.maximum(v_err, (v_clipped - mb.returns) ** 2)
    v_loss = 0.5 * mean(v_err)
    entropy = mean(masked_entropy(log_probs_all, mb.mask))
    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
    parts = {
        "loss": loss,
        "pg_loss": pg_loss,
        "v_loss": v_loss,
        "entropy": entropy,
        "approx_kl": mean((ratio - 1.0) - log_ratio),
        "clip_frac": mean((jnp.abs(ratio - 1.0) > cfg.clip_coef).astype(jnp.float32)),
    }
    return loss, parts


def explained_variance(values: jax.Array, returns: jax.Array, valid: jax.Array) -> jax.Array:
    """``1 - Var(returns - values) / Var(returns)`` over the valid samples."""
    w = valid.astype(jnp.float32)
    count = jnp.maximum(w.sum(), 1.0)

    def var(x: jax.Array) -> jax.Array:
        mu = jnp.sum(w * x) / count
        return jnp.sum(w * (x - mu) ** 2) / count

    var_ret = var(returns)
    return jnp.where(var_ret > 0, 1.0 - var(returns - values) / var_ret, jnp.nan)


# --- Training ----------------------------------------------------------------------


class RunnerState(NamedTuple):
    """Everything the jitted training loop carries between chunks."""

    params: Params
    opt_state: Any
    env_states: State  # [num_envs] running games (never done)
    key: jax.Array
    env_steps: jax.Array  # int32[] env steps taken
    updates: jax.Array  # int32[] gradient steps taken


def make_envs(cfg: PPOConfig) -> tuple[BattlesnakeEnv, BattlesnakeEnv]:
    """``(env, sim)``: ``env`` computes egocentric observations, ``sim`` steps without them."""
    return BattlesnakeEnv(cfg.game, obs="egocentric"), BattlesnakeEnv(cfg.game, obs=None)


def learning_rate(cfg: PPOConfig) -> optax.Schedule:
    """Constant, or a linear decay from ``lr`` to 0 over the run's gradient steps."""
    if not cfg.anneal_lr:
        return optax.constant_schedule(cfg.lr)
    return optax.linear_schedule(cfg.lr, 0.0, schedule(cfg).total_updates)


def make_optimizer(cfg: PPOConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate(cfg), eps=cfg.adam_eps),
    )


def init_runner(cfg: PPOConfig, sim: BattlesnakeEnv, key: jax.Array) -> RunnerState:
    k_net, k_env, k_run = jax.random.split(key, 3)
    params = init_network(k_net, cfg)
    return RunnerState(
        params=params,
        opt_state=make_optimizer(cfg).init(params),
        env_states=jax.vmap(sim.init_state)(jax.random.split(k_env, cfg.num_envs)),
        key=k_run,
        env_steps=jnp.zeros((), jnp.int32),
        updates=jnp.zeros((), jnp.int32),
    )


def make_rollout(
    cfg: PPOConfig, env: BattlesnakeEnv, sim: BattlesnakeEnv
) -> Callable[[Params, State, jax.Array], tuple[State, Rollout, dict[str, jax.Array]]]:
    """``rollout(params, states, key) -> (states, Rollout, metrics)`` over ``num_steps`` turns.

    Metrics (games, lengths, draws, deaths) are summed over the rollout.
    """
    observe = jax.vmap(env.observe)  # [E] states -> [E, N, rows, cols, C]
    masks = jax.vmap(env.action_mask)  # [E] states -> [E, N, 4]
    num_envs = cfg.num_envs

    def turn(params, states, key):
        k_act, k_step, k_reset = jax.random.split(key, 3)
        obs, mask = observe(states), masks(states)
        logits, values = network(params, obs, cfg)
        actions = sample_actions(k_act, logits, mask)
        log_probs = jnp.take_along_axis(
            masked_log_probs(logits, mask), actions[..., None], axis=-1
        )[..., 0]

        new, ts = jax.vmap(sim.step)(jax.random.split(k_step, num_envs), states, actions)
        valid, terminal = agent_flags(states, new, ts.truncated)
        # Truncations are rare (max_turns): only then evaluate the reached states.
        boot_values = jax.lax.cond(
            jnp.any(ts.truncated),
            lambda: jnp.where(ts.truncated[:, None], network(params, observe(new), cfg)[1], 0.0),
            lambda: jnp.zeros_like(values),
        )
        fresh = jax.vmap(sim.init_state)(jax.random.split(k_reset, num_envs))
        next_states = jax.vmap(_tree_where)(ts.done, fresh, new)

        died = states.alive & ~new.alive  # [E, N]
        causes = jnp.arange(_NUM_CAUSES) == new.elim_cause[..., None]  # [E, N, causes]
        metrics = {
            "agent_samples": jnp.sum(valid),
            "games": jnp.sum(ts.done),
            "game_turns": jnp.sum(jnp.where(ts.done, new.turn, 0)),
            "draws": jnp.sum(ts.done & ~ts.truncated & ~jnp.any(new.alive, axis=-1)),
            "truncated": jnp.sum(ts.truncated),
            "deaths": jnp.sum(causes & died[..., None], axis=(0, 1)),
        }
        step = Rollout(
            obs=obs,
            mask=mask,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=ts.reward,
            valid=valid,
            terminal=terminal,
            done=ts.done,
            boot_values=boot_values,
        )
        return next_states, (step, metrics)

    def rollout(params, states, key):
        keys = jax.random.split(key, cfg.num_steps)
        states, (traj, metrics) = jax.lax.scan(lambda s, k: turn(params, s, k), states, keys)
        return states, traj, jax.tree.map(lambda x: jnp.sum(x, axis=0), metrics)

    return rollout


def make_train_chunk(
    cfg: PPOConfig, env: BattlesnakeEnv, sim: BattlesnakeEnv, num_iters: int
) -> Callable[[RunnerState], tuple[RunnerState, dict[str, jax.Array]]]:
    """``train_chunk(runner) -> (runner, metrics)``: ``num_iters`` PPO iterations in one scan.

    Each iteration collects a rollout of ``num_steps`` turns in all games,
    computes GAE, then takes ``update_epochs`` passes over the agent samples in
    ``num_minibatches`` shuffled minibatches. Rollout metrics are summed over
    the chunk; loss metrics are summed over its gradient steps.
    """
    optimizer = make_optimizer(cfg)
    rollout = make_rollout(cfg, env, sim)
    observe = jax.vmap(env.observe)
    lr = learning_rate(cfg)
    loss_and_grad = jax.value_and_grad(functools.partial(ppo_loss, cfg=cfg), has_aux=True)
    num_samples = cfg.steps_per_iteration * env.num_agents

    def gradient_step(batch, carry, idx):
        params, opt_state = carry
        mb = jax.tree.map(lambda x: x[idx], batch)
        (_, parts), grads = loss_and_grad(params, mb)
        step, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, step)
        return (params, opt_state), {**parts, "grad_norm": optax.tree.norm(grads)}

    def epoch(batch, carry, key):
        perm = jax.random.permutation(key, num_samples)
        idx = perm.reshape(cfg.num_minibatches, -1)
        return jax.lax.scan(lambda c, i: gradient_step(batch, c, i), carry, idx)

    def iteration(runner: RunnerState, _):
        key, k_roll, k_epochs = jax.random.split(runner.key, 3)
        env_states, traj, metrics = rollout(runner.params, runner.env_states, k_roll)
        last_values = network(runner.params, observe(env_states), cfg)[1]
        adv, ret = compute_gae(
            traj.rewards, traj.values, traj.valid, traj.terminal, traj.done,
            traj.boot_values, last_values, cfg.gamma, cfg.gae_lambda,
        )  # fmt: skip

        def flat(x: jax.Array) -> jax.Array:  # [T, E, N, ...] -> [T * E * N, ...]
            return x.reshape(num_samples, *x.shape[3:])

        batch = Batch(
            *map(flat, (traj.obs, traj.mask, traj.actions, traj.log_probs, traj.values)),
            advantages=flat(adv),
            returns=flat(ret),
            valid=flat(traj.valid),
        )
        carry = (runner.params, runner.opt_state)
        keys = jax.random.split(k_epochs, cfg.update_epochs)
        (params, opt_state), parts = jax.lax.scan(lambda c, k: epoch(batch, c, k), carry, keys)

        w = traj.valid.astype(jnp.float32)
        metrics.update({k: jnp.sum(v) for k, v in parts.items()})
        metrics["explained_var"] = explained_variance(traj.values, ret, traj.valid)
        metrics["value_mean"] = jnp.sum(w * traj.values) / jnp.maximum(w.sum(), 1.0)
        runner = RunnerState(
            params=params,
            opt_state=opt_state,
            env_states=env_states,
            key=key,
            env_steps=runner.env_steps + cfg.steps_per_iteration,
            updates=runner.updates + cfg.update_epochs * cfg.num_minibatches,
        )
        return runner, metrics

    def train_chunk(runner: RunnerState) -> tuple[RunnerState, dict[str, jax.Array]]:
        runner, metrics = jax.lax.scan(iteration, runner, None, length=num_iters)
        means = ("explained_var", "value_mean")
        metrics = {k: jnp.mean(v, 0) if k in means else jnp.sum(v, 0) for k, v in metrics.items()}
        metrics["env_steps"] = runner.env_steps
        metrics["updates"] = jnp.int32(num_iters * cfg.update_epochs * cfg.num_minibatches)
        metrics["total_updates"] = runner.updates
        metrics["lr"] = lr(runner.updates)
        return runner, metrics

    return train_chunk


def _tree_where(cond: jax.Array, a: Any, b: Any) -> Any:
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


# --- Evaluation and checkpoints ---------------------------------------------------


def evaluate_params(
    params: Params,
    cfg: PPOConfig,
    env: BattlesnakeEnv,
    key: jax.Array,
    num_games: int,
    opponent: str = "heuristic",
    greedy: bool = True,
) -> MatchResult:
    """The network (greedy or sampled) vs a ``slinky.agents`` opponent, seats balanced.

    The parameters are baked into the compiled match (see ``slinky.evaluate``),
    so every call compiles once more (a few seconds).
    """
    from slinky.agents import make_agent  # the registry can import this file in turn

    # A private copy: the training loop donates (deletes) the runner's arrays.
    params = jax.tree.map(jnp.copy, params)
    policy = make_policy(params, cfg, greedy)
    other = make_agent(opponent, env.config).policy
    return play_match(env, policy, other, key, num_games, max_turns=cfg.max_turns)


def run_evals(
    params: Params,
    cfg: PPOConfig,
    env: BattlesnakeEnv,
    key: jax.Array,
    num_games: int,
    opponents: tuple[str, ...],
    modes: tuple[str, ...],
    env_steps: int,
    write: Callable[[dict[str, Any]], None] = lambda record: None,
) -> list[dict[str, Any]]:
    """Every (opponent, mode) match with the same games; prints and returns eval records."""
    records = []
    for opponent in opponents:
        for mode in modes:
            t = time.perf_counter()
            r = evaluate_params(params, cfg, env, key, num_games, opponent, mode == "greedy")
            seconds = time.perf_counter() - t
            record = {
                "type": "eval",
                "env_steps": env_steps,
                "opponent": opponent,
                "mode": mode,
                **r._asdict(),
                "seconds": seconds,
            }
            write(record)
            records.append(record)
            where = f"@ {env_steps:,} " if env_steps >= 0 else ""
            print(f"eval {where}{mode:>6} vs {opponent}: {format_eval(r)} ({seconds:.0f}s)")
    return records


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


def save_config(run_dir: str, cfg: PPOConfig) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)


def load_config(run_dir: str) -> PPOConfig:
    with open(os.path.join(run_dir, "config.json")) as f:
        d = json.load(f)
    fields = {f.name: f for f in dataclasses.fields(PPOConfig)}
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in fields}
    return PPOConfig(**kwargs)


def load_params(run_dir: str, cfg: PPOConfig | None = None) -> Params:
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


def save_runner(run_dir: str, runner: RunnerState, agent_samples: int) -> None:
    """Write the whole training state to ``runner.npz`` atomically."""
    runner = runner._replace(key=jax.random.key_data(runner.key))
    leaves = {f"leaf{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(runner))}
    tmp = os.path.join(run_dir, "runner.tmp.npz")
    np.savez(tmp, agent_samples=agent_samples, **leaves)
    os.replace(tmp, os.path.join(run_dir, "runner.npz"))


def load_runner(run_dir: str, template: RunnerState) -> tuple[RunnerState, int]:
    """``(runner, agent_samples)`` from ``runner.npz``, structured like ``template``."""
    template = template._replace(key=jax.random.key_data(template.key))
    refs, treedef = jax.tree.flatten(template)
    with np.load(os.path.join(run_dir, "runner.npz")) as f:
        leaves = [f[f"leaf{i}"] for i in range(len(refs))]
        agent_samples = int(f["agent_samples"])
    for leaf, ref in zip(leaves, refs, strict=True):
        if leaf.shape != ref.shape or leaf.dtype != ref.dtype:
            raise ValueError(f"runner.npz does not match the config: {leaf.shape} vs {ref.shape}")
    runner = jax.tree.unflatten(treedef, [jnp.asarray(x) for x in leaves])
    return runner._replace(key=jax.random.wrap_key_data(runner.key)), agent_samples


def _truncate_metrics(path: str, env_steps: int) -> None:
    """Drop records logged after the checkpoint we resume from (they will be redone)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        records = [line for line in f if json.loads(line)["env_steps"] <= env_steps]
    with open(path, "w") as f:
        f.writelines(records)


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
    """A JSON-friendly summary of one chunk's metrics."""
    games, n_upd = int(m["games"]), int(m["updates"])
    deaths = {name: int(m["deaths"][c]) for name, c in DEATH_CAUSES.items()}
    ev = float(m["explained_var"])
    return {
        "type": "train",
        "env_steps": int(m["env_steps"]),
        "agent_samples": int(m["agent_samples"]),
        "total_updates": int(m["total_updates"]),
        "lr": float(m["lr"]),
        **{k: float(m[k]) / n_upd for k in (*LOSS_METRICS, "grad_norm")},
        "explained_var": ev if math.isfinite(ev) else None,
        "value_mean": float(m["value_mean"]),
        "games": games,
        "mean_game_len": float(m["game_turns"]) / games if games else None,
        "draws": int(m["draws"]),
        "truncated": int(m["truncated"]),
        "deaths": deaths,
        "env_steps_per_s": chunk_steps / seconds,
        "seconds": seconds,
    }


def format_record(r: dict[str, Any], total_steps: int) -> str:
    deaths = r["deaths"]
    n_dead = sum(deaths.values())
    shown = [k for k in DEATH_CAUSES if k != "hazard" or deaths[k]]
    length = f"{r['mean_game_len']:.0f}" if r["mean_game_len"] is not None else "-"
    ev = f"{r['explained_var']:.2f}" if r["explained_var"] is not None else "-"
    return (
        f"steps {r['env_steps']:>9,} ({100 * r['env_steps'] / total_steps:3.0f}%) | "
        f"pg {r['pg_loss']:+.4f} v {r['v_loss']:.4f} ent {r['entropy']:.3f} "
        f"kl {r['approx_kl']:.4f} clip {r['clip_frac']:.2f} ev {ev} V {r['value_mean']:+.2f} | "
        f"games {r['games']:,} len {length} draw {_pct(r['draws'], r['games'])} | deaths "
        + " ".join(f"{k} {_pct(deaths[k], n_dead)}" for k in shown)
        + f" | {r['env_steps_per_s']:,.0f} steps/s"
    )


# --- Main loop -----------------------------------------------------------------------


def train(cfg: PPOConfig, resume: bool = False) -> tuple[str, Params]:
    """Run a full training; returns ``(run_dir, final params)``.

    With ``resume``, continue from ``run_dir/runner.npz`` (written every
    ``checkpoint_every`` env steps) if it exists. Resuming is exact: the run
    continues as if it had never stopped.
    """
    run_dir = cfg.run_dir or os.path.join("runs", time.strftime("ppo-%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    cfg = dataclasses.replace(cfg, run_dir=run_dir)
    save_config(run_dir, cfg)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    env, sim = make_envs(cfg)
    if cfg.eval_games > 0:  # fail now, not at the first evaluation, if one can't be built
        from slinky.agents import make_agent

        for opponent in cfg.eval_opponents:
            make_agent(opponent, env.config)
    k_eval, k_init = jax.random.split(jax.random.key(cfg.seed))
    runner = jax.jit(init_runner, static_argnums=(0, 1))(cfg, sim, k_init)
    agent_total = 0
    if resume and os.path.exists(os.path.join(run_dir, "runner.npz")):
        runner, agent_total = load_runner(run_dir, runner)
        _truncate_metrics(metrics_path, int(runner.env_steps))
        print(f"resuming from env step {int(runner.env_steps):,}")
    elif os.path.exists(metrics_path):
        os.remove(metrics_path)  # a fresh run in an old directory

    sched = schedule(cfg)
    chunk_steps = sched.chunk_steps
    n_params = sum(x.size for x in jax.tree.leaves(runner.params))
    print(f"run dir: {run_dir}")
    print(f"config: {json.dumps(dataclasses.asdict(cfg))}")
    print(
        f"{n_params:,} parameters | {sched.num_chunks} chunks of {sched.chunk_iters} iterations "
        f"x {cfg.steps_per_iteration:,} env steps = {sched.total_steps:,} env steps | "
        f"{sched.total_updates:,} gradient steps of {cfg.minibatch_size:,} agent samples"
    )

    t0 = time.perf_counter()
    train_chunk = (
        jax.jit(make_train_chunk(cfg, env, sim, sched.chunk_iters), donate_argnums=0)
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

        for chunk in range(int(runner.env_steps) // chunk_steps, sched.num_chunks):
            t = time.perf_counter()
            runner, m = train_chunk(runner)
            m = jax.device_get(m)
            seconds = time.perf_counter() - t
            record = chunk_record(m, seconds, chunk_steps)
            agent_total += record["agent_samples"]
            record["agent_samples_total"] = agent_total
            record["elapsed"] = time.perf_counter() - start
            write(record)
            print(format_record(record, sched.total_steps), flush=True)

            steps = record["env_steps"]
            last = chunk == sched.num_chunks - 1
            if last or crossed(cfg.eval_every, steps):
                save_checkpoint(run_dir, runner.params)
                if cfg.eval_games > 0:
                    k = jax.random.fold_in(k_eval, steps)
                    run_evals(
                        runner.params, cfg, env, k, cfg.eval_games, cfg.eval_opponents,
                        cfg.eval_modes, steps, write,
                    )  # fmt: skip
            # After the eval, so an interrupted eval is redone on resume.
            if last or crossed(cfg.checkpoint_every, steps):
                save_runner(run_dir, runner, agent_total)
    print(f"done in {time.perf_counter() - start:.0f}s; checkpoint in {run_dir}")
    return run_dir, runner.params


def evaluate_run(
    run_dir: str,
    num_games: int,
    seed: int = 0,
    opponents: tuple[str, ...] = PPOConfig.eval_opponents,
    modes: tuple[str, ...] = EVAL_MODES,
) -> list[dict[str, Any]]:
    """Load ``run_dir``'s checkpoint and play it against each opponent in each mode."""
    cfg = load_config(run_dir)
    env, _ = make_envs(cfg)
    params = load_params(run_dir, cfg)
    return run_evals(params, cfg, env, jax.random.key(seed), num_games, opponents, modes, -1)


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


def parse_args(argv: list[str] | None = None) -> tuple[PPOConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--eval-only",
        metavar="RUN_DIR",
        help="load RUN_DIR's checkpoint and play --eval-games games (with --seed) against "
        "each of --eval-opponents in each of --eval-modes instead of training",
    )
    p.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help="continue an interrupted run from its last full checkpoint, with its saved "
        "config (other flags are ignored)",
    )
    for f in dataclasses.fields(PPOConfig):
        default = f.default
        if isinstance(default, bool):
            parse: Callable[[str], Any] = _parse_bool
        elif isinstance(default, tuple):
            parse = _parse_tuple(type(default[0]))
        elif default is None:
            parse = str
        else:
            parse = type(default)
        shown = ",".join(map(str, default)) if isinstance(default, tuple) else default
        p.add_argument(
            "--" + f.name.replace("_", "-"), type=parse, default=default, help=f"(default: {shown})"
        )
    args = p.parse_args(argv)
    cfg = PPOConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(PPOConfig)})
    return cfg, args


def main(argv: list[str] | None = None) -> None:
    cfg, args = parse_args(argv)
    if args.eval_only:
        records = evaluate_run(
            args.eval_only, cfg.eval_games, cfg.seed, cfg.eval_opponents, cfg.eval_modes
        )
        for record in records:
            print(json.dumps({k: v for k, v in record.items() if k != "env_steps"}))
        return
    if args.resume:
        train(dataclasses.replace(load_config(args.resume), run_dir=args.resume), resume=True)
        return
    train(cfg)


if __name__ == "__main__":
    main()
