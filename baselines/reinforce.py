"""Self-play REINFORCE baseline for the 1v1 duel (single file, cleanRL / PureJaxRL style).

* **Self-play with parameter sharing.** One policy network (with a value head
  for the baseline) acts for both snakes, each from its own egocentric
  observation. Illegal moves (``env.action_mask``) are masked out of the policy
  everywhere: in sampling, in the log-probabilities and in the entropy.
* **Whole games, Monte Carlo returns.** Each update plays ``num_envs`` fresh
  games to the end (one jitted ``lax.while_loop`` that stops once every game is
  over), then takes one gradient step on every move of every game, weighted by
  its return ``G_t = r_t + gamma * G_{t+1}``. Nothing is bootstrapped, cut off
  mid-game or reused: this is Williams' (1992) REINFORCE. A game cut off by
  ``max_turns`` (rare) ends with return 0, like a draw.
* **Baseline.** With ``--baseline value`` (the default) a value head regresses
  ``G_t`` and the policy gradient uses ``G_t - V(s_t)``, which lowers its
  variance without biasing it. ``--baseline none`` is plain REINFORCE.
* **Compact storage.** The games are kept as :class:`~slinky.types.State`
  (under 1 KB per game turn) instead of observations (23 KB per snake). The
  update recomputes the observations ``block_turns`` turns at a time and skips
  the blocks after the longest game.
* **Learning.** Adam with gradient clipping and an entropy bonus; one gradient
  step per batch of games, on the mean over all of their moves.

The reward is the environment's sparse win/loss reward (no shaping), as in the
DQN and PPO baselines; the network is PPO's. Everything between two log lines
(``updates_per_log`` updates: playing and learning) is one jitted ``lax.scan``
(:func:`make_train_chunk`). The Python loop only logs, evaluates (greedy and
sampled play against ``random_legal``, the heuristic and the DQN baseline) and
saves checkpoints::

    python baselines/reinforce.py                               # default run
    python baselines/reinforce.py --baseline none --lr 3e-4     # plain REINFORCE
    python baselines/reinforce.py --eval-only runs/reinforce-20261010-120000
    python baselines/reinforce.py --resume runs/reinforce-20261010-120000

An *env step* is one game advancing one turn, as in the other baselines. It
gives one *agent sample* per living snake. A batch of games runs until its
longest game ends, so the shorter games idle in the meantime; the logs report
the share of game slots that were playing (``busy``).
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
BASELINES = ("value", "none")
# Logit of an illegal move: exp() underflows to exactly 0, but unlike -inf it keeps
# p * log p and every gradient finite.
MASKED_LOGIT = -1e9


@dataclasses.dataclass(frozen=True)
class ReinforceConfig:
    """All hyperparameters. Every field can be overridden on the command line.

    One update plays ``num_envs`` whole games, so its env steps vary with the
    length of the games. Training stops after the first log chunk that reaches
    ``total_env_steps``.
    """

    # Game: the 11x11 standard duel, truncated after max_turns turns.
    max_turns: int = 1000
    # Collection.
    num_envs: int = 8  # games per update
    total_env_steps: int = 5_242_880  # PPO's budget
    # REINFORCE. From short pilots (one seed each; see baselines/README.md):
    # the policy stayed uniform for its first ~230k env steps, lr 3e-3 started
    # learning sooner than 1e-3, and without an entropy bonus it collapsed.
    # gamma 0.95 concentrates the credit on the last ~20 moves, like PPO's
    # GAE(0.95); the pilots were too short to compare it with 0.99.
    gamma: float = 0.95
    baseline: str = "value"  # "value": G_t - V(s_t); "none": plain G_t
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    # Optimization.
    lr: float = 3e-3
    max_grad_norm: float = 0.5
    adam_eps: float = 1e-5
    # Network (PPO's): 3x3 convs (ReLU), a dense hidden layer (ReLU), then linear
    # policy and value heads.
    conv_channels: tuple[int, ...] = (16, 32, 32)
    conv_strides: tuple[int, ...] = (2, 2, 1)
    hidden: int = 256
    # Turns per block when recomputing observations in the update (memory vs. the
    # work wasted on the turns after the longest game).
    block_turns: int = 50
    # Logging, evaluation and checkpoints.
    seed: int = 0
    updates_per_log: int = 32  # updates per jitted chunk (one log line)
    eval_every: int = 1_048_576  # env steps between evaluations (and checkpoints)
    eval_games: int = 256  # per opponent and mode
    eval_opponents: tuple[str, ...] = ("random_legal", "heuristic", "dqn")  # slinky.agents
    eval_modes: tuple[str, ...] = EVAL_MODES  # greedy (masked argmax) and/or sample
    checkpoint_every: int = 262_144  # env steps between full checkpoints (for --resume)
    run_dir: str | None = None  # default: runs/reinforce-<timestamp>

    def __post_init__(self) -> None:
        if len(self.conv_channels) != len(self.conv_strides):
            raise ValueError("conv_channels and conv_strides must have the same length")
        positive = (
            "max_turns", "num_envs", "total_env_steps", "hidden", "block_turns",
            "updates_per_log", "eval_every", "checkpoint_every",
        )  # fmt: skip
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.baseline not in BASELINES:
            raise ValueError(f"unknown baseline {self.baseline!r}; expected one of {BASELINES}")
        if bad := set(self.eval_modes) - set(EVAL_MODES):
            raise ValueError(f"unknown eval_modes {sorted(bad)}; expected {EVAL_MODES}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

    @property
    def game(self) -> GameConfig:
        return GameConfig(max_turns=self.max_turns)

    @property
    def num_blocks(self) -> int:
        return -(-self.max_turns // self.block_turns)

    @property
    def buffer_turns(self) -> int:
        """Turns stored per game: ``max_turns`` rounded up to whole blocks."""
        return self.num_blocks * self.block_turns


# --- Policy network (PPO's) ----------------------------------------------------------


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
    equivalent im2col form, which XLA:CPU runs much faster inside a ``lax.scan``
    (the same trick as ``baselines/dqn.py`` and ``baselines/ppo.py``).
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


def init_network(key: jax.Array, cfg: ReinforceConfig) -> Params:
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


def network(params: Params, obs: jax.Array, cfg: ReinforceConfig) -> tuple[jax.Array, jax.Array]:
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
    """Log-probabilities ``f32[..., 4]`` of the policy restricted to the legal moves."""
    return jax.nn.log_softmax(jnp.where(mask, logits, MASKED_LOGIT), axis=-1)


def masked_entropy(log_probs: jax.Array, mask: jax.Array) -> jax.Array:
    """Entropy ``f32[...]`` of the masked policy (illegal moves contribute nothing)."""
    return -jnp.sum(jnp.where(mask, jnp.exp(log_probs) * log_probs, 0.0), axis=-1)


def sample_actions(key: jax.Array, logits: jax.Array, mask: jax.Array) -> jax.Array:
    """One legal move per row of ``logits``/``mask`` (``[..., 4]``), sampled from the policy."""
    return jax.random.categorical(key, jnp.where(mask, logits, MASKED_LOGIT)).astype(jnp.int32)


def make_policy(params: Params, cfg: ReinforceConfig, greedy: bool) -> Policy:
    """An ``evaluate.Policy`` for the network: greedy (masked argmax) or sampled.

    The parameters are baked into whatever traces the policy (e.g. a compiled
    ``play_match``), so a new call compiles again.
    """

    def logits_fn(obs: jax.Array) -> jax.Array:
        return network(params, obs, cfg)[0]

    if greedy:
        return greedy_from_q(logits_fn)

    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        return sample_actions(key, logits_fn(ts.obs), ts.action_mask)

    return policy


# --- Playing whole games --------------------------------------------------------------


class Games(NamedTuple):
    """A batch of whole games. Per-turn leaves are ``[T, E, ...]``, ``T = buffer_turns``.

    Turn ``t`` of game ``e`` goes from ``states[t, e]`` to ``states[t + 1, e]``.
    Entries after a game has ended are padding (``valid`` False, reward 0).
    """

    states: State  # [T, E] the state each move was chosen in
    actions: jax.Array  # int32[T, E, N]
    rewards: jax.Array  # f32[T, E, N]
    valid: jax.Array  # bool[T, E, N] the snake was alive in a running game: it moved
    final: State  # [E] the final states
    truncated: jax.Array  # bool[E] the game was cut off by max_turns
    turns: jax.Array  # int32[] loop iterations: the length of the longest game


def make_play(
    cfg: ReinforceConfig, env: BattlesnakeEnv, sim: BattlesnakeEnv
) -> Callable[[Params, jax.Array], Games]:
    """``play(params, key) -> Games``: ``num_envs`` fresh self-play games, played to the end.

    The loop stops when every game is over; the env ends each game by
    ``max_turns`` at the latest. A finished game stays as it ended (stepping it
    is a no-op with zero reward) until the others finish.
    """
    observe = jax.vmap(env.observe)  # [E] states -> [E, N, rows, cols, C]
    masks = jax.vmap(env.action_mask)  # [E] states -> [E, N, 4]
    num_envs, num_turns = cfg.num_envs, cfg.buffer_turns

    def play(params: Params, key: jax.Array) -> Games:
        k_init, k_loop = jax.random.split(key)
        states = jax.vmap(sim.init_state)(jax.random.split(k_init, num_envs))
        n = states.alive.shape[-1]

        def buffer(x: jax.Array) -> jax.Array:
            return jnp.zeros((num_turns, *x.shape), x.dtype)

        empty = Games(
            states=jax.tree.map(buffer, states),
            actions=jnp.zeros((num_turns, num_envs, n), jnp.int32),
            rewards=jnp.zeros((num_turns, num_envs, n), jnp.float32),
            valid=jnp.zeros((num_turns, num_envs, n), bool),
            final=states,
            truncated=jnp.zeros(num_envs, bool),
            turns=jnp.zeros((), jnp.int32),
        )

        def cond(carry):
            t, states, _ = carry
            return (t < num_turns) & ~jnp.all(states.done)

        def body(carry):
            t, states, games = carry
            k_act, k_step = jax.random.split(jax.random.fold_in(k_loop, t))
            logits, _ = network(params, observe(states), cfg)
            actions = sample_actions(k_act, logits, masks(states))
            new, ts = jax.vmap(sim.step)(jax.random.split(k_step, num_envs), states, actions)

            def put(buf: jax.Array, x: jax.Array) -> jax.Array:
                return jax.lax.dynamic_update_index_in_dim(buf, x, t, 0)

            games = games._replace(
                states=jax.tree.map(put, games.states, states),
                actions=put(games.actions, actions),
                rewards=put(games.rewards, ts.reward),
                valid=put(games.valid, states.alive & ~states.done[:, None]),
                truncated=games.truncated | ts.truncated,
            )
            return t + 1, new, games

        t, final, games = jax.lax.while_loop(cond, body, (jnp.int32(0), states, empty))
        return games._replace(final=final, turns=t)

    return play


def discounted_returns(rewards: jax.Array, gamma: float) -> jax.Array:
    """``G_t = r_t + gamma * G_{t+1}`` along axis 0 (``G`` is 0 after the last turn).

    Each game slot holds one game and a snake's reward ends with its last move,
    so the sum never crosses into another game or snake.
    """

    def step(following: jax.Array, r: jax.Array) -> tuple[jax.Array, jax.Array]:
        g = r + gamma * following
        return g, g

    _, returns = jax.lax.scan(step, jnp.zeros_like(rewards[0]), rewards, reverse=True)
    return returns


# --- Loss ------------------------------------------------------------------------------

# Sums over the valid samples of a block (see :func:`block_loss`).
SUMS = ("pg_loss", "v_loss", "entropy", "value", "value_sq", "ret", "ret_sq", "err", "err_sq")


def block_loss(
    params: Params,
    obs: jax.Array,
    mask: jax.Array,
    actions: jax.Array,
    returns: jax.Array,
    valid: jax.Array,
    count: jax.Array,
    cfg: ReinforceConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The loss of some samples, divided by ``count``, the valid samples of the whole batch.

    Summed over every block of a batch it is the batch's mean loss
    ``-(G - b) log pi(a|s) + vf_coef * (V - G)^2 / 2 - ent_coef * H``, with
    ``b = V(s)`` (no gradient through it) or 0. Also returns plain sums over the
    valid samples, for the logs.
    """
    logits, values = network(params, obs, cfg)
    log_probs_all = masked_log_probs(logits, mask)
    log_probs = jnp.take_along_axis(log_probs_all, actions[..., None], axis=-1)[..., 0]
    w = valid.astype(jnp.float32)
    use_value = cfg.baseline == "value"
    adv = returns - jax.lax.stop_gradient(values) if use_value else returns
    pg = -jnp.sum(w * adv * log_probs)
    err = returns - values
    v = 0.5 * jnp.sum(w * err**2)
    ent = jnp.sum(w * masked_entropy(log_probs_all, mask))
    loss = (pg + (cfg.vf_coef * v if use_value else 0.0) - cfg.ent_coef * ent) / count
    sums = {
        "pg_loss": pg,
        "v_loss": v,
        "entropy": ent,
        "value": jnp.sum(w * values),
        "value_sq": jnp.sum(w * values**2),
        "ret": jnp.sum(w * returns),
        "ret_sq": jnp.sum(w * returns**2),
        "err": jnp.sum(w * err),
        "err_sq": jnp.sum(w * err**2),
    }
    return loss, jax.lax.stop_gradient(sums)


def batch_gradient(
    params: Params, games: Games, cfg: ReinforceConfig, env: BattlesnakeEnv
) -> tuple[Params, dict[str, jax.Array]]:
    """``(gradient, sums)`` of the mean loss over every move of ``games``, block by block.

    Observations and masks are recomputed from the stored states, ``block_turns``
    turns at a time; the blocks after the longest game hold no moves and are
    skipped.
    """
    observe = jax.vmap(jax.vmap(env.observe))
    masks = jax.vmap(jax.vmap(env.action_mask))
    returns = discounted_returns(games.rewards, cfg.gamma)
    count = jnp.maximum(jnp.sum(games.valid), 1).astype(jnp.float32)
    grad_fn = jax.grad(block_loss, has_aux=True)
    zero_grads = jax.tree.map(jnp.zeros_like, params)
    zero_sums = {k: jnp.zeros((), jnp.float32) for k in SUMS}
    size = cfg.block_turns

    def block(carry, i):
        grads, sums = carry

        def compute():
            def cut(x: jax.Array) -> jax.Array:
                return jax.lax.dynamic_slice_in_dim(x, i * size, size, 0)

            states = jax.tree.map(cut, games.states)
            g, s = grad_fn(
                params, observe(states), masks(states), cut(games.actions), cut(returns),
                cut(games.valid), count, cfg,
            )  # fmt: skip
            return g, s

        g, s = jax.lax.cond(i * size < games.turns, compute, lambda: (zero_grads, zero_sums))
        return (jax.tree.map(jnp.add, grads, g), jax.tree.map(jnp.add, sums, s)), None

    (grads, sums), _ = jax.lax.scan(block, (zero_grads, zero_sums), jnp.arange(cfg.num_blocks))
    return grads, {**sums, "samples": count}


# --- Training ------------------------------------------------------------------------


class RunnerState(NamedTuple):
    """Everything the jitted training loop carries between chunks."""

    params: Params
    opt_state: Any
    key: jax.Array
    env_steps: jax.Array  # int32[] env steps taken (turns of running games)
    updates: jax.Array  # int32[] gradient steps taken


def make_envs(cfg: ReinforceConfig) -> tuple[BattlesnakeEnv, BattlesnakeEnv]:
    """``(env, sim)``: ``env`` computes egocentric observations, ``sim`` steps without them."""
    return BattlesnakeEnv(cfg.game, obs="egocentric"), BattlesnakeEnv(cfg.game, obs=None)


def make_optimizer(cfg: ReinforceConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr, eps=cfg.adam_eps),
    )


def init_runner(cfg: ReinforceConfig, key: jax.Array) -> RunnerState:
    k_net, k_run = jax.random.split(key)
    params = init_network(k_net, cfg)
    return RunnerState(
        params=params,
        opt_state=make_optimizer(cfg).init(params),
        key=k_run,
        env_steps=jnp.zeros((), jnp.int32),
        updates=jnp.zeros((), jnp.int32),
    )


def make_train_chunk(
    cfg: ReinforceConfig, env: BattlesnakeEnv, sim: BattlesnakeEnv, num_updates: int
) -> Callable[[RunnerState], tuple[RunnerState, dict[str, jax.Array]]]:
    """``train_chunk(runner) -> (runner, metrics)``: ``num_updates`` updates in one scan.

    Each update plays a batch of whole games and takes one gradient step on
    them. Metrics are summed over the chunk.
    """
    optimizer = make_optimizer(cfg)
    play = make_play(cfg, env, sim)

    def update(runner: RunnerState, _):
        key, k_play = jax.random.split(runner.key)
        games = play(runner.params, k_play)
        grads, sums = batch_gradient(runner.params, games, cfg, env)
        step, opt_state = optimizer.update(grads, runner.opt_state, runner.params)
        params = optax.apply_updates(runner.params, step)

        final = games.final
        turns = jnp.sum(final.turn)
        died = ~final.alive  # every snake that died, by its cause
        causes = jnp.arange(_NUM_CAUSES) == final.elim_cause[..., None]
        truncated = games.truncated
        metrics = {
            **sums,
            "grad_norm": optax.tree.norm(grads),
            "games": jnp.int32(cfg.num_envs),
            "game_turns": turns,
            "slot_turns": games.turns * cfg.num_envs,
            "max_game_len": games.turns,
            "draws": jnp.sum(~truncated & ~jnp.any(final.alive, axis=-1)),
            "truncated": jnp.sum(truncated),
            "deaths": jnp.sum(causes & died[..., None], axis=(0, 1)),
        }
        runner = RunnerState(
            params=params,
            opt_state=opt_state,
            key=key,
            env_steps=runner.env_steps + turns,
            updates=runner.updates + 1,
        )
        return runner, metrics

    def train_chunk(runner: RunnerState) -> tuple[RunnerState, dict[str, jax.Array]]:
        runner, metrics = jax.lax.scan(update, runner, None, length=num_updates)
        metrics = {
            k: jnp.max(v, 0) if k == "max_game_len" else jnp.sum(v, 0) for k, v in metrics.items()
        }
        metrics["env_steps"] = runner.env_steps
        metrics["updates"] = jnp.int32(num_updates)
        metrics["total_updates"] = runner.updates
        return runner, metrics

    return train_chunk


# --- Evaluation and checkpoints ---------------------------------------------------


def evaluate_params(
    params: Params,
    cfg: ReinforceConfig,
    env: BattlesnakeEnv,
    key: jax.Array,
    num_games: int,
    opponent: str = "heuristic",
    greedy: bool = True,
) -> MatchResult:
    """The network (greedy or sampled) vs a ``slinky.agents`` opponent, seats balanced."""
    from slinky.agents import make_agent  # the registry can import this file in turn

    # A private copy: the training loop donates (deletes) the runner's arrays.
    params = jax.tree.map(jnp.copy, params)
    policy = make_policy(params, cfg, greedy)
    other = make_agent(opponent, env.config).policy
    return play_match(env, policy, other, key, num_games, max_turns=cfg.max_turns)


def run_evals(
    params: Params,
    cfg: ReinforceConfig,
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


def save_config(run_dir: str, cfg: ReinforceConfig) -> None:
    """``config.json``, with ``"algorithm": "reinforce"`` (the dashboard and agents read it)."""
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"algorithm": "reinforce", **dataclasses.asdict(cfg)}, f, indent=2)


def load_config(run_dir: str) -> ReinforceConfig:
    with open(os.path.join(run_dir, "config.json")) as f:
        d = json.load(f)
    if d.get("algorithm") != "reinforce":
        raise ValueError(
            f"{run_dir} is not a REINFORCE run (config.json has no algorithm=reinforce)"
        )
    fields = {f.name for f in dataclasses.fields(ReinforceConfig)}
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in fields}
    return ReinforceConfig(**kwargs)


def load_params(run_dir: str, cfg: ReinforceConfig | None = None) -> Params:
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


def chunk_record(
    m: dict[str, Any], seconds: float, chunk_steps: int, value: bool = True
) -> dict[str, Any]:
    """A JSON-friendly summary of one chunk's metrics (means are over its agent samples).

    Without a value baseline (``value`` False) the value head is never trained,
    so its loss, mean and explained variance are logged as None.
    """
    games, n = int(m["games"]), float(m["samples"])

    def mean(k: str) -> float:
        return float(m[k]) / n

    def var(k: str) -> float:
        return max(mean(f"{k}_sq") - mean(k) ** 2, 0.0)

    var_ret = var("ret")
    ev = 1.0 - var("err") / var_ret if var_ret > 0 and value else None
    deaths = {name: int(m["deaths"][c]) for name, c in DEATH_CAUSES.items()}
    return {
        "type": "train",
        "env_steps": int(m["env_steps"]),
        "agent_samples": int(n),
        "total_updates": int(m["total_updates"]),
        "pg_loss": mean("pg_loss"),
        "v_loss": mean("v_loss") if value else None,
        "entropy": mean("entropy"),
        "grad_norm": float(m["grad_norm"]) / int(m["updates"]),
        "explained_var": ev,
        "value_mean": mean("value") if value else None,
        "return_mean": mean("ret"),
        "games": games,
        "mean_game_len": float(m["game_turns"]) / games if games else None,
        "max_game_len": int(m["max_game_len"]),
        "busy": float(m["game_turns"]) / max(float(m["slot_turns"]), 1.0),
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
    ev = f"{r['explained_var']:.2f}" if r["explained_var"] is not None else "-"
    value = (
        f"v {r['v_loss']:.4f} ent {r['entropy']:.3f} ev {ev} V {r['value_mean']:+.2f}"
        if r["v_loss"] is not None
        else f"ent {r['entropy']:.3f}"
    )
    return (
        f"steps {r['env_steps']:>9,} ({100 * r['env_steps'] / total_steps:3.0f}%) | "
        f"pg {r['pg_loss']:+.4f} {value} | games {r['games']:,} len {r['mean_game_len']:.0f} "
        f"max {r['max_game_len']} busy {r['busy']:.0%} draw {_pct(r['draws'], r['games'])} "
        "| deaths "
        + " ".join(f"{k} {_pct(deaths[k], n_dead)}" for k in shown)
        + f" | {r['env_steps_per_s']:,.0f} steps/s"
    )


# --- Main loop -----------------------------------------------------------------------


def train(cfg: ReinforceConfig, resume: bool = False) -> tuple[str, Params]:
    """Run a full training; returns ``(run_dir, final params)``.

    With ``resume``, continue from ``run_dir/runner.npz`` (written every
    ``checkpoint_every`` env steps) if it exists; on the same machine the run
    continues as if it had never stopped. A new run in an old directory first
    deletes that run's metrics and checkpoints.
    """
    run_dir = cfg.run_dir or os.path.join("runs", time.strftime("reinforce-%Y%m%d-%H%M%S"))
    cfg = dataclasses.replace(cfg, run_dir=run_dir)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    runner_path = os.path.join(run_dir, "runner.npz")

    env, sim = make_envs(cfg)
    if cfg.eval_games > 0:  # fail now, not at the first evaluation, if one can't be built
        from slinky.agents import make_agent

        for opponent in cfg.eval_opponents:
            make_agent(opponent, env.config)
    os.makedirs(run_dir, exist_ok=True)
    resuming = resume and os.path.exists(runner_path)
    if not resuming:  # a new run in an old directory: --resume must not find the old run
        for name in ("metrics.jsonl", "runner.npz", "params.npz"):
            if os.path.exists(path := os.path.join(run_dir, name)):
                os.remove(path)
    save_config(run_dir, cfg)
    k_eval, k_init = jax.random.split(jax.random.key(cfg.seed))
    runner = jax.jit(init_runner, static_argnums=0)(cfg, k_init)
    agent_total = 0
    if resuming:
        runner, agent_total = load_runner(run_dir, runner)
        _truncate_metrics(metrics_path, int(runner.env_steps))
        print(f"resuming from env step {int(runner.env_steps):,}")

    n_params = sum(x.size for x in jax.tree.leaves(runner.params))
    print(f"run dir: {run_dir}")
    print(f"config: {json.dumps(dataclasses.asdict(cfg))}")
    print(
        f"{n_params:,} parameters | {cfg.num_envs} whole games per update, "
        f"{cfg.updates_per_log} updates per log line | to {cfg.total_env_steps:,} env steps"
    )

    t0 = time.perf_counter()
    train_chunk = (
        jax.jit(make_train_chunk(cfg, env, sim, cfg.updates_per_log), donate_argnums=0)
        .lower(runner)
        .compile()
    )
    print(f"compiled train_chunk in {time.perf_counter() - t0:.1f}s", flush=True)

    start = time.perf_counter()
    with open(metrics_path, "a") as log:

        def write(record: dict[str, Any]) -> None:
            log.write(json.dumps(record) + "\n")
            log.flush()

        def crossed(every: int, before: int, after: int) -> bool:
            return after // every > before // every

        while (before := int(runner.env_steps)) < cfg.total_env_steps:
            t = time.perf_counter()
            runner, m = train_chunk(runner)
            m = jax.device_get(m)
            seconds = time.perf_counter() - t
            steps = int(m["env_steps"])
            record = chunk_record(m, seconds, steps - before, cfg.baseline == "value")
            agent_total += record["agent_samples"]
            record["agent_samples_total"] = agent_total
            record["elapsed"] = time.perf_counter() - start
            write(record)
            print(format_record(record, cfg.total_env_steps), flush=True)

            last = steps >= cfg.total_env_steps
            if last or crossed(cfg.eval_every, before, steps):
                save_checkpoint(run_dir, runner.params)
                if cfg.eval_games > 0:
                    k = jax.random.fold_in(k_eval, steps)
                    run_evals(
                        runner.params, cfg, env, k, cfg.eval_games, cfg.eval_opponents,
                        cfg.eval_modes, steps, write,
                    )  # fmt: skip
            # After the eval, so an interrupted eval is redone on resume.
            if last or crossed(cfg.checkpoint_every, before, steps):
                save_runner(run_dir, runner, agent_total)
    print(f"done in {time.perf_counter() - start:.0f}s; checkpoint in {run_dir}")
    return run_dir, runner.params


def evaluate_run(
    run_dir: str,
    num_games: int,
    seed: int = 0,
    opponents: tuple[str, ...] = ReinforceConfig.eval_opponents,
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


def parse_args(argv: list[str] | None = None) -> tuple[ReinforceConfig, argparse.Namespace]:
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
    for f in dataclasses.fields(ReinforceConfig):
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
    fields = dataclasses.fields(ReinforceConfig)
    cfg = ReinforceConfig(**{f.name: getattr(args, f.name) for f in fields})
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
