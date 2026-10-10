"""AlphaZero for the 1v1 duel: self-play with network-guided search (single file, all JAX).

* **One network, both snakes.** PPO's actor-critic torso (``baselines/ppo.py``:
  egocentric 21x21x13 observations, 3x3 convs, a dense layer) with a policy head
  (4 logits, illegal moves masked) and a ``tanh`` value head, evaluated for each
  snake from its own view. ``--init-from`` copies the torso and policy head of a
  PPO run (the value head too from an AZ run; PPO's was fit to discounted returns
  with draws worth 0, a different target).
* **Search.** :func:`slinky.mcts.search` with decoupled PUCT: each snake picks
  its own move by ``q + c_puct * P * sqrt(N) / (1 + n)`` with the network's
  priors and leaf values, one search for both snakes (see ``slinky.mcts``). The
  network enters as an evaluator closure over the *traced* parameters, so new
  parameters never recompile; under ``vmap`` over ``num_envs`` games each leaf
  is one network call on ``2 * num_envs`` observations.
* **Self-play.** Every turn searches every game (with Dirichlet noise in the
  root priors), samples each snake's move from its root visit share before
  ``temperature_turns``, then plays the search's own (most visited) move,
  steps with the real food spawns and starts a new game where one ended.
* **Targets.** A position gives one sample per living snake: the policy target
  is its root visit share and the value target its game outcome,
  ``heuristic.terminal_values`` of the final state with the search's
  ``draw_value`` (win +1, loss -1, a mutual elimination -0.5, a game cut off at
  ``max_turns`` 0): the scale of the tree's exact values. Values are per snake,
  not forced zero-sum, since draws are not. A snake certain to starve has its
  move fixed by the search, so it is left out of the policy loss.
* **Replay with late outcomes.** Positions are stored as compact
  :class:`~slinky.types.State` (under 1 KB, as in ``baselines/dqn.py``; the
  observations would be 46 KB) with both targets and the game's id. A game's
  outcome goes into a table when it ends; a sampled position of a game still
  running gets weight 0 (:class:`Replay` argues why the table never aliases).
  Sampling is uniform over the filled slots; ``replay_ratio`` sets the
  gradient steps per iteration (the expected uses of each position).
* **Augmentation.** A random one of the 8 board symmetries per sampled
  position. The rules, the observations, the legal masks and the heuristic
  commute with all 8 exactly (``tests/test_az.py`` checks it on real
  positions), so a mirrored position is a real position with the permuted
  policy target and the same outcome. The State is mirrored before the
  observations are built (one gather per grid, cheaper than mirroring 2 x 23 KB
  of observations).
* **Learning.** Cross-entropy to the visit shares plus ``value_coef`` times the
  value MSE, AdamW with gradient clipping and a linear learning-rate decay to
  ``final_lr_frac`` of ``lr``.
* **Expert iteration** (``teacher_env_steps``). The first env steps play the
  heuristic-leaf DUCT search (``mcts-<teacher_simulations>``, which scores
  0.715 against the heuristic at 256 simulations) instead of the network's,
  with the same sampling and targets, so the network starts from a strong
  teacher's visit distributions (Anthony et al. 2017).
* **Heuristic blend** (``blend_env_steps``). The leaf value
  ``lam * heuristic.evaluate + (1 - lam) * network``, lam falling linearly from
  ``blend_start`` to 0 over the first env steps of the network phase.

All of it runs on the device: a jitted chunk (:func:`make_train_chunk`) is a
``lax.scan`` of iterations, each a ``lax.scan`` of ``num_steps`` self-play turns
followed by a ``lax.scan`` of gradient steps, with the replay buffer updated in
place (no copies in the compiled program). The Python loop only logs, evaluates
(the network with search, ``az-<eval_simulations>``, and without, ``az-0``,
against ``slinky.agents`` opponents) and saves checkpoints. The teacher and the
network phase are two compiled chunks::

    python baselines/az.py                                     # default run
    python baselines/az.py --teacher-env-steps 1048576         # expert-iteration start
    python baselines/az.py --init-from baselines/checkpoints/ppo-duel-seed0
    python baselines/az.py --eval-only runs/az-20261010-120000 --eval-simulations 256
    python baselines/az.py --resume runs/az-20261010-120000    # after an interruption

The trained network plays as ``az-<n>[:<run dir>]`` in ``slinky.agents``
(``az-0``: the raw network). ``--resume`` is exact when ``replay.npz`` was
saved (``save_replay``, about 1 GB at the default capacity, rewritten at every
checkpoint) along with ``runner.npz``; without it, or if the two files come
from different checkpoints (a crash between the two writes), the run resumes
with an empty buffer and waits for ``learning_starts`` positions.

**Defaults and why.** Measured on the 4-core CPU of this repository's other
baselines, one process on cores 0-3, PPO's network:

* ``num_simulations=64``. The network is about 95% of the cost of a search
  (85-89 us per simulation against DUCT's 4.9; ``slinky.mcts``), so the budget
  per move is the main price of self-play. The design study recommended 64;
  fewer is faster but gives coarser visit-count targets.
  Self-play alone runs at about 390 env steps/s, and 315-345 with the default
  learning (16 gradient steps of 1024 samples per 2048 env steps, about 20% of
  the time). The teacher phase at 256 simulations runs at 380-430 env steps/s
  with learning (510-530 without). A 14-minute pilot with defaults otherwise
  (65,536 teacher env steps, then 131,072 network env steps, a 262,144-position
  buffer, 828 s with the evaluation) took the policy loss from 0.84 to 0.78
  and the value loss from 0.97 to 0.80 in the teacher phase; the network
  phase's broader targets (entropy 0.65 to 0.73) held them at 0.80 and 0.84.
  The raw network then scored 0.266 against the heuristic (PPO's 5.2M-step
  baseline: 0.296) and ``az-64`` 0.391, 0.328 against PPO and 0.305 against
  ``mcts-64`` (64 games each).
* ``c_puct=0.5``, ``fpu_reduction=0.2``: ``slinky.mcts``'s PUCT defaults. The
  search maps values to ``[0, 1]``, so 0.5 here is AlphaZero's 1.0 on its
  ``[-1, 1]`` scale (it used 1.25). Untuned with network priors.
* ``root_noise_frac=0.25`` (AlphaZero's), ``root_dirichlet_alpha=0.5``
  (``slinky.mcts``'s default for 2-3 legal moves; untuned).
* ``temperature_turns=16``: the opening varies (heuristic mirror games are
  over 90% draws, and opening head-on collisions happen at a median of turn 4),
  then the most visited move: sampling every move from visit counts scored
  0.127 against the heuristic, where the argmax scored 0.594 (``slinky.mcts``).
* ``draw_value=-0.5``: the search's contempt, which the targets must match.
* ``teacher_env_steps=0``: a plain AlphaZero run by default; the expert
  iteration start is one flag away, left to the pilots to measure.
* ``buffer_capacity=1_048_576`` positions, about 1 GB of device memory and
  about 55 minutes of self-play at 310 env steps/s (a quarter of the default
  run). ``batch_size=512`` positions (1024 samples), ``replay_ratio=4``,
  ``learning_starts=32_768`` (16 iterations, after most of the first games
  ended).
* ``lr=1e-3`` decaying to 1e-4, as PPO's 1e-3; ``weight_decay=1e-4`` (AdamW,
  scaled by the learning rate: mild); ``max_grad_norm=1``; ``value_coef=1``
  (AlphaZero's equal weights).
* ``total_env_steps=4_194_304``: about 4 hours with the evaluations at the
  default settings, the scale of the PPO baseline's 5.2M env steps.
* ``eval_games=128`` per opponent and mode every 1,048,576 env steps (about
  55 minutes): in the pilot, 64 games of ``az-64`` took 18 s against
  ``random_legal``, 59 against the heuristic, 66 against PPO and 80 against
  ``mcts-64`` (the raw network 3-13 s), so a default evaluation takes about
  8 minutes, 13% of the run.

On a GPU the same code runs unchanged; raise ``num_envs`` (the network's batch
is ``2 * num_envs`` observations per simulation) and keep ``replay_ratio``.
An *env step* is one game advancing one turn; it stores one position, worth
one agent sample per living snake (two in a duel). The file is self-contained:
it shares no code with ``baselines/ppo.py`` (the network and checkpoint
helpers follow it).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import itertools
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

from slinky import heuristic, rules
from slinky.env import BattlesnakeEnv
from slinky.evaluate import MatchResult, Policy, play_match
from slinky.mcts import Evaluator, MCTSConfig, SearchOutput, search
from slinky.observations import make_obs_fn, obs_shape
from slinky.types import ACTION_DELTAS, NUM_ACTIONS, Cause, GameConfig, State, TimeStep

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
# Evaluation modes: the network with search (``az-<eval_simulations>``) and without (``az-0``).
EVAL_MODES = ("search", "raw")
PHASES = ("teacher", "network")
# Logit of an illegal move: exp() underflows to exactly 0, but unlike -inf it keeps
# p * log p and every gradient finite.
MASKED_LOGIT = -1e9
# The search fields an ``az-<n>`` agent name may override (``slinky.agents``). The rest are
# fixed by what the network was trained for: decoupled PUCT, network priors, no root noise.
AGENT_SEARCH_FIELDS = (
    "c_puct", "fpu_reduction", "tie_noise", "leaf", "max_depth", "final", "draw_value",
    "spawn_food",
)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class AZConfig:
    """All hyperparameters. Every field can be overridden on the command line.

    Steps are *env steps* (games advanced one turn). One iteration plays
    ``num_steps`` turns in each of ``num_envs`` games (``num_envs * num_steps``
    env steps, one stored position each, worth one agent sample per living
    snake), then takes ``updates_per_iteration`` gradient steps.
    """

    # Game: the 11x11 standard duel, truncated after max_turns turns.
    max_turns: int = 1000
    # Self-play.
    num_envs: int = 64  # games searched and stepped in parallel
    num_steps: int = 32  # turns per iteration
    total_env_steps: int = 4_194_304  # rounded up to whole log chunks
    num_simulations: int = 64  # per move, both snakes in one search
    temperature_turns: int = 16  # sample moves from the root visits before this turn
    root_noise_frac: float = 0.25  # Dirichlet noise in the self-play root priors
    root_dirichlet_alpha: float = 0.5
    # Search constants (self-play, evaluation and the az agent).
    c_puct: float = 0.5
    fpu_reduction: float = 0.2
    draw_value: float = -0.5  # a mutual elimination, in the search and in the value targets
    # Expert iteration: the first env steps play the heuristic-leaf DUCT search instead.
    teacher_env_steps: int = 0  # rounded up to whole log chunks; 0 = no teacher phase
    teacher_simulations: int = 256
    # Heuristic value blend in the network phase (blend_env_steps = 0: off).
    blend_start: float = 1.0  # leaf = lam * heuristic + (1 - lam) * network, lam from here to 0
    blend_env_steps: int = 0
    # Replay: positions (one State and both snakes' targets each).
    buffer_capacity: int = 1_048_576  # rounded down to whole iterations
    learning_starts: int = 32_768  # positions in the buffer before the first update
    batch_size: int = 512  # positions per gradient step (num_snakes agent samples each)
    replay_ratio: float = 4.0  # expected times each stored position is sampled
    augment: bool = True  # a random board symmetry per sampled position
    # Optimization.
    lr: float = 1e-3
    final_lr_frac: float = 0.1  # linear decay to lr * final_lr_frac over the run
    weight_decay: float = 1e-4  # AdamW (decoupled, scaled by the learning rate)
    max_grad_norm: float = 1.0
    adam_eps: float = 1e-8
    value_coef: float = 1.0
    # Network: 3x3 convs (ReLU), a dense hidden layer (ReLU), a policy head and a tanh
    # value head. PPO's sizes, so --init-from can warm-start from a PPO run.
    conv_channels: tuple[int, ...] = (16, 32, 32)
    conv_strides: tuple[int, ...] = (2, 2, 1)
    hidden: int = 256
    init_from: str | None = None  # a PPO (or AZ) run dir with the same network sizes
    # Logging, evaluation and checkpoints.
    seed: int = 0
    log_every: int = 32_768  # env steps per jitted chunk (one log line), whole iterations
    eval_every: int = 1_048_576  # env steps between evaluations (and checkpoints)
    eval_games: int = 128  # per opponent and mode
    eval_opponents: tuple[str, ...] = ("random_legal", "heuristic", "ppo", "mcts-64")
    eval_modes: tuple[str, ...] = EVAL_MODES
    eval_simulations: int = 64
    checkpoint_every: int = 262_144  # env steps between full checkpoints (for --resume)
    save_replay: bool = True  # also checkpoint the replay buffer (exact --resume)
    run_dir: str | None = None  # default: runs/az-<timestamp>

    def __post_init__(self) -> None:
        if len(self.conv_channels) != len(self.conv_strides):
            raise ValueError("conv_channels and conv_strides must have the same length")
        positive = (
            "max_turns", "num_envs", "num_steps", "total_env_steps", "num_simulations",
            "teacher_simulations", "batch_size", "hidden", "log_every", "eval_every",
            "checkpoint_every", "eval_simulations",
        )  # fmt: skip
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("temperature_turns", "teacher_env_steps", "blend_env_steps", "eval_games"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.buffer_capacity < self.steps_per_iteration:
            raise ValueError(
                f"buffer_capacity must hold at least one iteration (num_envs * num_steps = "
                f"{self.steps_per_iteration} positions)"
            )
        if self.learning_starts > self.capacity:
            raise ValueError("learning_starts must not exceed the (rounded) buffer_capacity")
        if not 0.0 < self.replay_ratio < math.inf:
            raise ValueError("replay_ratio must be finite and > 0")
        for name in ("blend_start", "final_lr_frac"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if bad := set(self.eval_modes) - set(EVAL_MODES):
            raise ValueError(f"unknown eval_modes {sorted(bad)}; expected {EVAL_MODES}")
        # The search configs check c_puct, fpu_reduction, the noise and draw_value.
        selfplay_search(self)
        teacher_search(self)

    @property
    def game(self) -> GameConfig:
        return GameConfig(max_turns=self.max_turns)

    @property
    def steps_per_iteration(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def capacity(self) -> int:
        """Replay positions: ``buffer_capacity`` rounded down to whole iterations."""
        return self.buffer_capacity // self.steps_per_iteration * self.steps_per_iteration

    @property
    def updates_per_iteration(self) -> int:
        """Gradient steps per iteration: ``replay_ratio`` uses of each new position."""
        return max(round(self.replay_ratio * self.steps_per_iteration / self.batch_size), 1)

    @property
    def blend(self) -> bool:
        """Whether the network phase blends the heuristic into the leaf values."""
        return self.blend_env_steps > 0 and self.blend_start > 0


def selfplay_search(cfg: AZConfig) -> MCTSConfig:
    """The network phase's search: decoupled PUCT, network priors and leaves, root noise."""
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
    )


def teacher_search(cfg: AZConfig) -> MCTSConfig:
    """The teacher phase's search: the default heuristic-leaf DUCT (``mcts-<n>``)."""
    return MCTSConfig(num_simulations=cfg.teacher_simulations, draw_value=cfg.draw_value)


def agent_search(cfg: AZConfig, simulations: int, **overrides: Any) -> MCTSConfig:
    """The search of the trained network as a player (``az-<n>``): no noise, final max.

    ``overrides`` may set the fields in :data:`AGENT_SEARCH_FIELDS`.
    """
    if bad := set(overrides) - set(AGENT_SEARCH_FIELDS):
        raise ValueError(f"az search options {sorted(bad)} can't be set; known: "
                         f"{', '.join(AGENT_SEARCH_FIELDS)}")  # fmt: skip
    settings = dict(
        selection="puct",
        prior="network",
        leaf="network",
        c_puct=cfg.c_puct,
        fpu_reduction=cfg.fpu_reduction,
        draw_value=cfg.draw_value,
    )
    return MCTSConfig(num_simulations=simulations, **{**settings, **overrides})


class Schedule(NamedTuple):
    """How a run's env steps split into iterations, log chunks and phases."""

    chunk_iters: int  # iterations per jitted chunk
    chunk_steps: int  # env steps per chunk
    num_chunks: int
    total_steps: int  # env steps of the whole run (total_env_steps rounded up)
    teacher_chunks: int  # the first chunks play the teacher's search
    total_updates: int  # gradient steps of the whole run (the learning-rate decay's length)

    @property
    def teacher_steps(self) -> int:
        return self.teacher_chunks * self.chunk_steps

    def phase(self, chunk: int) -> str:
        return PHASES[0] if chunk < self.teacher_chunks else PHASES[1]


def schedule(cfg: AZConfig) -> Schedule:
    chunk_iters = max(cfg.log_every // cfg.steps_per_iteration, 1)
    chunk_steps = chunk_iters * cfg.steps_per_iteration
    num_chunks = -(-cfg.total_env_steps // chunk_steps)
    teacher_chunks = min(-(-cfg.teacher_env_steps // chunk_steps), num_chunks)
    # Iteration i (from 1) stores i * steps_per_iteration positions; the first that
    # reaches learning_starts also learns. Counting the iterations before it too, the
    # learning rate never reached its floor (1.07e-4 instead of 1e-4 at the defaults).
    first_learning = max(-(-cfg.learning_starts // cfg.steps_per_iteration), 1)
    learning_iters = max(num_chunks * chunk_iters - (first_learning - 1), 0)
    updates = learning_iters * cfg.updates_per_iteration
    return Schedule(
        chunk_iters, chunk_steps, num_chunks, num_chunks * chunk_steps, teacher_chunks, updates
    )


# --- Network -------------------------------------------------------------------------


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
    outside one; matmuls don't suffer (the same trick as ``baselines/ppo.py``).
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


def init_network(key: jax.Array, cfg: AZConfig) -> Params:
    """He-normal torso, a near-zero policy head (uniform priors at first), a small value head."""
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
    params["value"] = layer((cfg.hidden, 1), cfg.hidden, gain=1e-2)  # tanh starts linear, near 0
    return params


def network(params: Params, obs: jax.Array, cfg: AZConfig) -> tuple[jax.Array, jax.Array]:
    """``(logits f32[..., 4], value f32[...])`` for observations ``f32[..., rows, cols, C]``.

    The value is ``tanh`` of the value head, in ``(-1, 1)`` like the search's
    values. The logits are unmasked; see :func:`masked_log_probs`.
    """
    lead = obs.shape[:-3]
    x = obs.reshape(-1, *obs.shape[-3:])
    for i, stride in enumerate(cfg.conv_strides):
        p = params[f"conv{i}"]
        x = jax.nn.relu(conv3x3(x, p["w"], stride) + p["b"])
    x = x.reshape(x.shape[0], -1)
    x = jax.nn.relu(x @ params["hidden"]["w"] + params["hidden"]["b"])
    logits = x @ params["policy"]["w"] + params["policy"]["b"]
    value = jnp.tanh(x @ params["value"]["w"] + params["value"]["b"])
    return logits.reshape(*lead, NUM_ACTIONS), value.reshape(lead)


def masked_log_probs(logits: jax.Array, mask: jax.Array) -> jax.Array:
    """Log-probabilities ``f32[..., 4]`` of the policy restricted to the legal moves."""
    return jax.nn.log_softmax(jnp.where(mask, logits, MASKED_LOGIT), axis=-1)


def masked_entropy(log_probs: jax.Array, mask: jax.Array) -> jax.Array:
    """Entropy ``f32[...]`` of the masked policy (illegal moves contribute nothing)."""
    return -jnp.sum(jnp.where(mask, jnp.exp(log_probs) * log_probs, 0.0), axis=-1)


def init_from_run(params: Params, cfg: AZConfig, run_dir: str) -> Params:
    """``params`` with the torso and policy head of ``run_dir``'s checkpoint.

    ``run_dir`` is a PPO run (``baselines/ppo.py``: the same layers and parameter
    names) or an AZ run with the same ``conv_channels``, ``conv_strides`` and
    ``hidden``. The value head is copied only from an AZ run: PPO's head is
    linear and was fit to discounted returns with draws worth 0, a different
    target from the game outcome with ``draw_value`` under ``tanh``.
    """
    with open(os.path.join(run_dir, "config.json")) as f:
        source = json.load(f)
    for name in ("conv_channels", "conv_strides", "hidden"):
        theirs = tuple(source[name]) if isinstance(source[name], list) else source[name]
        if theirs != getattr(cfg, name):
            raise ValueError(f"--init-from {run_dir}: {name} {theirs} != {getattr(cfg, name)}")
    layers = [k for k in params if k != "value" or source.get("algorithm") == "az"]
    out = dict(params)
    with np.load(os.path.join(run_dir, "params.npz")) as f:
        for layer in layers:
            out[layer] = {}
            for name, ref in params[layer].items():
                if f"{layer}/{name}" not in f.files:
                    raise ValueError(f"--init-from {run_dir}: no {layer}/{name} in params.npz")
                leaf = f[f"{layer}/{name}"]
                if leaf.shape != ref.shape:
                    raise ValueError(f"--init-from: {layer}/{name} {leaf.shape} != {ref.shape}")
                out[layer][name] = jnp.asarray(leaf, ref.dtype)
    return out


# --- Board symmetries ------------------------------------------------------------------

# A symmetry acts on cells (x, y): first swap x and y if ``transpose``, then mirror x if
# ``flip_x`` and y if ``flip_y``. Index ``4 * transpose + 2 * flip_x + flip_y``; 0 is the
# identity. Without the transposes (indices 0-3) it also works on non-square boards.
SYMMETRIES = tuple(itertools.product((False, True), repeat=3))


def action_permutation(sym: int) -> np.ndarray:
    """int32[4]: ``perm[a]`` is the move that ``a`` becomes on the transformed board.

    Flipping y swaps UP and DOWN (``[1, 0, 2, 3]``), flipping x swaps LEFT and RIGHT
    (``[0, 1, 3, 2]``), and the transpose maps UP to RIGHT and DOWN to LEFT
    (``[3, 2, 1, 0]``).
    """
    transpose, flip_x, flip_y = SYMMETRIES[sym]
    out = []
    for dx, dy in ACTION_DELTAS:
        if transpose:
            dx, dy = dy, dx
        out.append(ACTION_DELTAS.index((-dx if flip_x else dx, -dy if flip_y else dy)))
    return np.array(out, np.int32)


class SymmetryTables(NamedTuple):
    """Lookup tables of the board symmetries of one board size (``S`` symmetries)."""

    flags: np.ndarray  # bool[S, 3]: (transpose, flip_x, flip_y)
    perm: np.ndarray  # int32[S, 4]: action_permutation
    inverse: np.ndarray  # int32[S, 4]: its inverse; a [..., 4] move array maps as x[..., inverse]
    source: np.ndarray  # int32[S, H * W]: flat source cell of every flat target cell


@functools.cache
def symmetry_tables(width: int, height: int) -> SymmetryTables:
    """All 8 symmetries on a square board; only the 4 flips (no transpose) otherwise."""
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
    """The game ``state`` (one unbatched game) mirrored/rotated by symmetry ``sym``.

    Grids are permuted cell by cell, heads mapped like cells and ``last_move``
    like moves. ``sym`` may be traced (one gather per grid). The rules, the
    observations and the heuristic all commute with it exactly (tests check it
    on real positions), so a transformed position is a real position whose
    policy target is the permuted target and whose outcome is the same.
    """
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
    """A per-move array ``x[..., 4]`` (policy, mask) of a position, moved with the position."""
    inverse = jnp.asarray(symmetry_tables(game.width, game.height).inverse)[sym]
    return jnp.take(x, inverse, axis=-1)


def transform_obs(obs: jax.Array, sym: int) -> jax.Array:
    """An egocentric observation ``[..., rows, cols, C]`` under symmetry ``sym`` (static).

    The head sits at the centre of the window, so the symmetry acts on the
    window's own axes. :func:`transform_state` followed by the observation
    gives exactly this (tested); training transforms states, which is cheaper.
    """
    transpose, flip_x, flip_y = SYMMETRIES[sym]
    if transpose:
        obs = jnp.swapaxes(obs, -3, -2)
    if flip_x:
        obs = obs[..., :, ::-1, :]
    if flip_y:
        obs = obs[..., ::-1, :, :]
    return obs


# --- Search and policies ---------------------------------------------------------------


def make_evaluator(
    params: Params, cfg: AZConfig, game: GameConfig, blend: jax.Array | None = None
) -> Evaluator:
    """``slinky.mcts`` evaluator: ``state -> (value f32[N], logits f32[N, 4])`` of one game.

    It builds both snakes' egocentric observations from the state itself.
    ``params`` and ``blend`` may be traced. With ``blend`` (lam), the value is
    ``lam * heuristic.evaluate + (1 - lam) * network``.
    """
    obs_fn = make_obs_fn(game)

    def evaluator(state: State) -> tuple[jax.Array, jax.Array]:
        logits, value = network(params, obs_fn(state), cfg)
        if blend is not None:
            value = blend * heuristic.evaluate(state, game) + (1.0 - blend) * value
        return value, logits

    return evaluator


def make_policy(
    params: Params,
    cfg: AZConfig,
    game: GameConfig,
    simulations: int,
    search_config: MCTSConfig | None = None,
) -> Policy:
    """An ``evaluate.Policy`` for the network, built from the state alone (no observations).

    ``simulations == 0`` is the raw network: the legal argmax of its policy
    logits. Otherwise it plays the search's move with ``search_config``
    (default :func:`agent_search`). The parameters are baked into whatever
    traces the policy (e.g. a compiled ``play_match``), so a new call compiles
    again.
    """
    env = BattlesnakeEnv(game, obs=None)
    if simulations == 0:
        obs_fn = make_obs_fn(game)

        def raw(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
            logits = network(params, obs_fn(state), cfg)[0]
            masked = jnp.where(env.action_mask(state), logits, -jnp.inf)
            return jnp.argmax(masked, axis=-1).astype(jnp.int32)

        return raw
    search_config = search_config or agent_search(cfg, simulations)
    if search_config.num_simulations != simulations:
        raise ValueError("search_config.num_simulations != simulations")
    evaluator = make_evaluator(params, cfg, game) if search_config.uses_evaluator else None

    def searched(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        return search(key, state, env, search_config, evaluator).action

    return searched


# --- Replay buffer with outcomes that arrive later --------------------------------------


class Replay(NamedTuple):
    """Positions of the last ``C`` env steps and the outcomes of finished games.

    A position stores the game :class:`State` (under 1 KB), both snakes' root
    visit distributions and the id of its game. Observations, legal masks and
    sample weights are recomputed when sampling (as ``baselines/dqn.py`` does).

    The outcome of a game is only known when it ends, many iterations after
    its first positions were stored, so it goes to an outcome table instead:
    row ``uid % C``, tagged with the uid. A position is a valid sample iff its
    row holds its own game's uid. Uids are ``k * num_envs + e`` for the
    ``k``-th game of env slot ``e``, and ``C`` is a multiple of ``num_envs``,
    so a row is shared only by games of one slot that are ``C / num_envs``
    games apart. That slot writes one position per turn, the buffer wraps
    every ``C / num_envs`` turns, and each game lasts at least one turn, so by
    the time game ``u + C`` ends and overwrites row ``u % C``, every position
    of game ``u`` has been overwritten too: a lookup finds its own outcome or,
    while its game is running, an older game's uid (or -1), never a stranger's.
    """

    state: State  # [C] positions (s_t of a running game)
    policy: jax.Array  # f32[C, N, 4] each snake's root visit share at s_t
    uid: jax.Array  # int32[C] the game of each position
    outcome_uid: jax.Array  # int32[C] the game whose outcome row r holds (-1: none yet)
    outcome: jax.Array  # f32[C, N] that game's outcome per snake
    ptr: jax.Array  # int32[] next write slot
    size: jax.Array  # int32[] filled slots


def replay_init(example: State, capacity: int) -> Replay:
    """An empty buffer for positions like ``example`` (one unbatched state)."""
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
    """Write ``K`` positions at ``ptr`` (one contiguous in-place update; ``K`` divides ``C``)."""
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
    """Record ``outcome[K, N]`` for the games ``uid[K]`` that ended (``done[K]``)."""
    capacity = replay.uid.shape[0]
    rows = jnp.where(done, uid % capacity, capacity)  # rows of running games are dropped
    return replay._replace(
        outcome_uid=replay.outcome_uid.at[rows].set(uid, mode="drop"),
        outcome=replay.outcome.at[rows].set(outcome, mode="drop"),
    )


def replay_lookup(replay: Replay, idx: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(finished bool[B], outcome f32[B, N])`` of the positions ``idx[B]``."""
    uid = replay.uid[idx]
    row = uid % replay.uid.shape[0]
    return replay.outcome_uid[row] == uid, replay.outcome[row]


class Batch(NamedTuple):
    """A minibatch of ``B`` positions; per-snake fields are ``[B, N, ...]``."""

    obs: jax.Array  # f32[B, N, rows, cols, C]
    legal: jax.Array  # bool[B, N, 4] env.action_mask
    target: jax.Array  # f32[B, N, 4] root visit share
    z: jax.Array  # f32[B, N] game outcome (meaningful where finished)
    w_policy: jax.Array  # bool[B, N] in the policy loss
    w_value: jax.Array  # bool[B, N] in the value loss
    finished: jax.Array  # bool[B] the position's game has ended


def sample_batch(replay: Replay, key: jax.Array, cfg: AZConfig, env: BattlesnakeEnv) -> Batch:
    """``cfg.batch_size`` positions drawn uniformly (with replacement) from the filled slots.

    Each position gives one sample per snake. The value loss counts the snakes
    alive at the position, and the policy loss those of them that had a
    choice (not certain to starve, whose move the search fixes); positions of
    running games count in neither (weights 0). With ``cfg.augment`` each
    position is first moved by a random board symmetry.
    """
    k_idx, k_sym = jax.random.split(key)
    game = env.config
    idx = jax.random.randint(k_idx, (cfg.batch_size,), 0, jnp.maximum(replay.size, 1))
    state = jax.tree.map(lambda x: x[idx], replay.state)
    target = replay.policy[idx]
    finished, z = replay_lookup(replay, idx)
    if cfg.augment:
        count = symmetry_tables(game.width, game.height).perm.shape[0]
        sym = jax.random.randint(k_sym, (cfg.batch_size,), 0, count)
        state = jax.vmap(lambda s, g: transform_state(s, g, game))(state, sym)
        target = jax.vmap(lambda t, g: transform_moves(t, g, game))(target, sym)
    obs = jax.vmap(make_obs_fn(game))(state)
    legal = jax.vmap(env.action_mask)(state)
    starving = jax.vmap(lambda s: rules.certainly_starving(s, game))(state)
    w_value = finished[:, None] & state.alive
    return Batch(obs, legal, target, z, w_value & ~starving, w_value, finished)


# --- Loss ------------------------------------------------------------------------------

LOSS_METRICS = (
    "loss", "policy_loss", "value_loss", "target_entropy", "net_entropy", "value_abs_mean",
    "value_target_var", "masked_frac",
)  # fmt: skip


def az_loss(params: Params, batch: Batch, cfg: AZConfig) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Cross-entropy to the search policy plus ``value_coef`` times the value MSE.

    Means over the samples that count (:class:`Batch` weights). Also returns
    diagnostics: the entropies of the targets and of the network's policy,
    the mean ``|value|``, the variance of the value targets (the log's
    explained variance is ``1 - value_loss / value_target_var``) and the
    fraction of positions whose game had not ended (masked out). The mean
    value and mean target would say nothing in a duel: outcomes are zero-sum
    but for draws, so both stay near 0 (-0.005 throughout a smoke run).
    """
    logits, value = network(params, batch.obs, cfg)
    log_probs = masked_log_probs(logits, batch.legal)
    target = jnp.where(batch.legal, batch.target, 0.0)
    cross_entropy = -jnp.sum(target * log_probs, axis=-1)
    wp, wv = batch.w_policy.astype(jnp.float32), batch.w_value.astype(jnp.float32)

    def mean(x: jax.Array, w: jax.Array) -> jax.Array:
        return jnp.sum(w * x) / jnp.maximum(jnp.sum(w), 1.0)

    policy_loss = mean(cross_entropy, wp)
    value_loss = mean((value - batch.z) ** 2, wv)
    loss = policy_loss + cfg.value_coef * value_loss
    target_log = jnp.log(jnp.where(target > 0, target, 1.0))
    z_mean = mean(batch.z, wv)
    parts = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "target_entropy": mean(-jnp.sum(target * target_log, axis=-1), wp),
        "net_entropy": mean(masked_entropy(log_probs, batch.legal), wp),
        "value_abs_mean": mean(jnp.abs(value), wv),
        "value_target_var": mean((batch.z - z_mean) ** 2, wv),
        "masked_frac": 1.0 - jnp.mean(batch.finished.astype(jnp.float32)),
    }
    return loss, parts


# --- Training --------------------------------------------------------------------------


class RunnerState(NamedTuple):
    """Everything the jitted training loop carries between chunks."""

    params: Params
    opt_state: Any
    env_states: State  # [num_envs] running games (never done)
    uids: jax.Array  # int32[num_envs] id of each running game (see Replay)
    key: jax.Array
    # int32[] iterations taken. The host counts env steps (iterations *
    # steps_per_iteration, a Python int): an int32 env-step counter would wrap after
    # 2.1B env steps, about a day of self-play at 20k env steps/s on a GPU.
    iterations: jax.Array
    updates: jax.Array  # int32[] gradient steps taken
    replay: Replay


class Positions(NamedTuple):
    """What a self-play turn records per game (leaves ``[E, ...]``, stacked ``[T, E, ...]``)."""

    state: State  # s_t
    policy: jax.Array  # f32[E, N, 4] root visit shares
    uid: jax.Array  # int32[E]
    done: jax.Array  # bool[E] the game ended at t + 1 (rules or max_turns)
    outcome: jax.Array  # f32[E, N] terminal values of s_{t+1} (used where done)


def make_env(cfg: AZConfig) -> BattlesnakeEnv:
    """The env for self-play, without observations (the network builds its own)."""
    return BattlesnakeEnv(cfg.game, obs=None)


def learning_rate(cfg: AZConfig) -> optax.Schedule:
    """A linear decay from ``lr`` to ``lr * final_lr_frac`` over the run's gradient steps."""
    return optax.linear_schedule(cfg.lr, cfg.lr * cfg.final_lr_frac, schedule(cfg).total_updates)


def make_optimizer(cfg: AZConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adamw(learning_rate(cfg), eps=cfg.adam_eps, weight_decay=cfg.weight_decay),
    )


def init_runner(cfg: AZConfig, env: BattlesnakeEnv, key: jax.Array) -> RunnerState:
    k_net, k_env, k_run = jax.random.split(key, 3)
    params = init_network(k_net, cfg)
    env_states = jax.vmap(env.init_state)(jax.random.split(k_env, cfg.num_envs))
    return RunnerState(
        params=params,
        opt_state=make_optimizer(cfg).init(params),
        env_states=env_states,
        uids=jnp.arange(cfg.num_envs, dtype=jnp.int32),
        key=k_run,
        iterations=jnp.zeros((), jnp.int32),
        updates=jnp.zeros((), jnp.int32),
        replay=replay_init(jax.tree.map(lambda x: x[0], env_states), cfg.capacity),
    )


def blend_at(cfg: AZConfig, iterations: jax.Array, teacher_steps: int) -> jax.Array | None:
    """The heuristic's weight lam in the leaf value (None when the blend is off).

    It falls linearly from ``blend_start`` to 0 over the first
    ``blend_env_steps`` env steps of the network phase, which starts after
    ``teacher_steps`` env steps (whole iterations). ``iterations`` is
    :attr:`RunnerState.iterations`.
    """
    if not cfg.blend:
        return None
    network_iters = iterations - teacher_steps // cfg.steps_per_iteration
    steps = network_iters.astype(jnp.float32) * cfg.steps_per_iteration
    progress = steps / cfg.blend_env_steps
    return cfg.blend_start * jnp.clip(1.0 - progress, 0.0, 1.0)


def make_selfplay(
    cfg: AZConfig, env: BattlesnakeEnv, phase: str
) -> Callable[..., tuple[tuple[State, jax.Array], Positions, dict[str, jax.Array]]]:
    """``selfplay(params, blend, states, uids, key) -> ((states, uids), Positions, metrics)``.

    ``num_steps`` turns in every game: search both snakes at once, sample
    each snake's move from its root visit share before ``temperature_turns``
    (the search's own move after), step with the real food spawns, record the
    position, and start a new game where one ended. The teacher phase searches
    with the heuristic-leaf DUCT (``params`` unused), the network phase with
    the network's priors and values. Metrics are summed over the turns.
    """
    game = env.config
    num_envs = cfg.num_envs
    if phase == "teacher":
        mcfg = teacher_search(cfg)

        def run_search(params, blend, keys, states) -> SearchOutput:
            return jax.vmap(lambda k, s: search(k, s, env, mcfg))(keys, states)

    elif phase == "network":
        mcfg = selfplay_search(cfg)

        def run_search(params, blend, keys, states) -> SearchOutput:
            evaluator = make_evaluator(params, cfg, game, blend)
            return jax.vmap(lambda k, s: search(k, s, env, mcfg, evaluator))(keys, states)

    else:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    outcome_fn = jax.vmap(lambda s: heuristic.terminal_values(s, game, cfg.draw_value))
    starving_fn = jax.vmap(lambda s: rules.certainly_starving(s, game))

    def turn(params, blend, carry, key):
        states, uids = carry
        k_search, k_move, k_step, k_reset = jax.random.split(key, 4)
        out = run_search(params, blend, jax.random.split(k_search, num_envs), states)
        log_policy = jnp.where(out.policy > 0, jnp.log(jnp.maximum(out.policy, 1e-30)), -jnp.inf)
        sampled = jax.random.categorical(k_move, log_policy).astype(jnp.int32)
        explore = (states.turn < cfg.temperature_turns)[:, None]
        actions = jnp.where(explore, sampled, out.action)

        new, ts = jax.vmap(env.step)(jax.random.split(k_step, num_envs), states, actions)
        fresh = jax.vmap(env.init_state)(jax.random.split(k_reset, num_envs))
        next_states = jax.vmap(_tree_where)(ts.done, fresh, new)
        next_uids = jnp.where(ts.done, uids + num_envs, uids)

        died = states.alive & ~new.alive  # [E, N]
        causes = jnp.arange(_NUM_CAUSES) == new.elim_cause[..., None]  # [E, N, causes]
        metrics = {
            "agent_samples": jnp.sum(states.alive & ~starving_fn(states)),
            "games": jnp.sum(ts.done),
            "game_turns": jnp.sum(jnp.where(ts.done, new.turn, 0)),
            "draws": jnp.sum(ts.done & ~ts.truncated & ~jnp.any(new.alive, axis=-1)),
            "truncated": jnp.sum(ts.truncated),
            "deaths": jnp.sum(causes & died[..., None], axis=(0, 1)),
            "search_depth": jnp.sum(out.depth),
        }
        record = Positions(states, out.policy, uids, ts.done, outcome_fn(new))
        return (next_states, next_uids), (record, metrics)

    def selfplay(params, blend, states, uids, key):
        keys = jax.random.split(key, cfg.num_steps)
        carry, (traj, metrics) = jax.lax.scan(
            lambda c, k: turn(params, blend, c, k), (states, uids), keys
        )
        return carry, traj, jax.tree.map(lambda x: jnp.sum(x, axis=0), metrics)

    return selfplay


def make_train_chunk(
    cfg: AZConfig, env: BattlesnakeEnv, phase: str, num_iters: int, teacher_steps: int = 0
) -> Callable[[RunnerState], tuple[RunnerState, dict[str, jax.Array]]]:
    """``train_chunk(runner) -> (runner, metrics)``: ``num_iters`` iterations in one scan.

    An iteration plays ``num_steps`` turns of self-play (:func:`make_selfplay`),
    stores the positions and the outcomes of the games that ended in the
    replay buffer, then takes ``updates_per_iteration`` gradient steps on
    minibatches sampled from it (none until it holds ``learning_starts``
    positions). Self-play metrics are summed over the chunk, loss metrics over
    its gradient steps.
    """
    optimizer = make_optimizer(cfg)
    selfplay = make_selfplay(cfg, env, phase)
    lr = learning_rate(cfg)
    loss_and_grad = jax.value_and_grad(functools.partial(az_loss, cfg=cfg), has_aux=True)
    num_updates = cfg.updates_per_iteration
    k = cfg.steps_per_iteration

    def gradient_step(replay, carry, key):
        params, opt_state = carry
        batch = sample_batch(replay, key, cfg, env)
        (_, parts), grads = loss_and_grad(params, batch)
        step, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, step)
        return (params, opt_state), {**parts, "grad_norm": optax.tree.norm(grads)}

    def learn(replay, carry, key):
        keys = jax.random.split(key, num_updates)
        carry, parts = jax.lax.scan(lambda c, kk: gradient_step(replay, c, kk), carry, keys)
        return carry, jax.tree.map(lambda x: jnp.sum(x, axis=0), parts)

    def iteration(runner: RunnerState, _):
        key, k_play, k_learn = jax.random.split(runner.key, 3)
        blend = blend_at(cfg, runner.iterations, teacher_steps) if phase == "network" else None
        (env_states, uids), traj, metrics = selfplay(
            runner.params, blend, runner.env_states, runner.uids, k_play
        )

        def flat(x: jax.Array) -> jax.Array:  # [T, E, ...] -> [T * E, ...]
            return x.reshape(k, *x.shape[2:])

        replay = replay_add(
            runner.replay, jax.tree.map(flat, traj.state), flat(traj.policy), flat(traj.uid)
        )
        replay = replay_record_outcomes(replay, flat(traj.uid), flat(traj.done), flat(traj.outcome))
        carry = (runner.params, runner.opt_state)
        learning = replay.size >= cfg.learning_starts
        skipped = {name: jnp.zeros((), jnp.float32) for name in (*LOSS_METRICS, "grad_norm")}
        (params, opt_state), parts = jax.lax.cond(
            learning,
            lambda: learn(replay, carry, k_learn),
            lambda: (carry, skipped),
        )
        updates = jnp.where(learning, num_updates, 0).astype(jnp.int32)
        metrics.update(parts)
        metrics["updates"] = updates
        if blend is not None:
            metrics["blend"] = blend
        runner = RunnerState(
            params=params,
            opt_state=opt_state,
            env_states=env_states,
            uids=uids,
            key=key,
            iterations=runner.iterations + 1,
            updates=runner.updates + updates,
            replay=replay,
        )
        return runner, metrics

    def train_chunk(runner: RunnerState) -> tuple[RunnerState, dict[str, jax.Array]]:
        runner, metrics = jax.lax.scan(iteration, runner, None, length=num_iters)
        means = ("blend",)
        metrics = {k: jnp.mean(v, 0) if k in means else jnp.sum(v, 0) for k, v in metrics.items()}
        metrics["iterations"] = runner.iterations
        metrics["total_updates"] = runner.updates
        metrics["buffer_size"] = runner.replay.size
        metrics["lr"] = lr(runner.updates)
        return runner, metrics

    return train_chunk


def _tree_where(cond: jax.Array, a: Any, b: Any) -> Any:
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


# --- Evaluation and checkpoints --------------------------------------------------------


def eval_agent_name(mode: str, cfg: AZConfig) -> str:
    """The registry name the mode plays as (with this run's checkpoint)."""
    return "az-0" if mode == "raw" else f"az-{cfg.eval_simulations}"


def evaluate_params(
    params: Params,
    cfg: AZConfig,
    key: jax.Array,
    num_games: int,
    opponent: str = "heuristic",
    simulations: int = 0,
) -> MatchResult:
    """The network (raw, or with ``simulations`` of search) vs a ``slinky.agents`` opponent.

    Seats are balanced. The parameters are baked into the compiled match (see
    ``slinky.evaluate``), so every call compiles once more.
    """
    from slinky.agents import make_agent, make_env  # the registry can import this file in turn

    # A private copy: the training loop donates (deletes) the runner's arrays.
    params = jax.tree.map(jnp.copy, params)
    game = cfg.game
    policy = make_policy(params, cfg, game, simulations)
    other = make_agent(opponent, game)
    env = make_env(game, other.needs_obs)
    return play_match(env, policy, other.policy, key, num_games, max_turns=cfg.max_turns)


def run_evals(
    params: Params,
    cfg: AZConfig,
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
            sims = 0 if mode == "raw" else cfg.eval_simulations
            t = time.perf_counter()
            r = evaluate_params(params, cfg, key, num_games, opponent, sims)
            seconds = time.perf_counter() - t
            record = {
                "type": "eval",
                "env_steps": env_steps,
                "opponent": opponent,
                "mode": mode,
                "agent": eval_agent_name(mode, cfg),
                **r._asdict(),
                "seconds": seconds,
            }
            write(record)
            records.append(record)
            where = f"@ {env_steps:,} " if env_steps >= 0 else ""
            print(
                f"eval {where}{record['agent']:>6} vs {opponent}: {format_eval(r)} ({seconds:.0f}s)"
            )
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


def save_config(run_dir: str, cfg: AZConfig) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"algorithm": "az", **dataclasses.asdict(cfg)}, f, indent=2)


def load_config(run_dir: str) -> AZConfig:
    with open(os.path.join(run_dir, "config.json")) as f:
        d = json.load(f)
    if d.get("algorithm") != "az":
        raise ValueError(f"{run_dir} is not an AlphaZero run (config.json has no algorithm=az)")
    fields = {f.name for f in dataclasses.fields(AZConfig)}
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in fields}
    return AZConfig(**kwargs)


def load_params(run_dir: str, cfg: AZConfig | None = None) -> Params:
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


def _save_tree(path: str, tree: Any, **extra: Any) -> None:
    leaves = {f"leaf{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(tree))}
    tmp = path.removesuffix(".npz") + ".tmp.npz"
    np.savez(tmp, **extra, **leaves)
    os.replace(tmp, path)


def _load_tree(path: str, template: Any) -> tuple[Any, dict[str, np.ndarray]]:
    refs, treedef = jax.tree.flatten(template)
    with np.load(path) as f:
        leaves = [f[f"leaf{i}"] for i in range(len(refs))]
        extra = {k: f[k] for k in f.files if not k.startswith("leaf")}
    for leaf, ref in zip(leaves, refs, strict=True):
        if leaf.shape != ref.shape or leaf.dtype != ref.dtype:
            raise ValueError(f"{path} does not match the config: {leaf.shape} vs {ref.shape}")
    return jax.tree.unflatten(treedef, [jnp.asarray(x) for x in leaves]), extra


def save_runner(run_dir: str, runner: RunnerState, agent_samples: int, replay: bool) -> None:
    """Write the training state to ``runner.npz``, and the replay buffer to ``replay.npz``.

    Both atomically, each tagged with ``runner.iterations``. Without ``replay``,
    ``replay.npz`` is left out (and an old one deleted): a resumed run then
    starts with an empty buffer.
    """
    rest = runner._replace(key=jax.random.key_data(runner.key), replay=None)
    iterations = int(runner.iterations)
    replay_path = os.path.join(run_dir, "replay.npz")
    if replay:
        # The buffer first. A crash between the two files leaves it beside an older
        # runner, which would replay games already in it (and, where self-play is not
        # bit-identical, store positions under uids whose outcomes the buffer already
        # holds): load_runner uses a buffer only if its tag matches the runner's.
        _save_tree(replay_path, runner.replay, iterations=iterations)
    elif os.path.exists(replay_path):
        os.remove(replay_path)
    path = os.path.join(run_dir, "runner.npz")
    _save_tree(path, rest, agent_samples=agent_samples, iterations=iterations)


def load_runner(run_dir: str, template: RunnerState) -> tuple[RunnerState, int, bool]:
    """``(runner, agent_samples, replay restored)``, structured like ``template``.

    Without a ``replay.npz`` of the same checkpoint as ``runner.npz`` (the same
    ``iterations`` tag) the template's (empty) buffer is kept.
    """
    path = os.path.join(run_dir, "runner.npz")
    rest = template._replace(key=jax.random.key_data(template.key), replay=None)
    rest, extra = _load_tree(path, rest)
    if "iterations" not in extra or int(extra["iterations"]) != int(rest.iterations):
        raise ValueError(f"{path} was written by an older baselines/az.py; start a new run")
    replay, restored = template.replay, False
    replay_path = os.path.join(run_dir, "replay.npz")
    if os.path.exists(replay_path):
        buffer, tag = _load_tree(replay_path, template.replay)
        if "iterations" in tag and int(tag["iterations"]) == int(rest.iterations):
            replay, restored = buffer, True
    runner = rest._replace(key=jax.random.wrap_key_data(rest.key), replay=replay)
    return runner, int(extra["agent_samples"]), restored


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


# --- Logging ---------------------------------------------------------------------------


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
    m: dict[str, Any], seconds: float, sched: Schedule, phase: str, cfg: AZConfig
) -> dict[str, Any]:
    """A JSON-friendly summary of one chunk's metrics."""
    games, n_upd = int(m["games"]), int(m["updates"])
    deaths = {name: int(m["deaths"][c]) for name, c in DEATH_CAUSES.items()}
    losses = {k: float(m[k]) / n_upd if n_upd else None for k in (*LOSS_METRICS, "grad_norm")}
    target_var = losses["value_target_var"]
    # 1 - MSE / Var(z) over the chunk's value samples (R^2: a constant guess scores 0).
    explained = 1.0 - losses["value_loss"] / target_var if target_var else None
    return {
        "type": "train",
        "phase": phase,
        "env_steps": int(m["iterations"]) * cfg.steps_per_iteration,
        "agent_samples": int(m["agent_samples"]),
        "updates": n_upd,
        "total_updates": int(m["total_updates"]),
        "lr": float(m["lr"]),
        **losses,
        "explained_var": explained,
        "blend": float(m["blend"]) if "blend" in m else None,
        "games": games,
        "mean_game_len": float(m["game_turns"]) / games if games else None,
        "draws": int(m["draws"]),
        "truncated": int(m["truncated"]),
        "deaths": deaths,
        "search_depth": float(m["search_depth"]) / sched.chunk_steps,
        "buffer_size": int(m["buffer_size"]),
        "buffer_fill": int(m["buffer_size"]) / cfg.capacity,
        "env_steps_per_s": sched.chunk_steps / seconds,
        "samples_per_s": n_upd * cfg.batch_size * cfg.game.num_snakes / seconds,
        "seconds": seconds,
    }


def format_record(r: dict[str, Any], total_steps: int) -> str:
    deaths = r["deaths"]
    n_dead = sum(deaths.values())
    shown = [k for k in DEATH_CAUSES if k != "hazard" or deaths[k]]
    length = f"{r['mean_game_len']:.0f}" if r["mean_game_len"] is not None else "-"
    if r["updates"]:
        ev = f"{r['explained_var']:.2f}" if r["explained_var"] is not None else "-"
        learn = (
            f"pi {r['policy_loss']:.3f} (H {r['target_entropy']:.2f} net "
            f"{r['net_entropy']:.2f}) v {r['value_loss']:.3f} (ev {ev} |V| "
            f"{r['value_abs_mean']:.2f}) masked {100 * r['masked_frac']:.0f}%"
        )
    else:
        learn = "no updates yet"
    blend = f" lam {r['blend']:.2f}" if r["blend"] is not None else ""
    return (
        f"steps {r['env_steps']:>9,} ({100 * r['env_steps'] / total_steps:3.0f}%) "
        f"{r['phase']}{blend} | {learn} | games {r['games']:,} len {length} "
        f"draw {_pct(r['draws'], r['games'])} | deaths "
        + " ".join(f"{k} {_pct(deaths[k], n_dead)}" for k in shown)
        + f" | buf {100 * r['buffer_fill']:.0f}% lr {r['lr']:.1e} | "
        f"{r['env_steps_per_s']:,.0f} steps/s {r['samples_per_s']:,.0f} samples/s"
    )


# --- Main loop -------------------------------------------------------------------------


def train(cfg: AZConfig, resume: bool = False) -> tuple[str, Params]:
    """Run a full training; returns ``(run_dir, final params)``.

    With ``resume``, continue from ``run_dir/runner.npz`` (written every
    ``checkpoint_every`` env steps) if it exists, with the replay buffer of
    ``replay.npz`` (``save_replay``, and only if it was saved with that
    ``runner.npz``; otherwise the buffer starts empty and learning waits for
    ``learning_starts`` positions again). With the buffer, on the same machine
    and core count, resuming is exact. A new run in an old directory first
    deletes that run's metrics and checkpoints.
    """
    run_dir = cfg.run_dir or os.path.join("runs", time.strftime("az-%Y%m%d-%H%M%S"))
    cfg = dataclasses.replace(cfg, run_dir=run_dir)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    runner_path = os.path.join(run_dir, "runner.npz")

    env = make_env(cfg)
    if cfg.eval_games > 0:  # fail now, not at the first evaluation, if one can't be built
        from slinky.agents import make_agent

        for opponent in cfg.eval_opponents:
            make_agent(opponent, env.config)
    os.makedirs(run_dir, exist_ok=True)
    resuming = resume and os.path.exists(runner_path)
    if not resuming:  # a new run in an old directory: --resume must not find the old run
        for name in ("metrics.jsonl", "runner.npz", "replay.npz", "params.npz"):
            if os.path.exists(path := os.path.join(run_dir, name)):
                os.remove(path)
    save_config(run_dir, cfg)
    k_eval, k_init = jax.random.split(jax.random.key(cfg.seed))
    runner = jax.jit(init_runner, static_argnums=(0, 1))(cfg, env, k_init)
    agent_total = 0
    if resuming:
        runner, agent_total, restored = load_runner(run_dir, runner)
        resumed_steps = int(runner.iterations) * cfg.steps_per_iteration
        _truncate_metrics(metrics_path, resumed_steps)
        note = "" if restored else " with an empty replay buffer (no replay.npz of this checkpoint)"
        print(f"resuming from env step {resumed_steps:,}{note}")
    elif cfg.init_from:
        runner = runner._replace(params=init_from_run(runner.params, cfg, cfg.init_from))
        runner = runner._replace(opt_state=make_optimizer(cfg).init(runner.params))
        print(f"initialised the torso and policy head from {cfg.init_from}")

    sched = schedule(cfg)
    chunk_steps = sched.chunk_steps
    n_params = sum(x.size for x in jax.tree.leaves(runner.params))
    replay_mb = sum(x.nbytes for x in jax.tree.leaves(runner.replay)) / 2**20
    print(f"run dir: {run_dir}")
    print(f"config: {json.dumps(dataclasses.asdict(cfg))}")
    print(
        f"{n_params:,} parameters | {sched.num_chunks} chunks of {sched.chunk_iters} iterations "
        f"x {cfg.steps_per_iteration:,} env steps = {sched.total_steps:,} env steps "
        f"({sched.teacher_steps:,} with the teacher) | {sched.total_updates:,} gradient steps "
        f"of {cfg.batch_size:,} positions | replay {cfg.capacity:,} positions ({replay_mb:,.0f} MB)"
    )

    compiled: dict[str, Callable] = {}

    def chunk_fn(phase: str) -> Callable:
        if phase not in compiled:
            t0 = time.perf_counter()
            fn = make_train_chunk(cfg, env, phase, sched.chunk_iters, sched.teacher_steps)
            compiled[phase] = jax.jit(fn, donate_argnums=0).lower(runner).compile()
            print(f"compiled the {phase} chunk in {time.perf_counter() - t0:.1f}s", flush=True)
        return compiled[phase]

    start = time.perf_counter()
    with open(metrics_path, "a") as log:

        def write(record: dict[str, Any]) -> None:
            log.write(json.dumps(record) + "\n")
            log.flush()

        def crossed(every: int, steps: int) -> bool:
            return steps // every > (steps - chunk_steps) // every

        for chunk in range(int(runner.iterations) // sched.chunk_iters, sched.num_chunks):
            phase = sched.phase(chunk)
            train_chunk = chunk_fn(phase)
            t = time.perf_counter()
            runner, m = train_chunk(runner)
            m = jax.device_get(m)
            seconds = time.perf_counter() - t
            record = chunk_record(m, seconds, sched, phase, cfg)
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
                        runner.params, cfg, k, cfg.eval_games, cfg.eval_opponents,
                        cfg.eval_modes, steps, write,
                    )  # fmt: skip
            # After the eval, so an interrupted eval is redone on resume.
            if last or crossed(cfg.checkpoint_every, steps):
                save_runner(run_dir, runner, agent_total, cfg.save_replay)
    print(f"done in {time.perf_counter() - start:.0f}s; checkpoint in {run_dir}")
    return run_dir, runner.params


def evaluate_run(
    run_dir: str,
    num_games: int,
    seed: int = 0,
    opponents: tuple[str, ...] = AZConfig.eval_opponents,
    modes: tuple[str, ...] = EVAL_MODES,
    simulations: int | None = None,
) -> list[dict[str, Any]]:
    """Load ``run_dir``'s checkpoint and play it against each opponent in each mode.

    ``simulations`` replaces the run's ``eval_simulations`` for the search mode.
    """
    cfg = load_config(run_dir)
    if simulations is not None:
        cfg = dataclasses.replace(cfg, eval_simulations=simulations)
    params = load_params(run_dir, cfg)
    return run_evals(params, cfg, jax.random.key(seed), num_games, opponents, modes, -1)


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


def parse_args(argv: list[str] | None = None) -> tuple[AZConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--eval-only",
        metavar="RUN_DIR",
        help="load RUN_DIR's checkpoint and play --eval-games games (with --seed) against "
        "each of --eval-opponents in each of --eval-modes (search: --eval-simulations) instead "
        "of training",
    )
    p.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help="continue an interrupted run from its last full checkpoint, with its saved "
        "config (other flags are ignored)",
    )
    for f in dataclasses.fields(AZConfig):
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
    cfg = AZConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(AZConfig)})
    return cfg, args


def main(argv: list[str] | None = None) -> None:
    cfg, args = parse_args(argv)
    if args.eval_only:
        records = evaluate_run(
            args.eval_only, cfg.eval_games, cfg.seed, cfg.eval_opponents, cfg.eval_modes,
            cfg.eval_simulations,
        )  # fmt: skip
        for record in records:
            print(json.dumps({k: v for k, v in record.items() if k != "env_steps"}))
        return
    if args.resume:
        train(dataclasses.replace(load_config(args.resume), run_dir=args.resume), resume=True)
        return
    train(cfg)


if __name__ == "__main__":
    main()
