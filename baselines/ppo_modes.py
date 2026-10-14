"""Self-play PPO for the four leaderboard modes, one policy told its mode (single file).

One actor-critic network plays every mode of :mod:`slinky.modes` (``standard``:
4 snakes, ``duel``: 2, ``royale``: 4 with shrinking hazards, ``constrictor``: 4)
and is told which one it is playing. The same file trains a *specialist* on a
single mode (``--game-modes duel``), so at the same settings a generalist and its
specialists differ in nothing but the modes they see (the reported generalist used
``--num-minibatches 16``; see "Defaults and why"). It is ``baselines/ppo.py`` (self-play with
parameter sharing, masked policy, per-agent episodes, PPO-clip, the same
hyperparameters) with these changes:

* **One group of games per mode.** Modes cannot share a ``vmap``: the snake
  count changes the ``[N, ...]`` shapes, royale carries its own ``map_state``,
  and the ruleset and map are static. Each :class:`Group` has its own
  :class:`~slinky.types.GameConfig`, an egocentric env and an observation-free
  ``sim``, and ``num_envs`` games. A rollout is still one jitted ``lax.scan``
  over turns; each turn builds every group's inputs, makes **one** network call
  on the concatenation of all groups' rows (896 with the defaults: 128 duel +
  3 x 256), samples, splits the moves back at static offsets and steps each
  group. Finished games are replaced by fresh ones as in ``ppo.py``.
* **The network** is ``ppo.py``'s (3x3 convs, a dense layer of ``hidden``
  units, linear policy and value heads) plus one zero-initialised matrix
  ``feat/w`` ``[NUM_FEATURES, hidden]`` whose product with the features is added
  to the dense layer's pre-activation. That is the same as concatenating the
  features to its input, but every ``ppo.py`` parameter keeps its shape, so at
  initialisation the network *is* ``ppo.py``'s and ``--init-from`` a ``ppo.py``
  run is a copy. The features (:func:`slinky.modes.mode_features`) are the mode
  one-hot, the turn, the royale shrink phase and the share of opponents alive:
  what the board image lacks, and all available from a real game request.
  ``--features no_mode`` zeroes the one-hot and the phase (an ablation),
  ``--features none`` leaves the layer out (the parameters are then exactly
  ``ppo.py``'s). One value head serves every mode (see the reward).
* **Constrictor food is hidden.** The map keeps spawning food in constrictor,
  but eating it changes nothing, so the food channel is zeroed there
  (:func:`slinky.modes.hide_food`), following the real ruleset, never the mode
  the network is told. A shared network then never has to learn that food is
  good in three modes and worthless in the fourth.
* **Placement reward** (``--reward placement``, :func:`slinky.env.placement_reward`):
  ``+1/(N-1)`` for every opponent a snake outlasts, ``-1/(N-1)`` for every one
  that outlasts it, paid on the turn it happens. It is zero-sum, so self-play
  has a baseline of 0 in every mode (one value head suffices), it equals the
  win/loss reward in a duel, and it credits outlasting opponents in 4-snake
  games, where win/loss gives second to fourth place the same -1.
  ``--reward win_loss`` is ``ppo.py``'s reward.
* **Per-group learning.** GAE runs per group (``ppo.py``'s
  :func:`compute_gae`, which is generic in the number of snakes). Each epoch
  permutes every group's samples separately and gradient step ``m`` takes
  slice ``m`` of every group (stratified minibatches; one forward pass on the
  concatenation). Advantages are normalised per group within the minibatch, and
  the loss is the weighted mean of the per-group PPO losses
  (``--mode-weights``, equal by default), so every mode weighs the same
  whatever its sample count.
* **Value scale per mode** (``--value-norm``, default on). The return scale
  differs by mode (RMS 0.34 in 4-snake standard, 0.49 in the duel, 0.57 in
  constrictor under duel-PPO self-play), and the shared torso should not
  favour the mode with the largest one. So group ``g``'s value loss is
  multiplied by ``(1/v_g) / sum_h w_h (1/v_h)``, where ``v_g`` is the variance
  of its returns over the iteration's batch (at least 0.01) and ``w`` the
  normalised ``mode_weights`` (:func:`value_loss_weights`; the value clip stays
  in raw units). Between modes the weights are ``1/Var``; their weighted mean
  is 1, so the value term keeps ``ppo.py``'s overall scale and a specialist's
  factor is exactly 1: one mode is ``ppo.py`` with the flag on or off.
  Dividing by ``v_g`` outright would not be: the returns' variance is 0.03-0.2
  early on (a duel run: 1/Var about 13), so the value term would be 5-30
  times ``ppo.py``'s, the gradient clip (0.5) would bind in every chunk and
  the policy would move 20-45% less per step (measured over the first 131k
  duel env steps), and the specialist runs could no longer be checked against
  ``ppo.py``'s curve.
* **Budgets count per mode.** ``num_envs``, ``total_env_steps``,
  ``log_every``, ``eval_every`` and ``checkpoint_every`` are per mode, so every
  mode gets exactly a specialist's env steps, and one iteration takes
  ``update_epochs * num_minibatches`` gradient steps, the specialist's count at the
  same ``num_minibatches`` (the learning-rate decay is the same). The generalist's
  minibatches are the union of the specialists', so it costs about as much as all of
  them together.

**Records.** ``metrics.jsonl`` holds ``ppo.py``'s train records (``env_steps``
per mode; ``env_steps_all`` over all modes; the losses are the weighted means
over modes; ``games``, ``draws``, ``truncated``, ``deaths``, ``mean_game_len``,
``explained_var`` and ``value_mean`` are the *first* mode's, the headline) with
a ``game_modes`` block of per-mode values (also ``return_var`` and
``value_weight``, the value-loss factor). Evaluations (every ``eval_every``
env steps per mode, and at the end) play :func:`slinky.evaluate.play_table` in
each mode against each of ``eval_opponents``, the network in one seat and the
opponent in all others (``A, B, B, B``; ``A, B`` in the duel), seats rotating,
games stopped once decided. Each writes an eval record with ``game_mode``,
``num_snakes``, ``score`` (the network's share of the win: 1 for a sole
survivor, ``1/k`` for ``k`` tied; parity 0.25 with four snakes, 0.5 in the
duel, where it is ``ppo.py``'s score), ``placement`` (1 to N, parity 2.5 with
four snakes), wins / draws (ties) / losses by share, and ``mean_turns`` (the
turn the games were decided).

**Checkpoints.** ``params.npz`` (``ppo.py``'s ``layer/name`` keys, plus
``feat/w``), ``params-<env steps per mode>.npz`` snapshots at every evaluation
point and at each of ``snapshot_at`` (learning curves can be compared after the
fact), ``config.json`` (``{"algorithm": "ppo_modes", ...}``;
:func:`load_config` refuses a config without it) and ``runner.npz`` for
``--resume``, exact on the same machine and core count as in ``ppo.py``.
:func:`make_policy`, :func:`load_config` and :func:`load_params` are what the
agent registry uses; the policy computes its inputs with the training code
(:func:`policy_inputs`).

**Defaults and why.**

* ``game_modes`` all four, ``standard`` first: the headline of the top-level
  train metrics. ``mode_snakes`` (``constrictor=2``) overrides
  :data:`slinky.modes.DEFAULT_SNAKES`.
* ``num_envs=64``, ``num_steps=128``, ``total_env_steps=5_242_880`` per mode,
  ``num_minibatches=8``, ``update_epochs=4`` and the rest of the PPO and
  network settings: ``ppo.py``'s, whose duel run they produced (0.296 against
  the heuristic). Nothing was tuned for several modes; generalist and
  specialists share them so the comparison is fair. The exception is the
  reported generalist (``baselines/checkpoints/ppo-modes-seed0``), trained with
  ``--num-minibatches 16``. With 8, one step serves all four modes, and its
  updates were too small in each (duel KL 0.47x the duel specialist's; see
  ``baselines/README.md``). That gives it twice its specialists' gradient steps,
  and specialists trained with 16 are the control.
* ``reward=placement``, ``features=all``, ``value_norm=True``,
  ``hide_constrictor_food=True``: as argued above.
* ``snapshot_at=1_310_720``: a quarter of the run, where the four-mode
  generalist has taken as many env steps in all as one specialist's whole run
  (equal total compute), besides the snapshots at every evaluation.
* ``eval_opponents=random_legal,heuristic`` (the DQN only plays duels),
  ``eval_modes=greedy`` (``ppo.py``'s greedy play was at least as strong as
  sampled), ``eval_games=256`` per mode and opponent (95% half-width about
  0.05 on the share near parity).
* Cost, measured at the defaults on 2 pinned cores of the 4-core cloud CPU:
  compiling the four-mode chunk takes 15 s, and training runs at 143-158 env
  steps per second per mode (peak memory 6.2 GB; the stored observations of
  one iteration are 2.6 GB of it). That is the cost of the four specialists
  together: alone, the duel runs at 1,180 env steps/s and 4-snake standard at
  565 on the same cores, so a step of every mode costs ``1/1180 + 3/565`` =
  6.2 ms against the generalist's 6.3 ms, or 7.5 duel steps. A default run
  (5.24M env steps per mode) is therefore about 9 hours of training on 2
  cores; ``ppo.py`` scaled about linearly from 1 to 3 cores.

Commands::

    python baselines/ppo_modes.py --num-minibatches 16              # the generalist as reported
    python baselines/ppo_modes.py --game-modes duel                 # a specialist
    python baselines/ppo_modes.py --game-modes royale --run-dir runs/ppo-modes-royale-seed0
    # ppo.py, bit for bit (in the duel the placement reward is win/loss):
    python baselines/ppo_modes.py --game-modes duel --features none
    python baselines/ppo_modes.py --init-from baselines/checkpoints/ppo-duel-seed0
    python baselines/ppo_modes.py --eval-only runs/ppo-modes-seed0 --eval-games 1024
    python baselines/ppo_modes.py --resume runs/ppo-modes-seed0     # after an interruption

An *env step* is one game of one mode advancing one turn; it gives one agent
sample per living snake. The file is self-contained: it shares no code with
``baselines/ppo.py`` (whose network, GAE, loss and checkpoint helpers it
copies), so changes here never change the ``ppo`` agent.
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
import optax

from slinky import modes
from slinky.env import BattlesnakeEnv, placement_reward, win_loss_reward
from slinky.evaluate import Policy, TableResult, play_table
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
REWARDS = {"placement": placement_reward, "win_loss": win_loss_reward}
FEATURES = ("all", "no_mode", "none")
# Logit of an illegal move: exp() underflows to exactly 0, but unlike -inf it keeps
# p * log p and every gradient finite.
MASKED_LOGIT = -1e9
# value_norm weighs a group's value loss in proportion to 1 / max(Var(returns), MIN_RETURN_VAR).
MIN_RETURN_VAR = 0.01
# Feature columns that features="no_mode" keeps (it zeroes the mode one-hot and the
# royale shrink phase, which tell the modes apart).
_NO_MODE_KEEP = np.array(
    [name in ("turn", "opponents_alive") for name in modes.FEATURE_NAMES], np.float32
)


@dataclasses.dataclass(frozen=True)
class ModesConfig:
    """All hyperparameters. Every field can be overridden on the command line.

    Steps are *env steps per mode* (games of one mode advanced one turn). One
    iteration collects ``num_envs * num_steps`` env steps in every mode, then
    takes ``update_epochs * num_minibatches`` gradient steps, each on one slice
    of every mode's samples.
    """

    # Game modes (slinky.modes.MODE_NAMES): one group of num_envs games each. The first
    # is the headline of the top-level train metrics. One mode trains a specialist.
    game_modes: tuple[str, ...] = modes.MODE_NAMES
    mode_snakes: tuple[str, ...] = ()  # "constrictor=2": overrides modes.DEFAULT_SNAKES
    max_turns: int = 1000  # truncation (bootstrapped), in training and evaluation
    reward: str = "placement"  # "placement" (slinky.env.placement_reward) or "win_loss"
    features: str = "all"  # "all", "no_mode" (no one-hot or shrink phase) or "none" (no layer)
    hide_constrictor_food: bool = True  # zero the food channel in constrictor games
    value_norm: bool = True  # weight each mode's value loss by 1/Var(returns), mean weight 1
    mode_weights: tuple[float, ...] = ()  # loss weight of each game_modes entry; () = equal
    init_from: str | None = None  # a ppo.py or ppo_modes run with the same network sizes
    # Collection, per mode.
    num_envs: int = 64  # games of each mode stepped in parallel
    num_steps: int = 128  # turns per rollout (per iteration)
    total_env_steps: int = 5_242_880  # per mode; rounded up to whole log chunks
    # PPO.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 8  # per epoch; must divide every mode's agent samples
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    norm_adv: bool = True  # per mode, within each minibatch
    # Optimization.
    lr: float = 1e-3
    anneal_lr: bool = True  # linear decay to 0 over the run
    max_grad_norm: float = 0.5
    adam_eps: float = 1e-5
    # Network: ppo.py's (3x3 convs, a dense hidden layer, linear policy and value heads)
    # plus the feature layer feat/w.
    conv_channels: tuple[int, ...] = (16, 32, 32)
    conv_strides: tuple[int, ...] = (2, 2, 1)
    hidden: int = 256
    # Logging, evaluation and checkpoints (env steps per mode).
    seed: int = 0
    log_every: int = 32_768  # env steps per jitted chunk (one log line), whole iterations
    eval_every: int = 1_048_576  # env steps between evaluations (and snapshots)
    # Extra snapshots (env steps per mode). 1,310,720 = 5,242,880 / 4: the four-mode
    # generalist's equal-total-compute point against a full specialist run.
    snapshot_at: tuple[int, ...] = (1_310_720,)
    eval_games: int = 256  # per mode, opponent and eval mode
    eval_opponents: tuple[str, ...] = ("random_legal", "heuristic")  # slinky.agents
    eval_modes: tuple[str, ...] = ("greedy",)  # greedy (masked argmax) and/or sample
    checkpoint_every: int = 262_144  # env steps between full checkpoints (for --resume)
    run_dir: str | None = None  # default: runs/ppo-modes-<timestamp>

    def __post_init__(self) -> None:
        if not self.game_modes:
            raise ValueError("game_modes must name at least one mode")
        if bad := [m for m in self.game_modes if m not in modes.MODE_NAMES]:
            raise ValueError(f"unknown game_modes {bad}; expected some of {modes.MODE_NAMES}")
        if len(set(self.game_modes)) != len(self.game_modes):
            raise ValueError(f"game_modes has duplicates: {self.game_modes}")
        _parse_mode_snakes(self.mode_snakes, self.game_modes)
        if self.reward not in REWARDS:
            raise ValueError(f"unknown reward {self.reward!r}; expected one of {tuple(REWARDS)}")
        if self.features not in FEATURES:
            raise ValueError(f"unknown features {self.features!r}; expected one of {FEATURES}")
        if self.mode_weights and len(self.mode_weights) != len(self.game_modes):
            raise ValueError(
                f"mode_weights needs one weight per game mode ({len(self.game_modes)}), "
                f"got {len(self.mode_weights)}"
            )
        if any(w < 0 for w in self.mode_weights) or (
            self.mode_weights and sum(self.mode_weights) <= 0
        ):
            raise ValueError("mode_weights must be >= 0 with a positive sum")
        if len(self.conv_channels) != len(self.conv_strides):
            raise ValueError("conv_channels and conv_strides must have the same length")
        positive = (
            "max_turns", "num_envs", "num_steps", "total_env_steps", "update_epochs",
            "num_minibatches", "hidden", "log_every", "eval_every", "checkpoint_every",
        )  # fmt: skip
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for mode, n in self.snakes.items():
            samples = self.num_envs * self.num_steps * n
            if samples % self.num_minibatches:
                raise ValueError(
                    f"num_minibatches ({self.num_minibatches}) must divide the agent samples "
                    f"per iteration of every mode ({mode}: num_envs * num_steps * {n} = "
                    f"{samples})"
                )
        if any(s < 1 for s in self.snapshot_at):
            raise ValueError("snapshot_at entries must be >= 1")
        if bad := set(self.eval_modes) - set(EVAL_MODES):
            raise ValueError(f"unknown eval_modes {sorted(bad)}; expected {EVAL_MODES}")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")

    @property
    def snakes(self) -> dict[str, int]:
        """Snakes per game mode: ``modes.DEFAULT_SNAKES`` with the ``mode_snakes`` overrides."""
        overrides = _parse_mode_snakes(self.mode_snakes, self.game_modes)
        return {m: overrides.get(m, modes.DEFAULT_SNAKES[m]) for m in self.game_modes}

    @property
    def games(self) -> dict[str, GameConfig]:
        """The game of each mode, in ``game_modes`` order (``max_turns`` included)."""
        return {m: modes.mode_config(m, n, self.max_turns) for m, n in self.snakes.items()}

    @property
    def game(self) -> GameConfig:
        """The first (headline) mode's game; its board size is every mode's."""
        return self.games[self.game_modes[0]]

    @property
    def steps_per_iteration(self) -> int:
        """Env steps per iteration, per mode."""
        return self.num_envs * self.num_steps

    @property
    def loss_weights(self) -> tuple[float, ...]:
        """The normalised loss weight of each mode (summing to 1)."""
        w = self.mode_weights or (1.0,) * len(self.game_modes)
        return tuple(x / sum(w) for x in w)


def _parse_mode_snakes(items: tuple[str, ...], game_modes: tuple[str, ...]) -> dict[str, int]:
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


class Schedule(NamedTuple):
    """How a run's env steps (per mode) split into iterations and log chunks."""

    chunk_iters: int  # PPO iterations per jitted chunk
    chunk_steps: int  # env steps per chunk, per mode
    num_chunks: int
    total_steps: int  # env steps per mode of the whole run (total_env_steps rounded up)
    total_updates: int  # gradient steps of the whole run (the learning-rate decay)


def schedule(cfg: ModesConfig) -> Schedule:
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
    outside one; matmuls don't suffer (as in ``baselines/ppo.py``).
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


def init_network(key: jax.Array, cfg: ModesConfig) -> Params:
    """``ppo.py``'s network (same keys, same values) plus ``feat/w``, all zeros.

    He-normal torso, a near-zero policy head (so play starts uniform over legal
    moves). ``features="none"`` leaves ``feat`` out.
    """
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
    if cfg.features != "none":
        # Zero: the network starts as ppo.py's, and learns what the features are worth.
        params["feat"] = {"w": jnp.zeros((modes.NUM_FEATURES, cfg.hidden))}
    return params


def network(
    params: Params, obs: jax.Array, feats: jax.Array, cfg: ModesConfig
) -> tuple[jax.Array, jax.Array]:
    """``(logits f32[..., 4], value f32[...])`` for boards ``f32[..., rows, cols, C]``.

    ``feats`` is ``f32[..., NUM_FEATURES]`` (ignored without a ``feat`` layer).
    The logits are unmasked; see :func:`masked_log_probs`.
    """
    lead = obs.shape[:-3]
    x = obs.reshape(-1, *obs.shape[-3:])
    for i, stride in enumerate(cfg.conv_strides):
        p = params[f"conv{i}"]
        x = jax.nn.relu(conv3x3(x, p["w"], stride) + p["b"])
    x = x.reshape(x.shape[0], -1)
    pre = x @ params["hidden"]["w"] + params["hidden"]["b"]
    if "feat" in params:
        pre = pre + feats.reshape(-1, feats.shape[-1]) @ params["feat"]["w"]
    x = jax.nn.relu(pre)
    logits = x @ params["policy"]["w"] + params["policy"]["b"]
    value = x @ params["value"]["w"] + params["value"]["b"]
    return logits.reshape(*lead, NUM_ACTIONS), value.reshape(lead)


# --- The masked policy and its inputs ---------------------------------------------------


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


def policy_features(
    state: State, config: GameConfig, cfg: ModesConfig, game_mode: str | None = None
) -> jax.Array:
    """The feature input ``f32[N, NUM_FEATURES]`` of one (unbatched) game.

    :func:`slinky.modes.mode_features` with the mode the network is told
    (``game_mode``, by default the config's own), with the one-hot and the
    shrink phase zeroed for ``features="no_mode"``, all zeros for ``"none"``
    (whose network has no feature layer).
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
    cfg: ModesConfig,
    game_mode: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """The network inputs of one game: ``(boards, features)``, ``[N, ...]``.

    ``obs`` is the egocentric observation of ``state`` (``env.observe``). The
    boards are ``obs`` with the food channel zeroed if the *real* game is
    constrictor (and ``hide_constrictor_food``), whatever mode the network is
    told; the features are :func:`policy_features`. Training and play (the
    agent registry, via :func:`make_policy`) both go through here.
    """
    boards = modes.hide_food(obs, config) if cfg.hide_constrictor_food else obs
    return boards, policy_features(state, config, cfg, game_mode)


def _told_mode(cfg: ModesConfig, config: GameConfig, game_mode: str | None) -> str | None:
    if cfg.features == "none":
        return None  # no features: the mode never reaches the network
    mode = modes.mode_of(config) if game_mode is None else game_mode
    if mode not in modes.MODE_NAMES:
        raise ValueError(f"unknown game mode {mode!r}; expected one of {modes.MODE_NAMES}")
    return mode


def make_logits_fn(
    params: Params, cfg: ModesConfig, game_config: GameConfig, game_mode: str | None = None
) -> Callable[[State, jax.Array], jax.Array]:
    """``logits(state, obs) -> f32[N, 4]`` (unmasked) for one game with ``game_config``.

    ``obs`` is the game's egocentric observation (``ts.obs``). ``game_mode`` is
    the mode the network is told; by default ``modes.mode_of(game_config)``.
    """
    mode = _told_mode(cfg, game_config, game_mode)

    def logits_fn(state: State, obs: jax.Array) -> jax.Array:
        boards, feats = policy_inputs(obs, state, game_config, cfg, mode)
        return network(params, boards, feats, cfg)[0]

    return logits_fn


def make_policy(
    params: Params,
    cfg: ModesConfig,
    game_config: GameConfig,
    game_mode: str | None = None,
    greedy: bool = True,
) -> Policy:
    """An ``evaluate.Policy`` for games with ``game_config``: greedy (masked argmax) or sampled.

    It reads the egocentric observations (``ts.obs``: the env must compute
    them) and the state (for the features). ``game_mode`` is the mode the
    network is told (default: the config's own, :func:`slinky.modes.mode_of`);
    the food is hidden by the real config either way. The parameters are baked
    into whatever traces the policy (e.g. a compiled ``play_table``), so a new
    call compiles again.
    """
    logits_fn = make_logits_fn(params, cfg, game_config, game_mode)

    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        logits = logits_fn(state, ts.obs)
        if greedy:
            q = jnp.where(ts.action_mask, logits, -jnp.inf)
            return jnp.argmax(q, axis=-1).astype(jnp.int32)
        return sample_actions(key, logits, ts.action_mask)

    return policy


# --- Groups, rollouts and GAE ---------------------------------------------------------


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


def make_groups(cfg: ModesConfig) -> tuple[Group, ...]:
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


def group_offsets(groups: Sequence[Group], num_envs: int) -> tuple[tuple[int, int], ...]:
    """``(first row, rows)`` of each group in the concatenated network batch of a turn."""
    out, start = [], 0
    for g in groups:
        rows = num_envs * g.num_snakes
        out.append((start, rows))
        start += rows
    return tuple(out)


def _join(xs: Sequence[jax.Array]) -> jax.Array:
    """Per-group ``[E, N_g, ...]`` arrays as one ``[sum E * N_g, ...]`` batch.

    A single group keeps its shape (and so its computation is ``ppo.py``'s).
    """
    if len(xs) == 1:
        return xs[0]
    return jnp.concatenate([x.reshape(-1, *x.shape[2:]) for x in xs])


def _split(x: jax.Array, groups: Sequence[Group], num_envs: int) -> tuple[jax.Array, ...]:
    """The inverse of :func:`_join`: one ``[E, N_g, ...]`` array per group."""
    if len(groups) == 1:
        return (x,)
    return tuple(
        x[start : start + rows].reshape(num_envs, g.num_snakes, *x.shape[1:])
        for g, (start, rows) in zip(groups, group_offsets(groups, num_envs), strict=True)
    )


def _group_key(key: jax.Array, group: Group, single: bool) -> jax.Array:
    """The key of one group: the key itself in a single-mode run (as ``ppo.py``'s)."""
    return key if single else jax.random.fold_in(key, group.mode_id)


def group_inputs(
    cfg: ModesConfig, group: Group, states: State
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(boards, features, mask)`` of a batch of ``[E]`` games: ``[E, N, ...]`` each."""
    obs = jax.vmap(group.env.observe)(states)
    boards, feats = jax.vmap(lambda o, s: policy_inputs(o, s, group.game, cfg, group.name))(
        obs, states
    )
    return boards, feats, jax.vmap(group.env.action_mask)(states)


class Rollout(NamedTuple):
    """One group's rollout. Leaves are ``[T, E, N, ...]`` (``done``: ``[T, E]``).

    ``ppo.py``'s ``Rollout`` plus the features. Step ``t`` of env ``e`` goes
    from ``s_t`` (never a finished game) to ``s_{t+1}`` as reached, before any
    reset.
    """

    obs: jax.Array  # f32[T, E, N, rows, cols, C] boards of s_t (food hidden in constrictor)
    feats: jax.Array  # f32[T, E, N, NUM_FEATURES] features of s_t
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
      rules ended the game: what it was paid is final. Agents still alive when
      ``max_turns`` cuts the game off bootstrap.
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

    ``ppo.py``'s, unchanged. Arrays are ``[T, E, N]`` as in :class:`Rollout`,
    ``done`` is ``[T, E]`` and ``last_values`` ``[E, N]`` is ``V(s_T)``. The
    value of ``s_{t+1}`` as agent ``i`` of step ``t`` sees it is 0 if the step
    is terminal for it, ``boot_values[t]`` if the game was truncated at ``t+1``,
    and ``values[t+1]`` otherwise (``last_values`` for the last step). The
    recursion only continues while the agent's next step is in the same game
    and it is still alive.
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
    """Flat agent samples of one group for the update (leading axis ``B``)."""

    obs: jax.Array
    feats: jax.Array
    mask: jax.Array
    actions: jax.Array
    log_probs: jax.Array
    values: jax.Array
    advantages: jax.Array
    returns: jax.Array
    valid: jax.Array


LOSS_METRICS = ("loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac")
# Top-level train-record fields taken from the first (headline) game mode, not summed.
HEADLINE_FIELDS = (
    "explained_var", "value_mean", "games", "mean_game_len", "draws", "truncated", "deaths",
)  # fmt: skip


def value_loss_weights(return_vars: jax.Array, cfg: ModesConfig) -> jax.Array:
    """``f32[G]``: the factor on each group's value loss (``value_norm``; ones without it).

    Proportional to ``1 / v_g``, ``v_g = max(Var_g(returns), MIN_RETURN_VAR)``,
    so every mode's value error counts in units of its own return variance, and
    normalised so that their mean weighted by ``loss_weights`` is 1: the value
    term as a whole keeps ``ppo.py``'s scale, and with one mode the factor is
    exactly 1 (the run is ``ppo.py``'s, bit for bit). A constant per iteration
    (no gradient flows through it).
    """
    if not cfg.value_norm:
        return jnp.ones_like(return_vars)
    inv = 1.0 / jnp.maximum(jax.lax.stop_gradient(return_vars), MIN_RETURN_VAR)
    return inv / jnp.sum(jnp.asarray(cfg.loss_weights, jnp.float32) * inv)


def group_loss(
    logits: jax.Array, values: jax.Array, mb: Batch, value_weight: jax.Array, cfg: ModesConfig
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """``ppo.py``'s PPO-clip loss of one group's minibatch, from the network's outputs.

    Means are over the group's valid samples, and the advantages are
    normalised over them. With ``value_norm`` the value loss enters the total
    times ``value_weight`` (:func:`value_loss_weights`); the reported
    ``v_loss`` is in raw units.
    """
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
    v_term = v_loss * jax.lax.stop_gradient(value_weight) if cfg.value_norm else v_loss
    entropy = mean(masked_entropy(log_probs_all, mb.mask))
    loss = pg_loss + cfg.vf_coef * v_term - cfg.ent_coef * entropy
    parts = {
        "mode_loss": loss,
        "pg_loss": pg_loss,
        "v_loss": v_loss,
        "entropy": entropy,
        "approx_kl": mean((ratio - 1.0) - log_ratio),
        "clip_frac": mean((jnp.abs(ratio - 1.0) > cfg.clip_coef).astype(jnp.float32)),
    }
    return loss, parts


def modes_loss(
    params: Params,
    mbs: tuple[Batch, ...],
    value_weights: jax.Array,
    cfg: ModesConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The weighted mean of the groups' losses over one forward pass on all minibatches.

    ``value_weights`` ``f32[G]`` is :func:`value_loss_weights`. Returns
    ``(loss, parts)``, each part ``f32[G]`` (one entry per group). With one
    group this is ``ppo.py``'s loss.
    """
    if len(mbs) == 1:
        obs, feats = mbs[0].obs, mbs[0].feats
    else:
        obs = jnp.concatenate([mb.obs for mb in mbs])
        feats = jnp.concatenate([mb.feats for mb in mbs])
    logits, values = network(params, obs, feats, cfg)
    losses, parts, start = [], [], 0
    for g, mb in enumerate(mbs):
        if len(mbs) == 1:
            lg, v = logits, values
        else:
            n = mb.actions.shape[0]
            lg, v = logits[start : start + n], values[start : start + n]
            start += n
        loss_g, parts_g = group_loss(lg, v, mb, value_weights[g], cfg)
        losses.append(loss_g)
        parts.append(parts_g)
    weights = cfg.loss_weights
    loss = weights[0] * losses[0]
    for w, loss_g in zip(weights[1:], losses[1:], strict=True):
        loss = loss + w * loss_g
    return loss, {k: jnp.stack([p[k] for p in parts]) for k in parts[0]}


def _weighted_var(x: jax.Array, valid: jax.Array) -> jax.Array:
    w = valid.astype(jnp.float32)
    count = jnp.maximum(w.sum(), 1.0)
    mu = jnp.sum(w * x) / count
    return jnp.sum(w * (x - mu) ** 2) / count


def explained_variance(values: jax.Array, returns: jax.Array, valid: jax.Array) -> jax.Array:
    """``1 - Var(returns - values) / Var(returns)`` over the valid samples."""
    var_ret = _weighted_var(returns, valid)
    return jnp.where(var_ret > 0, 1.0 - _weighted_var(returns - values, valid) / var_ret, jnp.nan)


# --- Training ----------------------------------------------------------------------


class RunnerState(NamedTuple):
    """Everything the jitted training loop carries between chunks."""

    params: Params
    opt_state: Any
    env_states: tuple[State, ...]  # per group, [num_envs] running games (never done)
    key: jax.Array
    env_steps: jax.Array  # int32[] env steps taken, per mode
    updates: jax.Array  # int32[] gradient steps taken


def learning_rate(cfg: ModesConfig) -> optax.Schedule:
    """Constant, or a linear decay from ``lr`` to 0 over the run's gradient steps."""
    if not cfg.anneal_lr:
        return optax.constant_schedule(cfg.lr)
    return optax.linear_schedule(cfg.lr, 0.0, schedule(cfg).total_updates)


def make_optimizer(cfg: ModesConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate(cfg), eps=cfg.adam_eps),
    )


def init_runner(cfg: ModesConfig, groups: tuple[Group, ...], key: jax.Array) -> RunnerState:
    k_net, k_env, k_run = jax.random.split(key, 3)
    params = init_network(k_net, cfg)
    single = len(groups) == 1
    env_states = tuple(
        jax.vmap(g.sim.init_state)(jax.random.split(_group_key(k_env, g, single), cfg.num_envs))
        for g in groups
    )
    return RunnerState(
        params=params,
        opt_state=make_optimizer(cfg).init(params),
        env_states=env_states,
        key=k_run,
        env_steps=jnp.zeros((), jnp.int32),
        updates=jnp.zeros((), jnp.int32),
    )


def make_rollout(
    cfg: ModesConfig, groups: tuple[Group, ...]
) -> Callable[..., tuple[tuple[State, ...], tuple[Rollout, ...], dict[str, jax.Array]]]:
    """``rollout(params, states, key) -> (states, rollouts, metrics)`` over ``num_steps`` turns.

    ``states`` and ``rollouts`` are tuples with one entry per group. Metrics
    (games, lengths, draws, deaths) are summed over the rollout and stacked
    over the groups (leading axis ``G``).
    """
    num_envs = cfg.num_envs
    single = len(groups) == 1

    def step_group(params, group, states, actions, log_probs, values, inputs, k_step, k_reset):
        boards, feats, mask = inputs
        keys = jax.random.split(_group_key(k_step, group, single), num_envs)
        new, ts = jax.vmap(group.sim.step)(keys, states, actions)
        valid, terminal = agent_flags(states, new, ts.truncated)

        def boot() -> jax.Array:
            b, f, _ = group_inputs(cfg, group, new)
            return jnp.where(ts.truncated[:, None], network(params, b, f, cfg)[1], 0.0)

        # Truncations are rare (max_turns): only then evaluate the reached states.
        boot_values = jax.lax.cond(jnp.any(ts.truncated), boot, lambda: jnp.zeros_like(values))
        keys = jax.random.split(_group_key(k_reset, group, single), num_envs)
        fresh = jax.vmap(group.sim.init_state)(keys)
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
            obs=boards,
            feats=feats,
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
        return next_states, step, metrics

    def turn(params, states, key):
        k_act, k_step, k_reset = jax.random.split(key, 3)
        inputs = [group_inputs(cfg, g, s) for g, s in zip(groups, states, strict=True)]
        boards, feats, mask = (_join([x[i] for x in inputs]) for i in range(3))
        # One network call for every snake of every group.
        logits, values = network(params, boards, feats, cfg)
        actions = sample_actions(k_act, logits, mask)
        log_probs = jnp.take_along_axis(
            masked_log_probs(logits, mask), actions[..., None], axis=-1
        )[..., 0]
        per_group = zip(
            groups,
            states,
            *(_split(x, groups, num_envs) for x in (actions, log_probs, values)),
            inputs,
            strict=True,
        )
        out = [step_group(params, *args, k_step, k_reset) for args in per_group]
        next_states = tuple(o[0] for o in out)
        steps = tuple(o[1] for o in out)
        metrics = {k: jnp.stack([o[2][k] for o in out]) for k in out[0][2]}
        return next_states, (steps, metrics)

    def rollout(params, states, key):
        keys = jax.random.split(key, cfg.num_steps)
        states, (trajs, metrics) = jax.lax.scan(lambda s, k: turn(params, s, k), states, keys)
        return states, trajs, jax.tree.map(lambda x: jnp.sum(x, axis=0), metrics)

    return rollout


def make_values(
    cfg: ModesConfig, groups: tuple[Group, ...]
) -> Callable[[Params, tuple[State, ...]], tuple[jax.Array, ...]]:
    """``values(params, states) -> (V f32[E, N_g] per group)``, one network call."""

    def values(params, states):
        inputs = [group_inputs(cfg, g, s) for g, s in zip(groups, states, strict=True)]
        boards, feats = (_join([x[i] for x in inputs]) for i in range(2))
        return _split(network(params, boards, feats, cfg)[1], groups, cfg.num_envs)

    return values


def minibatch_indices(
    key: jax.Array, groups: Sequence[Group], sizes: Sequence[int], num_minibatches: int
) -> tuple[jax.Array, ...]:
    """Per group ``int32[M, S_g / M]``: a permutation of its ``S_g`` samples cut in ``M`` slices.

    Gradient step ``m`` of the epoch takes slice ``m`` of every group (stratified
    minibatches). Each group has its own permutation (one key per group; a
    single-mode run uses ``key`` itself, as ``ppo.py`` does).
    """
    single = len(groups) == 1
    return tuple(
        jax.random.permutation(_group_key(key, g, single), size).reshape(num_minibatches, -1)
        for g, size in zip(groups, sizes, strict=True)
    )


def make_train_chunk(
    cfg: ModesConfig, groups: tuple[Group, ...], num_iters: int
) -> Callable[[RunnerState], tuple[RunnerState, dict[str, jax.Array]]]:
    """``train_chunk(runner) -> (runner, metrics)``: ``num_iters`` PPO iterations in one scan.

    Each iteration collects a rollout of ``num_steps`` turns in all games of
    every mode, computes GAE per group, then takes ``update_epochs`` passes
    over the agent samples in ``num_minibatches`` stratified minibatches.
    Rollout metrics are summed over the chunk and loss metrics over its
    gradient steps; per-mode metrics have a leading axis of one entry per group.
    """
    optimizer = make_optimizer(cfg)
    rollout = make_rollout(cfg, groups)
    values_of = make_values(cfg, groups)
    lr = learning_rate(cfg)
    loss_and_grad = jax.value_and_grad(functools.partial(modes_loss, cfg=cfg), has_aux=True)
    sizes = tuple(cfg.steps_per_iteration * g.num_snakes for g in groups)

    def gradient_step(batches, value_weights, carry, idx):
        params, opt_state = carry
        mbs = tuple(jax.tree.map(lambda x, i=i: x[i], b) for b, i in zip(batches, idx, strict=True))
        (loss, parts), grads = loss_and_grad(params, mbs, value_weights)
        step, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, step)
        return (params, opt_state), {**parts, "loss": loss, "grad_norm": optax.tree.norm(grads)}

    def epoch(batches, value_weights, carry, key):
        idx = minibatch_indices(key, groups, sizes, cfg.num_minibatches)
        return jax.lax.scan(lambda c, i: gradient_step(batches, value_weights, c, i), carry, idx)

    def iteration(runner: RunnerState, _):
        key, k_roll, k_epochs = jax.random.split(runner.key, 3)
        env_states, trajs, metrics = rollout(runner.params, runner.env_states, k_roll)
        last_values = values_of(runner.params, env_states)
        batches, return_vars, evs, value_means = [], [], [], []
        for traj, last, size in zip(trajs, last_values, sizes, strict=True):
            adv, ret = compute_gae(
                traj.rewards, traj.values, traj.valid, traj.terminal, traj.done,
                traj.boot_values, last, cfg.gamma, cfg.gae_lambda,
            )  # fmt: skip

            def flat(x: jax.Array, size: int = size) -> jax.Array:  # [T, E, N, ...] -> [S, ...]
                return x.reshape(size, *x.shape[3:])

            data = (traj.obs, traj.feats, traj.mask, traj.actions, traj.log_probs, traj.values)
            batches.append(
                Batch(
                    *map(flat, data),
                    advantages=flat(adv),
                    returns=flat(ret),
                    valid=flat(traj.valid),
                )
            )
            return_vars.append(_weighted_var(ret, traj.valid))
            evs.append(explained_variance(traj.values, ret, traj.valid))
            w = traj.valid.astype(jnp.float32)
            value_means.append(jnp.sum(w * traj.values) / jnp.maximum(w.sum(), 1.0))
        batches, return_vars = tuple(batches), jnp.stack(return_vars)
        value_weights = value_loss_weights(return_vars, cfg)
        carry = (runner.params, runner.opt_state)
        keys = jax.random.split(k_epochs, cfg.update_epochs)
        (params, opt_state), parts = jax.lax.scan(
            lambda c, k: epoch(batches, value_weights, c, k), carry, keys
        )
        metrics.update({k: jnp.sum(v, axis=(0, 1)) for k, v in parts.items()})
        metrics["explained_var"] = jnp.stack(evs)
        metrics["value_mean"] = jnp.stack(value_means)
        metrics["return_var"] = return_vars
        metrics["value_weight"] = value_weights
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
        means = ("explained_var", "value_mean", "return_var", "value_weight")
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
    cfg: ModesConfig,
    group: Group,
    key: jax.Array,
    num_games: int,
    opponent: str = "heuristic",
    greedy: bool = True,
) -> TableResult:
    """The network (policy 0, greedy or sampled) in one seat, ``opponent`` in all others.

    ``play_table`` in the group's mode with the line-up ``(0, 1, ..., 1)``, seats
    rotating from game to game and games stopped once decided. The parameters
    are baked into the compiled table, so every call compiles once more.
    """
    from slinky.agents import make_agent  # the registry can import this file in turn

    # A private copy: the training loop donates (deletes) the runner's arrays.
    params = jax.tree.map(jnp.copy, params)
    policy = make_policy(params, cfg, group.game, group.name, greedy)
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


def run_evals(
    params: Params,
    cfg: ModesConfig,
    groups: tuple[Group, ...],
    key: jax.Array,
    num_games: int,
    opponents: tuple[str, ...],
    eval_modes: tuple[str, ...],
    env_steps: int,
    write: Callable[[dict[str, Any]], None] = lambda record: None,
) -> list[dict[str, Any]]:
    """Every (game mode, opponent, eval mode) table; prints and returns eval records.

    The tables of one game mode play the same games (same key).
    """
    records = []
    for group in groups:
        k = jax.random.fold_in(key, group.mode_id)
        for opponent in opponents:
            for mode in eval_modes:
                t = time.perf_counter()
                r = evaluate_params(params, cfg, group, k, num_games, opponent, mode == "greedy")
                seconds = time.perf_counter() - t
                record = eval_record(r, group, opponent, mode, env_steps, seconds)
                write(record)
                records.append(record)
                where = f"@ {env_steps:,} " if env_steps >= 0 else ""
                print(
                    f"eval {where}{group.name:>11} {mode:>6} vs {opponent}: {format_eval(r)} "
                    f"({seconds:.0f}s)",
                    flush=True,
                )
    return records


def save_checkpoint(run_dir: str, params: Params, name: str = "params.npz") -> str:
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
    """The file of the snapshot taken at ``env_steps`` (per mode)."""
    return f"params-{env_steps}.npz"


def save_config(run_dir: str, cfg: ModesConfig) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"algorithm": "ppo_modes", **dataclasses.asdict(cfg)}, f, indent=2)


def load_config(run_dir: str) -> ModesConfig:
    with open(os.path.join(run_dir, "config.json")) as f:
        d = json.load(f)
    if d.get("algorithm") != "ppo_modes":
        raise ValueError(
            f"{run_dir} is not a ppo_modes run (config.json has no algorithm=ppo_modes)"
        )
    fields = {f.name for f in dataclasses.fields(ModesConfig)}
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in fields}
    return ModesConfig(**kwargs)


def load_params(run_dir: str, cfg: ModesConfig | None = None, name: str = "params.npz") -> Params:
    """The parameters saved in ``run_dir/<name>`` (structure from ``run_dir/config.json``).

    ``name`` can also be a snapshot (:func:`snapshot_name`).
    """
    cfg = cfg or load_config(run_dir)
    template = init_network(jax.random.key(0), cfg)
    paths, treedef = jax.tree_util.tree_flatten_with_path(template)
    with np.load(os.path.join(run_dir, name)) as f:
        leaves = []
        for path, ref in paths:
            leaf = f["/".join(k.key for k in path)]
            if leaf.shape != ref.shape:
                raise ValueError(f"{path}: checkpoint shape {leaf.shape} != {ref.shape}")
            leaves.append(jnp.asarray(leaf))
    return jax.tree.unflatten(treedef, leaves)


def init_from_run(params: Params, cfg: ModesConfig, run_dir: str) -> tuple[Params, list[str]]:
    """``(params, copied)``: ``params`` with every leaf ``run_dir``'s checkpoint also has.

    ``run_dir`` is a ``ppo.py`` run or a ``ppo_modes`` run with the same
    ``conv_channels``, ``conv_strides`` and ``hidden``; every layer but ``feat``
    must be in it, with the same shapes. ``feat/w`` is copied when the source
    has it (and this network does), and otherwise keeps its zeros, so a
    ``ppo.py`` source plays exactly as before until training moves it.
    ``copied`` lists the ``layer/name`` keys taken from the source.
    """
    with open(os.path.join(run_dir, "config.json")) as f:
        source = json.load(f)
    algorithm = source.get("algorithm")
    if not (algorithm == "ppo_modes" or (algorithm is None and "gae_lambda" in source)):
        raise ValueError(f"--init-from {run_dir}: not a ppo.py or ppo_modes run")
    for name in ("conv_channels", "conv_strides", "hidden"):
        theirs = tuple(source[name]) if isinstance(source[name], list) else source[name]
        if theirs != getattr(cfg, name):
            raise ValueError(f"--init-from {run_dir}: {name} {theirs} != {getattr(cfg, name)}")
    out: Params = {layer: dict(leaves) for layer, leaves in params.items()}
    copied = []
    with np.load(os.path.join(run_dir, "params.npz")) as f:
        for layer, leaves in params.items():
            for name, ref in leaves.items():
                key = f"{layer}/{name}"
                if key not in f.files:
                    if layer == "feat":
                        continue  # a source without features: keep the zeros
                    raise ValueError(f"--init-from {run_dir}: no {key} in params.npz")
                leaf = f[key]
                if leaf.shape != ref.shape:
                    raise ValueError(f"--init-from: {key} {leaf.shape} != {ref.shape}")
                out[layer][name] = jnp.asarray(leaf, ref.dtype)
                copied.append(key)
    return out, copied


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
        if len(f.files) != len(refs) + 1:
            raise ValueError("runner.npz does not match the config (number of arrays)")
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


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def format_eval(r: TableResult) -> str:
    n = r.num_games
    return (
        f"W {_pct(r.wins[0], n)} T {_pct(r.ties[0], n)} L {_pct(r.losses[0], n)} | "
        f"share {r.share[0]:.3f} +/- {r.share_ci95[0]:.3f} | "
        f"place {r.placement[0]:.2f} +/- {r.placement_ci95[0]:.2f} | "
        f"decided at {r.mean_turns:.0f} | truncated {r.truncated}"
    )


def chunk_record(
    m: dict[str, Any], seconds: float, chunk_steps: int, cfg: ModesConfig
) -> dict[str, Any]:
    """A JSON-friendly summary of one chunk's metrics.

    ``ppo.py``'s fields at the top level: the losses are weighted means over
    the modes, ``agent_samples`` their sum, and the game statistics,
    ``explained_var`` and ``value_mean`` the first (headline) mode's. The
    ``game_modes`` block has every mode's.
    """
    n_upd = int(m["updates"])
    per_mode = {}
    for g, (name, n) in enumerate(cfg.snakes.items()):
        games = int(m["games"][g])
        samples = int(m["agent_samples"][g])
        per_mode[name] = {
            "num_snakes": n,
            "games": games,
            "mean_game_len": float(m["game_turns"][g]) / games if games else None,
            "draws": int(m["draws"][g]),
            "truncated": int(m["truncated"][g]),
            "deaths": {k: int(m["deaths"][g][c]) for k, c in DEATH_CAUSES.items()},
            "agent_samples": samples,
            "valid_frac": samples / (chunk_steps * n),
            "loss": float(m["mode_loss"][g]) / n_upd,
            **{k: float(m[k][g]) / n_upd for k in LOSS_METRICS[1:]},
            "explained_var": _finite(float(m["explained_var"][g])),
            "value_mean": float(m["value_mean"][g]),
            "return_var": float(m["return_var"][g]),
            "value_weight": float(m["value_weight"][g]),
        }
    weights = cfg.loss_weights
    head = per_mode[cfg.game_modes[0]]
    env_steps = int(m["env_steps"])
    return {
        "type": "train",
        "env_steps": env_steps,
        "env_steps_all": env_steps * len(cfg.game_modes),
        "agent_samples": sum(p["agent_samples"] for p in per_mode.values()),
        "total_updates": int(m["total_updates"]),
        "lr": float(m["lr"]),
        "loss": float(m["loss"]) / n_upd,
        **{
            k: sum(w * p[k] for w, p in zip(weights, per_mode.values(), strict=True))
            for k in LOSS_METRICS[1:]
        },
        "grad_norm": float(m["grad_norm"]) / n_upd,
        **{k: head[k] for k in HEADLINE_FIELDS},
        "env_steps_per_s": chunk_steps / seconds,
        "seconds": seconds,
        "game_modes": per_mode,
    }


def _format_mode(name: str, r: dict[str, Any]) -> str:
    deaths = r["deaths"]
    n_dead = sum(deaths.values())
    shown = [k for k in DEATH_CAUSES if k != "hazard" or deaths[k]]
    length = f"{r['mean_game_len']:.0f}" if r["mean_game_len"] is not None else "-"
    ev = f"{r['explained_var']:.2f}" if r["explained_var"] is not None else "-"
    return (
        f"  {name:>11} | pg {r['pg_loss']:+.4f} v {r['v_loss']:.4f} ent {r['entropy']:.3f} "
        f"kl {r['approx_kl']:.4f} clip {r['clip_frac']:.2f} ev {ev} V {r['value_mean']:+.2f} | "
        f"games {r['games']:,} len {length} draw {_pct(r['draws'], r['games'])} "
        f"trunc {r['truncated']} | deaths "
        + " ".join(f"{k} {_pct(deaths[k], n_dead)}" for k in shown)
    )


def format_record(r: dict[str, Any], total_steps: int) -> str:
    head = (
        f"steps {r['env_steps']:>9,} ({100 * r['env_steps'] / total_steps:3.0f}%) per mode | "
        f"loss {r['loss']:+.4f} pg {r['pg_loss']:+.4f} v {r['v_loss']:.4f} "
        f"ent {r['entropy']:.3f} kl {r['approx_kl']:.4f} clip {r['clip_frac']:.2f} | "
        f"{r['env_steps_per_s']:,.0f} steps/s per mode"
    )
    return "\n".join([head, *(_format_mode(k, v) for k, v in r["game_modes"].items())])


# --- Main loop -----------------------------------------------------------------------


def train(cfg: ModesConfig, resume: bool = False) -> tuple[str, Params]:
    """Run a full training; returns ``(run_dir, final params)``.

    With ``resume``, continue from ``run_dir/runner.npz`` (written every
    ``checkpoint_every`` env steps per mode) if it exists; exact on the same
    machine with the same number of CPU cores, as in ``ppo.py``. A new run in an
    old directory first deletes that run's metrics, checkpoints and snapshots.
    """
    run_dir = cfg.run_dir or os.path.join("runs", time.strftime("ppo-modes-%Y%m%d-%H%M%S"))
    cfg = dataclasses.replace(cfg, run_dir=run_dir)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    runner_path = os.path.join(run_dir, "runner.npz")

    groups = make_groups(cfg)
    if cfg.eval_games > 0:  # fail now, not at the first evaluation, if one can't be built
        from slinky.agents import make_agent

        for group in groups:
            for opponent in cfg.eval_opponents:
                make_agent(opponent, group.game)
    resuming = resume and os.path.exists(runner_path)
    k_eval, k_init = jax.random.split(jax.random.key(cfg.seed))
    runner = jax.jit(init_runner, static_argnums=(0, 1))(cfg, groups, k_init)
    if cfg.init_from and not resuming:  # before touching run_dir: a bad source fails here
        params, copied = init_from_run(runner.params, cfg, cfg.init_from)
        runner = runner._replace(params=params, opt_state=make_optimizer(cfg).init(params))
        print(f"initialised {', '.join(copied)} from {cfg.init_from}")
    os.makedirs(run_dir, exist_ok=True)
    if not resuming:  # a new run in an old directory: --resume must not find the old run
        names = ["metrics.jsonl", "runner.npz", "params.npz"]
        names += [os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "params-*.npz"))]
        for name in names:
            if os.path.exists(path := os.path.join(run_dir, name)):
                os.remove(path)
    save_config(run_dir, cfg)
    agent_total = 0
    if resuming:
        runner, agent_total = load_runner(run_dir, runner)
        _truncate_metrics(metrics_path, int(runner.env_steps))
        print(f"resuming from env step {int(runner.env_steps):,} per mode")

    sched = schedule(cfg)
    chunk_steps = sched.chunk_steps
    n_params = sum(x.size for x in jax.tree.leaves(runner.params))
    batch = sum(cfg.steps_per_iteration * g.num_snakes for g in groups) // cfg.num_minibatches
    print(f"run dir: {run_dir}")
    print(f"config: {json.dumps(dataclasses.asdict(cfg))}")
    print(
        f"{n_params:,} parameters | modes {', '.join(f'{g.name} ({g.num_snakes})' for g in groups)}"
        f" | {sched.num_chunks} chunks of {sched.chunk_iters} iterations x "
        f"{cfg.steps_per_iteration:,} env steps per mode = {sched.total_steps:,} env steps per "
        f"mode | {sched.total_updates:,} gradient steps of {batch:,} agent samples"
    )

    t0 = time.perf_counter()
    train_chunk = (
        jax.jit(make_train_chunk(cfg, groups, sched.chunk_iters), donate_argnums=0)
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
            record = chunk_record(m, seconds, chunk_steps, cfg)
            agent_total += record["agent_samples"]
            record["agent_samples_total"] = agent_total
            record["elapsed"] = time.perf_counter() - start
            write(record)
            print(format_record(record, sched.total_steps), flush=True)

            steps = record["env_steps"]
            last = chunk == sched.num_chunks - 1
            evaluate = last or crossed(cfg.eval_every, steps)
            if evaluate or any(steps - chunk_steps < s <= steps for s in cfg.snapshot_at):
                save_checkpoint(run_dir, runner.params, snapshot_name(steps))
            if evaluate:
                save_checkpoint(run_dir, runner.params)
                if cfg.eval_games > 0:
                    k = jax.random.fold_in(k_eval, steps)
                    run_evals(
                        runner.params, cfg, groups, k, cfg.eval_games, cfg.eval_opponents,
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
    opponents: tuple[str, ...] = ModesConfig.eval_opponents,
    eval_modes: tuple[str, ...] = ModesConfig.eval_modes,
) -> list[dict[str, Any]]:
    """Load ``run_dir``'s checkpoint and play it in each of its game modes against each opponent."""
    cfg = load_config(run_dir)
    params = load_params(run_dir, cfg)
    groups = make_groups(cfg)
    return run_evals(
        params, cfg, groups, jax.random.key(seed), num_games, opponents, eval_modes, -1
    )


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
    """The parser of a config field, from its type annotation (``ppo.py`` used the default's
    type, which an empty tuple does not have)."""
    if annotation is bool:
        return _parse_bool
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is tuple:
        return _parse_tuple(args[0])
    if origin in (typing.Union, types.UnionType):  # X | None
        (kind,) = (a for a in args if a is not type(None))
        return kind
    return annotation


def parse_args(argv: list[str] | None = None) -> tuple[ModesConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--eval-only",
        metavar="RUN_DIR",
        help="load RUN_DIR's checkpoint and play --eval-games games (with --seed) in each of "
        "its game modes against each of --eval-opponents in each of --eval-modes instead of "
        "training",
    )
    p.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help="continue an interrupted run from its last full checkpoint, with its saved "
        "config (other flags are ignored)",
    )
    hints = typing.get_type_hints(ModesConfig)
    for f in dataclasses.fields(ModesConfig):
        default = f.default
        shown = ",".join(map(str, default)) if isinstance(default, tuple) else default
        p.add_argument(
            "--" + f.name.replace("_", "-"),
            type=_field_parser(hints[f.name]),
            default=default,
            help=f"(default: {shown})",
        )
    args = p.parse_args(argv)
    cfg = ModesConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(ModesConfig)})
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
