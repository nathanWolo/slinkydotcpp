"""Self-play DQN for the four leaderboard modes, one Q-network told its mode.

``baselines/dqn.py`` (parameter-shared self-play, epsilon-greedy among legal moves,
compact replay of :class:`~slinky.types.State` pairs, Double DQN targets, Huber loss,
Adam with gradient clipping, a hard target copy, the same hyperparameters) trained on
every mode of :mod:`slinky.modes` at once with the multi-mode design of
``baselines/ppo_modes.py`` (shared code in :mod:`slinky.unified`): one group of
``num_envs`` games per mode, one network call per turn on every snake of every group,
the mode features and hidden constrictor food, and the placement reward (zero-sum, the
win/loss reward in a duel; the Double DQN target needs no change for more snakes).
Changes from ``dqn.py``:

* **One replay buffer per mode** of ``buffer_capacity`` game transitions, and every
  update samples ``batch_size`` game transitions from each (the duel run's replay
  ratio in every mode). The loss is the mean over the modes of each mode's mean Huber
  loss over its valid agent-transitions (``mode_weights``), so a mode with more snakes
  does not weigh more.
* **Exploration** decays linearly over a fixed ``eps_decay_steps`` env steps per mode,
  not a fraction of the run, whose length the plateau decides.
* **The learning rate** is constant (``dqn.py``'s 5e-4) with the campaign's optional
  cooldown (:func:`slinky.unified.lr_factor`).

Played as an agent (``dqn_modes``) the network is greedy: the legal argmax of its
Q-values; ``dqn_modes-sample`` plays epsilon-greedy with ``EVAL_EPSILON``.

Commands::

    python baselines/dqn_modes.py --size M --run-dir runs/dqn-modes-M
    python baselines/dqn_modes.py --resume runs/dqn-modes-M --cooldown 2097152
"""

from __future__ import annotations

import dataclasses
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

ALGORITHM = "dqn_modes"
# The exploration of a sampled (``dqn_modes-sample``) agent.
EVAL_EPSILON = 0.05


@dataclasses.dataclass(frozen=True)
class Config(unified.ModesSettings):
    """All hyperparameters (and :class:`slinky.unified.ModesSettings`'). Every field can be
    overridden on the command line. Steps are env steps per mode; buffers hold game
    transitions, each worth one agent-transition per snake."""

    lr: float = 5e-4
    mode_weights: tuple[float, ...] = ()  # loss weight of each game_modes entry; () = equal
    # Collection, per mode.
    num_envs: int = 32  # games of each mode stepped in parallel
    # Replay, per mode. The capacity is rounded down to a multiple of num_envs.
    buffer_capacity: int = 100_000
    learning_starts: int = 10_000  # game transitions in each buffer before learning
    batch_size: int = 128  # game transitions of each mode per update
    updates_per_step: int = 1  # gradient updates per vectorized step of every mode
    # Optimization (dqn.py's).
    max_grad_norm: float = 10.0
    gamma: float = 0.99
    target_update_period: int = 250  # gradient updates between target updates
    tau: float = 1.0  # target <- tau * online + (1 - tau) * target; 1.0 is a hard copy
    # Exploration: linear decay from eps_start to eps_end over eps_decay_steps (per mode).
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 1_572_864
    dueling: bool = False
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
            "num_envs", "batch_size", "updates_per_step", "target_update_period",
            "eps_decay_steps", "log_every",
        )  # fmt: skip
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")

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


# --- Q-network: the torso plus dqn.py's head ---------------------------------------


def init_network(key: jax.Array, cfg: Config) -> Params:
    """The torso (:func:`slinky.unified.init_torso`) and a linear Q head (or dueling heads)."""
    k_torso, k_v, k_a = jax.random.split(key, 3)
    params = unified.init_torso(k_torso, cfg)
    if cfg.dueling:
        params["value"] = unified.dense(k_v, (cfg.hidden, 1), cfg.hidden, gain=1.0)
        params["advantage"] = unified.dense(k_a, (cfg.hidden, NUM_ACTIONS), cfg.hidden, gain=1.0)
    else:
        params["q"] = unified.dense(k_v, (cfg.hidden, NUM_ACTIONS), cfg.hidden, gain=1.0)
    return params


def q_network(params: Params, boards: jax.Array, feats: jax.Array, cfg: Config) -> jax.Array:
    """Q-values ``f32[..., 4]`` for boards ``[..., rows, cols, C]`` and features."""
    lead = boards.shape[:-3]
    x = unified.torso(params, boards, feats, cfg)
    if cfg.dueling:
        v = x @ params["value"]["w"] + params["value"]["b"]
        a = x @ params["advantage"]["w"] + params["advantage"]["b"]
        q = v + a - a.mean(axis=-1, keepdims=True)
    else:
        q = x @ params["q"]["w"] + params["q"]["b"]
    return q.reshape(*lead, NUM_ACTIONS)


def epsilon_greedy(key: jax.Array, q: jax.Array, mask: jax.Array, eps: jax.Array) -> jax.Array:
    """Per agent: a random legal action with probability ``eps``, else the legal argmax."""
    k_explore, k_random = jax.random.split(key)
    greedy = jnp.argmax(jnp.where(mask, q, -jnp.inf), axis=-1)
    random = jax.random.categorical(k_random, jnp.where(mask, 0.0, -jnp.inf), axis=-1)
    explore = jax.random.uniform(k_explore, mask.shape[:-1]) < eps
    return jnp.where(explore, random, greedy).astype(jnp.int32)


def epsilon(cfg: Config, env_steps: jax.Array) -> jax.Array:
    """Linear decay from ``eps_start`` to ``eps_end`` over ``eps_decay_steps`` env steps."""
    frac = jnp.clip(env_steps / cfg.eps_decay_steps, 0.0, 1.0)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def make_policy(
    params: Params,
    cfg: Config,
    game_config: GameConfig,
    game_mode: str | None = None,
    greedy: bool = True,
) -> Policy:
    """An ``evaluate.Policy`` for games with ``game_config``: the legal argmax of the
    Q-values, or epsilon-greedy with ``EVAL_EPSILON`` (``greedy=False``).

    ``game_mode`` is the mode the network is told (default: the config's own). The
    parameters are baked into whatever traces the policy, so a new call compiles again.
    """
    mode = unified.told_mode(cfg, game_config, game_mode)

    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        boards, feats = unified.policy_inputs(ts.obs, state, game_config, cfg, mode)
        q = q_network(params, boards, feats, cfg)
        if greedy:
            return unified.greedy_actions(q, ts.action_mask)
        return epsilon_greedy(key, q, ts.action_mask, jnp.float32(EVAL_EPSILON))

    return policy


# --- Replay buffers ----------------------------------------------------------------------


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
    """Write a batch of ``B`` items at ``ptr`` (the capacity is a multiple of ``B``)."""
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


def example_transition(states: State) -> Transition:
    """A single transition shaped like those of the batch of games ``states``."""
    one = jax.tree.map(lambda x: x[0], states)
    n = states.alive.shape[-1]
    return Transition(
        one, jnp.zeros((n,), jnp.int32), jnp.zeros((n,), jnp.float32), one, jnp.zeros((), bool)
    )


# --- Targets and loss --------------------------------------------------------------


def agent_flags(state: State, next_state: State, truncated: jax.Array) -> tuple[jax.Array, ...]:
    """Per-agent ``(valid, terminal)`` flags, ``bool[..., N]``, of game transitions.

    * ``valid``: the agent was alive in a running game at ``t``, so it acted.
    * ``terminal``: no bootstrapping from ``s_{t+1}``: the agent died, or the rules ended
      the game. Agents still alive when ``max_turns`` cuts the game off bootstrap.
    """
    valid = state.alive & ~state.done[..., None]
    ended = next_state.done & ~truncated
    terminal = ~next_state.alive | ended[..., None]
    return valid, terminal


def td_targets(
    tr: Transition, q_next_online: jax.Array, q_next_target: jax.Array, next_mask: jax.Array,
    gamma: float,
) -> tuple[jax.Array, jax.Array]:  # fmt: skip
    """Double DQN targets and validity flags, ``f32[..., N]`` and ``bool[..., N]``."""
    valid, terminal = agent_flags(tr.state, tr.next_state, tr.truncated)
    a_star = jnp.argmax(jnp.where(next_mask, q_next_online, -jnp.inf), axis=-1)
    q_next = jnp.take_along_axis(q_next_target, a_star[..., None], axis=-1)[..., 0]
    return tr.rewards + gamma * jnp.where(terminal, 0.0, q_next), valid


def joined_inputs(
    cfg: Config, groups: tuple[Group, ...], states: tuple[State, ...]
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
    """``(boards, feats, masks)``: every group's inputs of ``[B_g]`` games, joined (the
    masks stay per group)."""
    inputs = [unified.group_inputs(cfg, g, s) for g, s in zip(groups, states, strict=True)]
    boards = unified.join([x[0] for x in inputs])
    feats = unified.join([x[1] for x in inputs])
    return boards, feats, tuple(x[2] for x in inputs)


def make_loss_fn(cfg: Config, groups: tuple[Group, ...]) -> Callable[..., Any]:
    """``loss(params, target_params, batches) -> (loss, (grads, parts))`` on one batch of
    game transitions per group; ``parts`` holds the per-group loss and mean Q, ``f32[G]``."""
    sizes = (cfg.batch_size,) * len(groups)

    def td_loss(params, boards, feats, actions, targets, valids):
        q_all = unified.split(q_network(params, boards, feats, cfg), groups, sizes)
        loss, losses, mean_qs = jnp.zeros(()), [], []
        for g, (q, a, target, valid) in enumerate(
            zip(q_all, actions, targets, valids, strict=True)
        ):
            q_taken = jnp.take_along_axis(q, a[..., None], axis=-1)[..., 0]
            w = valid.astype(jnp.float32)
            count = jnp.maximum(w.sum(), 1.0)
            loss_g = jnp.sum(w * optax.huber_loss(q_taken, target)) / count
            loss = loss + cfg.loss_weights[g] * loss_g
            losses.append(loss_g)
            mean_qs.append(jnp.sum(w * q_taken) / count)
        return loss, (jnp.stack(losses), jnp.stack(mean_qs))

    def loss_and_grad(params, target_params, batches):
        next_boards, next_feats, next_masks = joined_inputs(
            cfg, groups, tuple(tr.next_state for tr in batches)
        )
        q_online = unified.split(q_network(params, next_boards, next_feats, cfg), groups, sizes)
        q_target = unified.split(
            q_network(target_params, next_boards, next_feats, cfg), groups, sizes
        )
        targets, valids = [], []
        for tr, qo, qt, m in zip(batches, q_online, q_target, next_masks, strict=True):
            target, valid = td_targets(tr, qo, qt, m, cfg.gamma)
            targets.append(jax.lax.stop_gradient(target))
            valids.append(valid)
        boards, feats, _ = joined_inputs(cfg, groups, tuple(tr.state for tr in batches))
        actions = tuple(tr.actions for tr in batches)
        (loss, parts), grads = jax.value_and_grad(td_loss, has_aux=True)(
            params, boards, feats, actions, tuple(targets), tuple(valids)
        )
        return loss, (grads, parts)

    return loss_and_grad


# --- Training ----------------------------------------------------------------------


class RunnerState(NamedTuple):
    """Everything the jitted training loop carries between chunks."""

    params: Params
    target_params: Params
    opt_state: Any
    env_states: tuple[State, ...]  # per group, [num_envs] running games (never done)
    buffers: tuple[Buffer, ...]  # per group
    key: jax.Array
    env_steps: jax.Array  # int32[] env steps taken, per mode
    updates: jax.Array  # int32[] gradient updates taken


def make_optimizer(cfg: Config) -> optax.GradientTransformation:
    """Clipping and Adam's scaling; the learning rate is applied by the update."""
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.scale_by_adam())


def init_runner(cfg: Config, groups: tuple[Group, ...], key: jax.Array) -> RunnerState:
    k_net, k_env, k_run = jax.random.split(key, 3)
    params = init_network(k_net, cfg)
    env_states = tuple(
        jax.vmap(g.sim.init_state)(jax.random.split(unified.group_key(k_env, g), cfg.num_envs))
        for g in groups
    )
    return RunnerState(
        params=params,
        # A copy: the runner is donated, and one buffer can't be donated twice.
        target_params=jax.tree.map(jnp.copy, params),
        opt_state=make_optimizer(cfg).init(params),
        env_states=env_states,
        buffers=tuple(buffer_init(example_transition(s), cfg.capacity) for s in env_states),
        key=k_run,
        env_steps=jnp.zeros((), jnp.int32),
        updates=jnp.zeros((), jnp.int32),
    )


def make_train_chunk(
    cfg: Config, groups: tuple[Group, ...]
) -> Callable[[RunnerState], tuple[RunnerState, dict[str, jax.Array]]]:
    """``train_chunk(runner) -> (runner, metrics)``: ``chunk_iters`` iterations in one scan.

    Each iteration acts in and steps every mode's games, stores the transitions, and
    once the buffers hold ``learning_starts`` transitions takes ``updates_per_step``
    gradient steps. Per-mode metrics have a leading axis of one entry per group.
    """
    optimizer = make_optimizer(cfg)
    loss_and_grad = make_loss_fn(cfg, groups)
    envs = (cfg.num_envs,) * len(groups)
    start = max(cfg.learning_starts, 1)
    num_groups = len(groups)

    def gradient_step(buffers, env_steps, carry, key):
        params, target_params, opt_state, updates = carry
        keys = jax.random.split(key, num_groups)
        batches = tuple(
            buffer_sample(b, k, cfg.batch_size) for b, k in zip(buffers, keys, strict=True)
        )
        loss, (grads, parts) = loss_and_grad(params, target_params, batches)
        step, opt_state = optimizer.update(grads, opt_state, params)
        lr = cfg.lr * unified.lr_factor(env_steps, cfg)
        params = optax.apply_updates(params, unified.scale_updates(step, lr))
        updates = updates + 1
        sync = updates % cfg.target_update_period == 0
        target_params = jax.tree.map(
            lambda t, p: jnp.where(sync, optax.incremental_update(p, t, cfg.tau), t),
            target_params,
            params,
        )
        return (params, target_params, opt_state, updates), (loss, *parts)

    def learn(buffers, env_steps, carry, key):
        keys = jax.random.split(key, cfg.updates_per_step)
        carry, out = jax.lax.scan(lambda c, k: gradient_step(buffers, env_steps, c, k), carry, keys)
        return carry, (*(x.sum(0) for x in out), jnp.int32(cfg.updates_per_step))

    def skip(buffers, env_steps, carry, key):
        zeros = jnp.zeros(num_groups, jnp.float32)
        return carry, (jnp.float32(0), zeros, zeros, jnp.int32(0))

    def iteration(runner: RunnerState, _):
        key, k_act, k_step, k_reset, k_learn = jax.random.split(runner.key, 5)
        boards, feats, masks = joined_inputs(cfg, groups, runner.env_states)
        q = q_network(runner.params, boards, feats, cfg)
        eps = epsilon(cfg, runner.env_steps)
        actions = epsilon_greedy(k_act, q, jnp.concatenate([m.reshape(-1, 4) for m in masks]), eps)
        actions = unified.split(actions, groups, envs)

        env_states, buffers, per_group = [], [], []
        for g, states, a, buf in zip(
            groups, runner.env_states, actions, runner.buffers, strict=True
        ):
            keys = jax.random.split(unified.group_key(k_step, g), cfg.num_envs)
            new, ts = jax.vmap(g.sim.step)(keys, states, a)
            buffers.append(buffer_add(buf, Transition(states, a, ts.reward, new, ts.truncated)))
            keys = jax.random.split(unified.group_key(k_reset, g), cfg.num_envs)
            fresh = jax.vmap(g.sim.init_state)(keys)
            env_states.append(unified.tree_where(ts.done, fresh, new))
            per_group.append(
                {
                    "agent_transitions": jnp.sum(states.alive & ~states.done[:, None]),
                    **unified.game_metrics(states, new, ts.done, ts.truncated),
                }
            )
        buffers = tuple(buffers)
        carry = (runner.params, runner.target_params, runner.opt_state, runner.updates)
        ready = jnp.all(jnp.stack([b.size >= start for b in buffers]))
        carry, (loss, mode_loss, mean_q, n_updates) = jax.lax.cond(
            ready, learn, skip, buffers, runner.env_steps, carry, k_learn
        )
        params, target_params, opt_state, updates = carry
        metrics = {k: jnp.stack([p[k] for p in per_group]) for k in per_group[0]}
        metrics.update(loss=loss, mode_loss=mode_loss, mean_q=mean_q, updates=n_updates)
        runner = RunnerState(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            env_states=tuple(env_states),
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
        metrics["lr"] = cfg.lr * unified.lr_factor(runner.env_steps, cfg)
        metrics["env_steps"] = runner.env_steps
        metrics["total_updates"] = runner.updates
        metrics["buffer_size"] = runner.buffers[0].size
        return runner, metrics

    return train_chunk


# --- Records ----------------------------------------------------------------------------


def chunk_record(m: dict[str, Any], seconds: float, chunk_steps: int, cfg: Config) -> dict:
    """A JSON-friendly summary of one chunk: ``dqn.py``'s fields at the top level (``loss``
    the weighted mean over the modes, the game statistics and ``mean_q`` the first mode's)
    and a ``game_modes`` block with every mode's."""
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
        "lr": float(m["lr"]),
        "buffer_size": int(m["buffer_size"]),
        **{k: head[k] for k in ("games", "mean_game_len", "draws", "truncated", "deaths")},
        "env_steps_per_s": chunk_steps / seconds,
        "seconds": seconds,
        "game_modes": per_mode,
    }


def format_record(r: dict[str, Any], total_steps: int) -> str:
    head = (
        f"steps {r['env_steps']:>10,} ({100 * r['env_steps'] / total_steps:3.0f}%) per mode | "
        f"loss {r['loss']:.4f} eps {r['eps']:.3f} lr {r['lr']:.2e} buffer {r['buffer_size']:,} | "
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
