"""Self-play REINFORCE for the four leaderboard modes, one policy told its mode.

``baselines/reinforce.py`` (whole games, Monte Carlo returns, a value baseline, masked
policy, the same hyperparameters) trained on every mode of :mod:`slinky.modes` at once
with the multi-mode design of ``baselines/ppo_modes.py`` (shared code in
:mod:`slinky.unified`): one group of games per mode, one network call per turn on every
snake of every group, the mode features and hidden constrictor food, the placement
reward, and a loss that weighs every mode the same. Changes from ``reinforce.py``:

* **Games per mode.** One update plays ``mode_envs`` fresh games of each mode
  (``standard=8, duel=10, royale=12, constrictor=64`` by default) to the end, in one
  ``lax.while_loop`` that stops once every game of every mode is over. The counts are
  inversely proportional to the game lengths of the multi-mode PPO's self-play (257,
  208, 175 and 32 turns), so every mode gives about the duel run's 2,000 env steps per
  update. A finished game idles until the others end.
* **Loss.** Each mode's REINFORCE loss is its mean over its own moves, and the update
  minimises their mean (``mode_weights``), so a mode's weight does not depend on how
  many moves its games had.
* **The learning rate** is constant (``reinforce.py``'s 3e-3) with the campaign's
  optional cooldown (:func:`slinky.unified.lr_factor`), and env steps are per mode:
  the mean over the modes of their games' turns.

The reward is the placement reward (zero-sum, the win/loss reward in a duel), so the
returns of every mode have the same baseline of 0 under self-play and one value head
serves them all.

Commands::

    python baselines/reinforce_modes.py --size M --run-dir runs/reinforce-modes-M
    python baselines/reinforce_modes.py --resume runs/reinforce-modes-M
    python baselines/reinforce_modes.py --resume runs/reinforce-modes-M --cooldown 2097152
    python baselines/reinforce_modes.py --eval-only runs/reinforce-modes-M --eval-games 1024
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

ALGORITHM = "reinforce_modes"
BASELINES = ("value", "none")


@dataclasses.dataclass(frozen=True)
class Config(unified.ModesSettings):
    """All hyperparameters (and :class:`slinky.unified.ModesSettings`'). Every field can be
    overridden on the command line. Budgets are env steps per mode."""

    # Games of each mode per update, "<mode>=<games>"; modes not named play 8.
    mode_envs: tuple[str, ...] = ("standard=8", "duel=10", "royale=12", "constrictor=64")
    mode_weights: tuple[float, ...] = ()  # loss weight of each game_modes entry; () = equal
    # REINFORCE (reinforce.py's settings).
    gamma: float = 0.95
    baseline: str = "value"  # "value": G_t - V(s_t); "none": plain G_t
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    lr: float = 3e-3
    max_grad_norm: float = 0.5
    adam_eps: float = 1e-5
    # Turns per block when recomputing observations in the update.
    block_turns: int = 50
    updates_per_log: int = 32  # updates per jitted chunk (one log line)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.envs  # noqa: B018 (validates mode_envs)
        if self.baseline not in BASELINES:
            raise ValueError(f"unknown baseline {self.baseline!r}; expected one of {BASELINES}")
        if self.mode_weights and len(self.mode_weights) != len(self.game_modes):
            raise ValueError("mode_weights needs one weight per game mode")
        if any(w < 0 for w in self.mode_weights) or (
            self.mode_weights and sum(self.mode_weights) <= 0
        ):
            raise ValueError("mode_weights must be >= 0 with a positive sum")
        for name in ("block_turns", "updates_per_log"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

    @property
    def envs(self) -> tuple[int, ...]:
        """Games per update of each ``game_modes`` entry."""
        counts: dict[str, int] = {}
        for item in self.mode_envs:
            name, sep, n = (t.strip() for t in item.partition("="))
            if not sep or not n.isdigit() or int(n) < 1:
                raise ValueError(f"mode_envs items are <mode>=<games >= 1>, got {item!r}")
            counts[name] = int(n)
        return tuple(counts.get(m, 8) for m in self.game_modes)

    @property
    def loss_weights(self) -> tuple[float, ...]:
        w = self.mode_weights or (1.0,) * len(self.game_modes)
        return tuple(x / sum(w) for x in w)

    @property
    def num_blocks(self) -> int:
        return -(-self.max_turns // self.block_turns)

    @property
    def buffer_turns(self) -> int:
        """Turns stored per game: ``max_turns`` rounded up to whole blocks."""
        return self.num_blocks * self.block_turns


# --- Network: the torso plus PPO's policy and value heads -------------------------------


def init_network(key: jax.Array, cfg: Config) -> Params:
    """The torso (:func:`slinky.unified.init_torso`), a near-zero policy head (so play
    starts uniform over legal moves) and a value head."""
    k_torso, k_pi, k_v = jax.random.split(key, 3)
    params = unified.init_torso(k_torso, cfg)
    params["policy"] = unified.dense(k_pi, (cfg.hidden, NUM_ACTIONS), cfg.hidden, gain=1e-4)
    params["value"] = unified.dense(k_v, (cfg.hidden, 1), cfg.hidden, gain=1.0)
    return params


def network(
    params: Params, boards: jax.Array, feats: jax.Array, cfg: Config
) -> tuple[jax.Array, jax.Array]:
    """``(logits f32[..., 4], value f32[...])`` for boards ``[..., rows, cols, C]``."""
    lead = boards.shape[:-3]
    x = unified.torso(params, boards, feats, cfg)
    logits = x @ params["policy"]["w"] + params["policy"]["b"]
    value = x @ params["value"]["w"] + params["value"]["b"]
    return logits.reshape(*lead, NUM_ACTIONS), value.reshape(lead)


def make_policy(
    params: Params,
    cfg: Config,
    game_config: GameConfig,
    game_mode: str | None = None,
    greedy: bool = True,
) -> Policy:
    """An ``evaluate.Policy`` for games with ``game_config``: greedy (masked argmax) or sampled.

    It reads the egocentric observations (``ts.obs``) and the state (for the features).
    ``game_mode`` is the mode the network is told (default: the config's own). The
    parameters are baked into whatever traces the policy, so a new call compiles again.
    """
    mode = unified.told_mode(cfg, game_config, game_mode)

    def policy(key: jax.Array, state: State, ts: TimeStep) -> jax.Array:
        boards, feats = unified.policy_inputs(ts.obs, state, game_config, cfg, mode)
        logits = network(params, boards, feats, cfg)[0]
        if greedy:
            return unified.greedy_actions(logits, ts.action_mask)
        return unified.sample_actions(key, logits, ts.action_mask)

    return policy


# --- Playing whole games --------------------------------------------------------------


class Games(NamedTuple):
    """One group's batch of whole games. Per-turn leaves are ``[T, E, ...]``.

    Turn ``t`` of game ``e`` goes from ``states[t, e]`` to ``states[t + 1, e]``. Entries
    after a game has ended are padding (``valid`` False, reward 0).
    """

    states: State  # [T, E] the state each move was chosen in
    actions: jax.Array  # int32[T, E, N]
    rewards: jax.Array  # f32[T, E, N]
    valid: jax.Array  # bool[T, E, N] the snake was alive in a running game: it moved
    final: State  # [E] the final states
    truncated: jax.Array  # bool[E] the game was cut off by max_turns


def make_play(
    cfg: Config, groups: tuple[Group, ...]
) -> Callable[[Params, jax.Array], tuple[tuple[Games, ...], jax.Array]]:
    """``play(params, key) -> (games per group, turns)``: fresh self-play games of every
    mode, played until all of them are over (``turns`` loop iterations)."""
    envs, num_turns = cfg.envs, cfg.buffer_turns

    def play(params: Params, key: jax.Array) -> tuple[tuple[Games, ...], jax.Array]:
        k_init, k_loop = jax.random.split(key)
        states = tuple(
            jax.vmap(g.sim.init_state)(jax.random.split(unified.group_key(k_init, g), e))
            for g, e in zip(groups, envs, strict=True)
        )

        def empty(s: State, e: int) -> Games:
            n = s.alive.shape[-1]
            return Games(
                states=jax.tree.map(lambda x: jnp.zeros((num_turns, *x.shape), x.dtype), s),
                actions=jnp.zeros((num_turns, e, n), jnp.int32),
                rewards=jnp.zeros((num_turns, e, n), jnp.float32),
                valid=jnp.zeros((num_turns, e, n), bool),
                final=s,
                truncated=jnp.zeros(e, bool),
            )

        def cond(carry):
            t, states, _ = carry
            running = [~jnp.all(s.done) for s in states]
            return (t < num_turns) & jnp.any(jnp.stack(running))

        def body(carry):
            t, states, games = carry
            k_act, k_step = jax.random.split(jax.random.fold_in(k_loop, t))
            inputs = [unified.group_inputs(cfg, g, s) for g, s in zip(groups, states, strict=True)]
            boards, feats, mask = (unified.join([x[i] for x in inputs]) for i in range(3))
            logits, _ = network(params, boards, feats, cfg)
            actions = unified.split(unified.sample_actions(k_act, logits, mask), groups, envs)
            new_states, new_games = [], []
            for g, e, s, a, gm in zip(groups, envs, states, actions, games, strict=True):
                keys = jax.random.split(unified.group_key(k_step, g), e)
                new, ts = jax.vmap(g.sim.step)(keys, s, a)

                def put(buf: jax.Array, x: jax.Array) -> jax.Array:
                    return jax.lax.dynamic_update_index_in_dim(buf, x, t, 0)

                new_games.append(
                    gm._replace(
                        states=jax.tree.map(put, gm.states, s),
                        actions=put(gm.actions, a),
                        rewards=put(gm.rewards, ts.reward),
                        valid=put(gm.valid, s.alive & ~s.done[:, None]),
                        truncated=gm.truncated | ts.truncated,
                    )
                )
                new_states.append(new)
            return t + 1, tuple(new_states), tuple(new_games)

        init = (jnp.int32(0), states, tuple(empty(s, e) for s, e in zip(states, envs, strict=True)))
        t, final, games = jax.lax.while_loop(cond, body, init)
        return tuple(gm._replace(final=f) for gm, f in zip(games, final, strict=True)), t

    return play


def discounted_returns(rewards: jax.Array, gamma: float) -> jax.Array:
    """``G_t = r_t + gamma * G_{t+1}`` along axis 0 (``G`` is 0 after the last turn)."""

    def step(following: jax.Array, r: jax.Array) -> tuple[jax.Array, jax.Array]:
        g = r + gamma * following
        return g, g

    _, returns = jax.lax.scan(step, jnp.zeros_like(rewards[0]), rewards, reverse=True)
    return returns


# --- Loss ------------------------------------------------------------------------------

# Sums over a mode's valid samples (see block_loss).
SUMS = ("pg_loss", "v_loss", "entropy", "value", "value_sq", "ret", "ret_sq", "err", "err_sq")


def block_loss(
    params: Params,
    blocks: tuple[tuple[jax.Array, ...], ...],
    counts: jax.Array,
    cfg: Config,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The loss of one block of turns of every mode, each mode's divided by its valid
    samples in the whole batch (``counts``) and weighted by ``loss_weights``.

    ``blocks`` holds, per mode, ``(boards, feats, mask, actions, returns, valid)`` with a
    leading axis of the block's samples. Summed over every block it is the weighted mean
    over the modes of ``reinforce.py``'s loss. Also returns the plain per-mode sums
    ``f32[G]`` over the valid samples, for the logs.
    """
    boards = jnp.concatenate([b[0] for b in blocks])
    feats = jnp.concatenate([b[1] for b in blocks])
    logits_all, values_all = network(params, boards, feats, cfg)
    use_value = cfg.baseline == "value"
    loss, sums, start = jnp.zeros(()), {k: [] for k in SUMS}, 0
    for g, (_, _, mask, actions, returns, valid) in enumerate(blocks):
        n = actions.shape[0]
        logits, values = logits_all[start : start + n], values_all[start : start + n]
        start += n
        log_probs_all = unified.masked_log_probs(logits, mask)
        log_probs = jnp.take_along_axis(log_probs_all, actions[..., None], axis=-1)[..., 0]
        w = valid.astype(jnp.float32)
        adv = returns - jax.lax.stop_gradient(values) if use_value else returns
        pg = -jnp.sum(w * adv * log_probs)
        err = returns - values
        v = 0.5 * jnp.sum(w * err**2)
        ent = jnp.sum(w * unified.masked_entropy(log_probs_all, mask))
        mode_loss = pg + (cfg.vf_coef * v if use_value else 0.0) - cfg.ent_coef * ent
        loss = loss + cfg.loss_weights[g] * mode_loss / counts[g]
        parts = {
            "pg_loss": pg, "v_loss": v, "entropy": ent,
            "value": jnp.sum(w * values), "value_sq": jnp.sum(w * values**2),
            "ret": jnp.sum(w * returns), "ret_sq": jnp.sum(w * returns**2),
            "err": jnp.sum(w * err), "err_sq": jnp.sum(w * err**2),
        }  # fmt: skip
        for k in SUMS:
            sums[k].append(parts[k])
    return loss, jax.lax.stop_gradient({k: jnp.stack(v) for k, v in sums.items()})


def batch_gradient(
    params: Params,
    games: tuple[Games, ...],
    turns: jax.Array,
    cfg: Config,
    groups: tuple[Group, ...],
) -> tuple[Params, dict[str, jax.Array]]:
    """``(gradient, sums)`` of the loss over every move of every mode's games, block by block.

    Inputs are recomputed from the stored states ``block_turns`` turns at a time; the
    blocks after the longest game hold no moves and are skipped.
    """
    returns = [discounted_returns(gm.rewards, cfg.gamma) for gm in games]
    counts = jnp.stack([jnp.maximum(jnp.sum(gm.valid), 1).astype(jnp.float32) for gm in games])
    grad_fn = jax.grad(block_loss, has_aux=True)
    zero_grads = jax.tree.map(jnp.zeros_like, params)
    zero_sums = {k: jnp.zeros(len(groups), jnp.float32) for k in SUMS}
    size = cfg.block_turns

    def block(carry, i):
        grads, sums = carry

        def compute():
            def cut(x: jax.Array) -> jax.Array:
                return jax.lax.dynamic_slice_in_dim(x, i * size, size, 0)

            blocks = []
            for g, gm, ret in zip(groups, games, returns, strict=True):
                states = jax.tree.map(cut, gm.states)  # [B, E]
                b, f, m = jax.vmap(lambda s, g=g: unified.group_inputs(cfg, g, s))(states)

                def flat(x: jax.Array) -> jax.Array:  # [B, E, N, ...] -> [B * E * N, ...]
                    return x.reshape(-1, *x.shape[3:])

                blocks.append(tuple(map(flat, (b, f, m, cut(gm.actions), cut(ret), cut(gm.valid)))))
            return grad_fn(params, tuple(blocks), counts, cfg)

        g, s = jax.lax.cond(i * size < turns, compute, lambda: (zero_grads, zero_sums))
        return (jax.tree.map(jnp.add, grads, g), jax.tree.map(jnp.add, sums, s)), None

    (grads, sums), _ = jax.lax.scan(block, (zero_grads, zero_sums), jnp.arange(cfg.num_blocks))
    return grads, {**sums, "samples": counts}


# --- Training ------------------------------------------------------------------------


class RunnerState(NamedTuple):
    """Everything the jitted training loop carries between chunks."""

    params: Params
    opt_state: Any
    key: jax.Array
    env_steps_all: jax.Array  # int32[] turns of running games, summed over the modes
    env_steps: jax.Array  # int32[] per mode: env_steps_all // number of modes
    updates: jax.Array  # int32[] gradient steps taken


def make_optimizer(cfg: Config) -> optax.GradientTransformation:
    """Clipping and Adam's scaling; the learning rate is applied by the update."""
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm), optax.scale_by_adam(eps=cfg.adam_eps)
    )


def init_runner(cfg: Config, groups: tuple[Group, ...], key: jax.Array) -> RunnerState:
    k_net, k_run = jax.random.split(key)
    params = init_network(k_net, cfg)
    zero = jnp.zeros((), jnp.int32)
    return RunnerState(params, make_optimizer(cfg).init(params), k_run, zero, zero, zero)


def make_train_chunk(
    cfg: Config, groups: tuple[Group, ...]
) -> Callable[[RunnerState], tuple[RunnerState, dict[str, jax.Array]]]:
    """``train_chunk(runner) -> (runner, metrics)``: ``updates_per_log`` updates in one scan.

    Each update plays a batch of whole games of every mode and takes one gradient step on
    them. Per-mode metrics have a leading axis of one entry per group, summed over the chunk.
    """
    optimizer = make_optimizer(cfg)
    play = make_play(cfg, groups)
    num_modes = len(groups)

    def update(runner: RunnerState, _):
        key, k_play = jax.random.split(runner.key)
        games, turns = play(runner.params, k_play)
        grads, sums = batch_gradient(runner.params, games, turns, cfg, groups)
        lr = cfg.lr * unified.lr_factor(runner.env_steps, cfg)
        step, opt_state = optimizer.update(grads, runner.opt_state, runner.params)
        params = optax.apply_updates(runner.params, unified.scale_updates(step, lr))

        per_mode = []
        for gm, e in zip(games, cfg.envs, strict=True):
            final = gm.final
            died = ~final.alive  # every snake that died, by its cause
            causes = jnp.arange(unified.NUM_CAUSES) == final.elim_cause[..., None]
            per_mode.append(
                {
                    "games": jnp.int32(e),
                    "game_turns": jnp.sum(final.turn),
                    "slot_turns": turns * e,
                    "max_game_len": jnp.max(final.turn),
                    "draws": jnp.sum(~gm.truncated & ~jnp.any(final.alive, axis=-1)),
                    "truncated": jnp.sum(gm.truncated),
                    "deaths": jnp.sum(causes & died[..., None], axis=(0, 1)),
                }
            )
        metrics = {k: jnp.stack([p[k] for p in per_mode]) for k in per_mode[0]}
        metrics.update(sums)
        metrics["grad_norm"] = optax.tree.norm(grads)
        metrics["lr"] = lr
        steps_all = runner.env_steps_all + jnp.sum(metrics["game_turns"])
        runner = RunnerState(
            params=params,
            opt_state=opt_state,
            key=key,
            env_steps_all=steps_all,
            env_steps=steps_all // num_modes,
            updates=runner.updates + 1,
        )
        return runner, metrics

    def train_chunk(runner: RunnerState) -> tuple[RunnerState, dict[str, jax.Array]]:
        runner, metrics = jax.lax.scan(update, runner, None, length=cfg.updates_per_log)
        out = {
            k: jnp.max(v, 0) if k == "max_game_len" else jnp.sum(v, 0)
            for k, v in metrics.items() if k != "lr"
        }  # fmt: skip
        out["lr"] = metrics["lr"][-1]
        out["env_steps"] = runner.env_steps
        out["env_steps_all"] = runner.env_steps_all
        out["total_updates"] = runner.updates
        return runner, out

    return train_chunk


# --- Records ----------------------------------------------------------------------------


def chunk_record(m: dict[str, Any], seconds: float, chunk_steps: int, cfg: Config) -> dict:
    """A JSON-friendly summary of one chunk: ``reinforce.py``'s fields at the top level (the
    losses weighted means over the modes, the game statistics the first mode's) and a
    ``game_modes`` block with every mode's."""
    n_upd = cfg.updates_per_log
    per_mode = {}
    for g, (name, n) in enumerate(cfg.snakes.items()):
        count = float(m["samples"][g])

        def mean(k: str, count: float = count, g: int = g) -> float:
            return float(m[k][g]) / count if count else 0.0

        var_ret = mean("ret_sq") - mean("ret") ** 2
        var_err = mean("err_sq") - mean("err") ** 2
        per_mode[name] = {
            "num_snakes": n,
            **unified.mode_game_stats(m, g),
            "max_game_len": int(m["max_game_len"][g]),
            "busy": float(m["game_turns"][g]) / max(float(m["slot_turns"][g]), 1.0),
            "agent_samples": int(count),
            "pg_loss": mean("pg_loss"),
            "v_loss": mean("v_loss"),
            "entropy": mean("entropy"),
            "value_mean": mean("value"),
            "return_mean": mean("ret"),
            "return_var": var_ret,
            "explained_var": unified.finite(1.0 - var_err / var_ret) if var_ret > 0 else None,
        }
    weights = cfg.loss_weights
    head = per_mode[cfg.game_modes[0]]
    modes_ = per_mode.values()
    pg = sum(w * p["pg_loss"] for w, p in zip(weights, modes_, strict=True))
    v = sum(w * p["v_loss"] for w, p in zip(weights, modes_, strict=True))
    ent = sum(w * p["entropy"] for w, p in zip(weights, modes_, strict=True))
    vf = cfg.vf_coef if cfg.baseline == "value" else 0.0
    return {
        "type": "train",
        "env_steps": int(m["env_steps"]),
        "env_steps_all": int(m["env_steps_all"]),
        "agent_samples": sum(p["agent_samples"] for p in modes_),
        "total_updates": int(m["total_updates"]),
        "lr": float(m["lr"]),
        "loss": pg + vf * v - cfg.ent_coef * ent,
        "pg_loss": pg,
        "v_loss": v,
        "entropy": ent,
        "grad_norm": float(m["grad_norm"]) / n_upd,
        **{k: head[k] for k in ("games", "mean_game_len", "draws", "truncated", "deaths")},
        "explained_var": head["explained_var"],
        "value_mean": head["value_mean"],
        "env_steps_per_s": chunk_steps / seconds,
        "seconds": seconds,
        "game_modes": per_mode,
    }


def format_record(r: dict[str, Any], total_steps: int) -> str:
    head = (
        f"steps {r['env_steps']:>10,} ({100 * r['env_steps'] / total_steps:3.0f}%) per mode | "
        f"loss {r['loss']:+.4f} pg {r['pg_loss']:+.4f} v {r['v_loss']:.4f} "
        f"ent {r['entropy']:.3f} lr {r['lr']:.2e} | {r['env_steps_per_s']:,.0f} steps/s per mode"
    )
    lines = [head]
    for name, p in r["game_modes"].items():
        ev = f"{p['explained_var']:.2f}" if p["explained_var"] is not None else "-"
        lines.append(
            f"  {name:>11} | pg {p['pg_loss']:+.4f} v {p['v_loss']:.4f} ent {p['entropy']:.3f} "
            f"ev {ev} V {p['value_mean']:+.2f} busy {p['busy']:.2f} | {unified.format_games(p)}"
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
