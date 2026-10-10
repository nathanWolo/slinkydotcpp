"""Tests for the agent registry (``slinky.agents``): names, canonical forms, building agents."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import pytest

from slinky import agents as agents_mod
from slinky import mcts
from slinky.agents import (
    MCTS_SHORTHANDS,
    AgentSpec,
    agent_config,
    check_game,
    format_number,
    make_agent,
    mcts_name,
    parse_agent,
)
from slinky.replay import record_games
from slinky.types import GameConfig

CONFIG = GameConfig()

# Spellings and their canonical names.
CANONICAL = {
    "mcts-64": "mcts-64",
    " mcts-64 ": "mcts-64",
    "mcts-64-rollout": "mcts-64-rollout10",
    "mcts-64:rollout_steps=10": "mcts-64-rollout10",
    "mcts-64:rollout_steps=5:rollout_policy=heuristic": "mcts-64-hrollout5",
    "mcts-64-sample-hrollout5-spawn-noheur": "mcts-64-noheur-hrollout5-spawn-sample",
    "mcts-256-c0.5-rm": None,  # c has no effect with rm: an error, below
    "mcts-256-rm-gamma0.3": "mcts-256-rm-gamma0.3",
    "mcts-256:selection=rm:rm_gamma=0.3": "mcts-256-rm-gamma0.3",
    "mcts-64-c0.5-tuned": None,
    "mcts-64-tuned": "mcts-64-tuned",
    "mcts-64-c1.4": "mcts-64-c1.4",
    "mcts-64:exploration=1.4": "mcts-64-c1.4",
    "mcts-64-c0.25": "mcts-64",  # the default
    "mcts-64:selection=duct": "mcts-64",
    "mcts-64-noise0.00001": "mcts-64-noise0.00001",  # no exponent (1e-05 has a dash)
    "mcts-64-c0.123456789": "mcts-64-c0.123456789",  # every digit kept
    "mcts-64-contempt0": "mcts-64-contempt0",
    "mcts-64:draw_value=0": "mcts-64-contempt0",
    "mcts-64:draw_value=0.3": "mcts-64:draw_value=0.3",  # contempt can't spell -0.3
    "mcts-64-contempt0.5": "mcts-64",
    "mcts-64-depth64": "mcts-64-depth64",
    "mcts-16-depth32": "mcts-16",
    "mcts-16-depth8": "mcts-16-depth8",
    "mcts-128:spawn_food=yes:final=sample": "mcts-128-spawn-sample",
    "dqn": "dqn",
    "dqn:baselines/checkpoints/dqn-duel-seed0": "dqn",
    "dqn:/nonexistent/run": "dqn:/nonexistent/run",
    "random_legal": "random_legal",
    "heuristic": "heuristic",
}


def test_canonical_names_round_trip():
    names = {}
    for text, canonical in CANONICAL.items():
        if canonical is None:
            continue
        spec = parse_agent(text)
        assert spec.name == canonical, text
        assert parse_agent(spec.name) == spec  # the canonical name means the same agent
        names.setdefault(spec.name, spec)
        assert names[spec.name] == spec
    assert parse_agent("mcts-64-rollout").mcts == mcts.MCTSConfig(
        num_simulations=64, rollout_steps=10
    )
    assert parse_agent("mcts-64-contempt0.2").mcts.draw_value == -0.2
    assert parse_agent("heuristic") == AgentSpec("heuristic", "heuristic")
    assert parse_agent("dqn").checkpoint == agents_mod.DEFAULT_DQN_CHECKPOINT.resolve()


def test_every_shorthand_round_trips_and_names_are_unique():
    """Distinct configs get distinct names: config -> name -> config is the identity."""
    configs = {}
    for s in MCTS_SHORTHANDS:
        values = {None: [None], int: [1, 2, 3], float: [0.1, 0.35, 1.0]}[s.number]
        for v in values:
            for base in ("mcts-64-rm",) if s.name == "gamma" else ("mcts-64",):
                token = s.name if v is None else f"{s.name}{format_number(v)}"
                spec = parse_agent(f"{base}-{token}")
                assert parse_agent(spec.name) == spec
                assert configs.setdefault(spec.name, spec.mcts) == spec.mcts
    # Any config the names can make has a name that reads back as itself.
    for cfg in (
        mcts.MCTSConfig(num_simulations=7, draw_value=0.25, exploration=0.0),
        mcts.MCTSConfig(num_simulations=7, selection="rm", rm_gamma=0.05, final="sample"),
    ):
        assert parse_agent(mcts_name(cfg)).mcts == cfg
    assert mcts_name(mcts.MCTSConfig(num_simulations=7, draw_value=0.25)) == (
        "mcts-7:draw_value=0.25"
    )


def test_format_number():
    cases = {1e-05: "0.00001", 0.123456789: "0.123456789", 3.0: "3", 2.5e7: "25000000",
             0.5: "0.5", -0.3: "-0.3", 10: "10", 0.0: "0"}  # fmt: skip
    for value, text in cases.items():
        assert format_number(value) == text and float(text) == value


@pytest.mark.parametrize(
    "name, message",
    [
        ("alphasnake", "unknown agent"),
        ("mcts-0", "n >= 1"),
        ("mcts-x", "write mcts-<n>"),
        ("mcts-64-bogus", "unknown shorthand 'bogus'"),
        ("mcts-64-rm2", "takes no number"),
        ("mcts-64-c", "needs a number"),
        ("mcts-64-c1.2.3", "expected a float"),
        ("mcts-64-rollout0", "must be >= 1"),
        ("mcts-64-rollout-hrollout", "both set rollout_steps"),
        ("mcts-64-rm:selection=duct", "both set selection"),
        ("mcts-64-c1:exploration=1", "both set exploration"),
        ("mcts-64:bogus=1", "unknown mcts option 'bogus'"),
        ("mcts-64:exploration=lots", "expected a float"),
        ("mcts-64:exploration=nan", "finite"),
        ("mcts-64:exploration", "field=value"),
        ("mcts-64:num_simulations=3", "mcts-<n>"),
        ("mcts-64:weights=1", "can't be set"),
        ("mcts-64:spawn_food=maybe", "true or false"),
        ("mcts-64:selection=argmax", "selection must be one of"),
        # Settings that change nothing are rejected: they would give one agent two names.
        ("mcts-64-gamma0.3", "gamma (rm_gamma) has no effect without rm"),
        ("mcts-256-c0.5-rm", "c (exploration) has no effect with rm"),
        ("mcts-64-rm-tuned", "tuned (ucb1_tuned) has no effect with rm"),
        ("mcts-64-rm-noise0.1", "noise (tie_noise) has no effect with rm"),
        ("mcts-64-c0.5-tuned", "c (exploration) has no effect with tuned"),
        ("mcts-64:rollout_policy=heuristic", "rollout_policy has no effect"),
        ("mcts-16-depth40", "caps the depth"),
        ("mcts-16-depth16", "caps the depth"),
        ("dqn:", "write dqn:<run dir>"),
        ("dqn@seed0", "write dqn:<run dir>"),
        ("ppo:", "write ppo:<run dir>"),
        ("ppo-sample: ", "write ppo:<run dir>"),
        ("ppo@seed0", "write ppo:<run dir>"),
        ("ppo-bogus", "unknown ppo mode 'bogus'"),
        ("ppo-greedy-sample", "unknown ppo mode 'greedy-sample'"),
        ("ppox", "unknown agent"),
    ],
)
def test_bad_names(name, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_agent(name)


def test_agent_configs_are_complete_json():
    cfg = agent_config(parse_agent("mcts-64-rm"))
    fields = {f.name for f in dataclasses.fields(mcts.MCTSConfig)}
    assert set(cfg) == fields | {"type"} and cfg["selection"] == "rm"
    assert isinstance(cfg["weights"], dict) and json.loads(json.dumps(cfg)) == cfg
    assert agent_config(parse_agent("random")) == {"type": "random"}
    assert set(agent_config(parse_agent("heuristic"))["weights"]) >= {"territory", "contempt"}
    with pytest.raises(FileNotFoundError, match="no DQN checkpoint"):
        agent_config(parse_agent("dqn:/nonexistent/run"))
    with pytest.raises(FileNotFoundError, match="no PPO checkpoint"):
        agent_config(parse_agent("ppo-sample:/nonexistent/run"))


def test_agents_registry():
    for name in ("random_legal", "random", "heuristic"):
        agent = make_agent(name, CONFIG)
        assert agent.name == name and not agent.needs_obs and agent.description
        assert make_agent(name, CONFIG) is agent  # cached: play_match's jit cache hits
    assert (
        make_agent("heuristic", CONFIG).policy
        is not make_agent("heuristic", GameConfig(num_snakes=4)).policy
    )
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("alphasnake", CONFIG)
    with pytest.raises(FileNotFoundError, match="no DQN checkpoint"):
        make_agent("dqn:/nonexistent/run", CONFIG)
    with pytest.raises(FileNotFoundError, match="no PPO checkpoint"):
        make_agent("ppo:/nonexistent/run", CONFIG)
    with pytest.raises(ValueError, match="at most 4 snakes"):
        check_game(parse_agent("mcts-8"), GameConfig(num_snakes=5))


def test_dqn_agent_plays_from_observations():
    pytest.importorskip("optax")
    if not agents_mod.DEFAULT_DQN_CHECKPOINT.is_dir():
        pytest.skip("no DQN checkpoint in this checkout")
    agent = make_agent("dqn", CONFIG)
    assert agent.needs_obs
    assert make_agent("dqn:baselines/checkpoints/dqn-duel-seed0", CONFIG) is agent
    with pytest.raises(ValueError, match="trained on 11x11"):
        make_agent("dqn", GameConfig(width=7, height=7))
    cfg = agent_config(parse_agent("dqn"))
    assert cfg["checkpoint"] == "baselines/checkpoints/dqn-duel-seed0"
    assert len(cfg["params_sha256"]) == 16 and cfg["network"]
    r = record_games(CONFIG, ["dqn", "random_legal"], jax.random.key(0), 2, max_turns=6)
    assert [g["seats"][0] for g in r["games"]] == ["dqn", "random_legal"]


def test_ppo_names():
    default = agents_mod.PPO_DEFAULT_MODE
    (other,) = set(agents_mod.PPO_MODES) - {default}
    checkpoint = agents_mod.DEFAULT_PPO_CHECKPOINT.resolve()
    spec = parse_agent("ppo")
    assert (spec.name, spec.kind, spec.checkpoint) == ("ppo", "ppo", checkpoint)
    assert spec.greedy == (default == "greedy")
    # The default mode and the default checkpoint are left out of the canonical name.
    assert parse_agent(f"ppo-{default}") == spec
    assert parse_agent(f"ppo-{default}:{checkpoint}") == spec
    assert parse_agent(f" ppo:{checkpoint} ") == spec
    explicit = parse_agent(f"ppo-{other}")
    assert explicit.name == f"ppo-{other}" and explicit.greedy == (other == "greedy")
    assert parse_agent(f"ppo-{other}:{checkpoint}") == explicit
    run = parse_agent(f"ppo-{other}:/nonexistent/run")
    assert run.name == f"ppo-{other}:/nonexistent/run"
    assert run.checkpoint == Path("/nonexistent/run")
    assert parse_agent("ppo:/nonexistent/run").name == "ppo:/nonexistent/run"
    for s in (spec, explicit, run):
        assert parse_agent(s.name) == s  # canonical names read back as the same agent


def _tiny_ppo_checkpoint(run_dir: Path) -> Path:
    ppo = agents_mod.load_ppo_module()
    cfg = ppo.PPOConfig(conv_channels=(4,), conv_strides=(2,), hidden=16)
    run_dir.mkdir()
    ppo.save_config(str(run_dir), cfg)
    ppo.save_checkpoint(str(run_dir), ppo.init_network(jax.random.key(0), cfg))
    return run_dir


def test_ppo_agents_play_greedy_or_sampled(tmp_path):
    pytest.importorskip("optax")
    run = _tiny_ppo_checkpoint(tmp_path / "ppo-run")
    greedy = make_agent(f"ppo-greedy:{run}", CONFIG)
    sample = make_agent(f"ppo-sample:{run}", CONFIG)
    default = greedy if agents_mod.PPO_DEFAULT_MODE == "greedy" else sample
    assert make_agent(f"ppo:{run}", CONFIG) is default
    assert default.name == f"ppo:{run}"
    assert greedy.needs_obs and sample.needs_obs and greedy.policy is not sample.policy
    assert "greedy self-play PPO" in greedy.description
    assert "sampled self-play PPO" in sample.description
    cfg_g, cfg_s = agent_config(parse_agent(greedy.name)), agent_config(parse_agent(sample.name))
    assert cfg_g["type"] == "ppo" and cfg_g["greedy"] and not cfg_s["greedy"]
    assert cfg_g["params_sha256"] == cfg_s["params_sha256"] and cfg_g["network"]["hidden"] == 16
    with pytest.raises(ValueError, match="trained on 11x11"):
        make_agent(f"ppo-sample:{run}", GameConfig(width=7, height=7))

    # Sampled play follows the match's keys; greedy play doesn't depend on them.
    env = agents_mod.make_env(CONFIG, True)
    state, ts = env.reset(jax.random.key(0))
    keys = jax.random.split(jax.random.key(1), 32)
    moves = {
        name: jax.vmap(agent.policy, in_axes=(0, None, None))(keys, state, ts)
        for name, agent in (("greedy", greedy), ("sample", sample))
    }
    assert len({tuple(m) for m in moves["greedy"].tolist()}) == 1
    assert len({tuple(m) for m in moves["sample"].tolist()}) > 1
    for m in moves.values():
        assert bool(jax.numpy.take_along_axis(ts.action_mask[None], m[..., None], -1).all())

    r = record_games(CONFIG, [sample, "random_legal"], jax.random.key(0), 2, max_turns=6)
    assert r["agents"] == [sample.name, "random_legal"]


def test_ppo_without_its_default_checkpoint(tmp_path, monkeypatch):
    script, module, _ = agents_mod.BASELINES["ppo"]
    monkeypatch.setitem(agents_mod.BASELINES, "ppo", (script, module, tmp_path / "missing"))
    spec = parse_agent("ppo")
    assert spec.name == "ppo" and spec.checkpoint == (tmp_path / "missing").resolve()
    with pytest.raises(FileNotFoundError, match="no PPO checkpoint in .*missing"):
        make_agent("ppo", GameConfig(max_turns=7))  # a config no other test caches


def test_mcts_agents():
    agent = make_agent("mcts-4:max_depth=2", CONFIG)
    assert agent.name == "mcts-4-depth2"
    assert not agent.needs_obs and "4 simulations, max_depth=2" in agent.description
    # One agent, however it is spelled: the policy object (and play_match's cache) is shared.
    assert agent is make_agent("mcts-4-depth2", CONFIG)
    assert make_agent("mcts-2-rollout", CONFIG) is make_agent("mcts-2:rollout_steps=10", CONFIG)
    r = record_games(CONFIG, [agent, "random_legal"], jax.random.key(0), 1, max_turns=2)
    assert len(r["games"][0]["frames"]) >= 2
    assert r["agents"] == ["mcts-4-depth2", "random_legal"]
