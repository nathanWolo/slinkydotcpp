"""Tests for the agent registry (``slinky.agents``): names, canonical forms, building agents."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from slinky import agents as agents_mod
from slinky import mcts, modes
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
from slinky.observations import CHANNEL_NAMES
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
    "rainbow": "rainbow",
    "ppo_modes": "ppo_modes",
    "ppo_modes-greedy": "ppo_modes",  # greedy is the default
    "ppo_modes-sample": "ppo_modes-sample",
    f"ppo_modes:{agents_mod.DEFAULT_PPO_MODES_CHECKPOINT}": "ppo_modes",
    "ppo_modes:game_mode=duel": "ppo_modes:game_mode=duel",
    "ppo_modes:/nonexistent/run": "ppo_modes:/nonexistent/run",
    # Items in any order (the run dir first), as for az; the game mode goes last.
    "ppo_modes-sample:game_mode=royale:/nonexistent/run": (
        "ppo_modes-sample:/nonexistent/run:game_mode=royale"
    ),
    " ppo_modes:/nonexistent/run: game_mode = standard ": (
        "ppo_modes:/nonexistent/run:game_mode=standard"
    ),
    # A snapshot (the trainer's params-<steps>.npz) goes between the run dir and the game mode.
    "ppo_modes:snapshot=1310720": "ppo_modes:snapshot=1310720",
    "ppo_modes-sample:game_mode=duel:snapshot=5:/nonexistent/run": (
        "ppo_modes-sample:/nonexistent/run:snapshot=5:game_mode=duel"
    ),
    "ppo_modes:/nonexistent/run: snapshot = 0042 ": "ppo_modes:/nonexistent/run:snapshot=42",
    # Absolute: a relative path is looked up in the repository only if it exists.
    f"rainbow:{agents_mod.DEFAULT_RAINBOW_CHECKPOINT}": "rainbow",
    "rainbow:/nonexistent/run": "rainbow:/nonexistent/run",
    "reinforce": "reinforce",
    "reinforce-greedy": "reinforce",
    "reinforce-sample:/nonexistent/run": "reinforce-sample:/nonexistent/run",
    "az-64": "az-64",
    "az-0": "az-0",
    f"az-64:{agents_mod.DEFAULT_AZ_CHECKPOINT}": "az-64",
    "az-0:/nonexistent/run": "az-0:/nonexistent/run",
    # Settings after the run dir, in MCTSConfig field order, numbers in their shortest form.
    "az-64:c_puct=1.0:/nonexistent/run": "az-64:/nonexistent/run:c_puct=1",
    "az-8:final=sample:/nonexistent/run:leaf=heuristic:c_puct=0.25": (
        "az-8:/nonexistent/run:c_puct=0.25:leaf=heuristic:final=sample"
    ),
    "az-16:spawn_food=yes:max_depth=16": "az-16:max_depth=16:spawn_food=true",
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
        ("ppo-modes", "unknown ppo mode 'modes'"),
        ("ppo_modesx", "unknown agent"),
        ("ppo_modes:", "empty item"),
        ("ppo_modes::game_mode=duel", "empty item"),
        ("ppo_modes@seed0", "write ppo_modes:<run dir>"),
        ("ppo_modes-bogus", "unknown ppo_modes play mode 'bogus'"),
        ("ppo_modes-greedy-sample", "unknown ppo_modes play mode 'greedy-sample'"),
        ("ppo_modes:/a:/b", "two run dirs"),
        ("ppo_modes:game_mode=lava", "unknown game mode 'lava'"),
        ("ppo_modes:game_mode=", "unknown game mode ''"),
        ("ppo_modes:mode=duel", "unknown ppo_modes option 'mode'"),
        ("ppo_modes:hidden=16", "unknown ppo_modes option 'hidden'"),
        ("ppo_modes:game_mode=duel:game_mode=royale", "both set game_mode"),
        ("ppo_modes:snapshot=5:snapshot=6", "'snapshot=5' and 'snapshot=6' both set snapshot"),
        ("ppo_modes:snapshot=0", "snapshot must be a positive whole number"),
        ("ppo_modes:snapshot=", "snapshot must be a positive whole number"),
        ("ppo_modes:snapshot=-5", "snapshot must be a positive whole number"),
        ("ppo_modes:snapshot=1.5", "snapshot must be a positive whole number"),
        ("ppo_modes:snapshot=1_000", "snapshot must be a positive whole number"),
        ("ppo_modes:snapshot=last", "snapshot must be a positive whole number"),
        ("rainbow:", "write rainbow:<run dir>"),
        ("rainbow@seed0", "write rainbow:<run dir>"),
        ("reinforce:", "write reinforce:<run dir>"),
        ("reinforce@seed0", "write reinforce:<run dir>"),
        ("reinforce-bogus", "unknown reinforce mode 'bogus'"),
        ("az", "bad az agent name"),
        ("az-x", "bad az agent name"),
        ("az@seed0", "bad az agent name"),
        ("az-64:", "empty item"),
        ("az-64::c_puct=1", "empty item"),
        ("az-64:/a:/b", "two run dirs"),
        ("az-0:c_puct=1", "az-0 plays the raw network"),
        ("az-64:bogus=1", "unknown az option 'bogus'"),
        ("az-64:selection=duct", "az option 'selection' in 'az-64:selection=duct' can't be set"),
        ("az-64:root_noise_frac=0.25", "can't be set"),
        ("az-64:c_puct=1:c_puct=2", "'c_puct=1' and 'c_puct=2' both set c_puct"),
        ("az-64:leaf=foo", "leaf must be one of"),
        ("az-64:c_puct=lots", "expected a float"),
        ("az-16:max_depth=20", "caps the depth"),
    ],
)
def test_bad_names(name, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_agent(name)


def test_agent_configs_are_complete_json():
    cfg = agent_config(parse_agent("mcts-64-rm"))
    fields = {f.name for f in dataclasses.fields(mcts.MCTSConfig)}
    puct_only = set(agents_mod._PUCT_ONLY_FIELDS)
    assert set(cfg) == fields - puct_only | {"type"} and cfg["selection"] == "rm"
    assert isinstance(cfg["weights"], dict) and json.loads(json.dumps(cfg)) == cfg
    assert set(agent_config(parse_agent("mcts-64:selection=puct"))) == fields | {"type"}
    # Without PUCT the config, and so every stored DUCT and RM benchmark result's identity,
    # is what it was before PUCT's fields existed.
    assert set(agent_config(parse_agent("mcts-64"))) == {
        "draw_value", "exploration", "final", "leaf", "max_depth", "num_simulations",
        "rm_gamma", "rollout_policy", "rollout_steps", "selection", "spawn_food", "tie_noise",
        "type", "ucb1_tuned", "weights",
    }  # fmt: skip
    assert agent_config(parse_agent("random")) == {"type": "random"}
    assert set(agent_config(parse_agent("heuristic"))["weights"]) >= {"territory", "contempt"}
    with pytest.raises(FileNotFoundError, match="no DQN checkpoint"):
        agent_config(parse_agent("dqn:/nonexistent/run"))
    with pytest.raises(FileNotFoundError, match="no PPO checkpoint"):
        agent_config(parse_agent("ppo-sample:/nonexistent/run"))
    with pytest.raises(FileNotFoundError, match="no AlphaZero checkpoint"):
        agent_config(parse_agent("az-64:/nonexistent/run"))


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


def test_rainbow_agent_plays_from_observations(tmp_path):
    pytest.importorskip("optax")
    rainbow = agents_mod.load_rainbow_module()
    cfg = rainbow.RainbowConfig(conv_channels=(4,), conv_strides=(2,), hidden=16, num_atoms=11)
    rainbow.save_config(str(tmp_path), cfg)
    rainbow.save_checkpoint(str(tmp_path), rainbow.init_network(jax.random.key(0), cfg))
    name = f"rainbow:{tmp_path}"
    agent = make_agent(name, CONFIG)
    assert agent.needs_obs and "Rainbow" in agent.description
    assert parse_agent(name).kind == "rainbow"
    info = agent_config(parse_agent(name))
    assert info["type"] == "rainbow" and info["network"]["num_atoms"] == 11
    with pytest.raises(ValueError, match="trained on 11x11"):
        make_agent(name, GameConfig(width=7, height=7))
    with pytest.raises(FileNotFoundError, match="no Rainbow checkpoint"):
        make_agent("rainbow:/nonexistent/run", CONFIG)
    if agents_mod.DEFAULT_DQN_CHECKPOINT.is_dir():  # a DQN run is not a Rainbow run
        with pytest.raises(ValueError, match="not a Rainbow run"):
            make_agent("rainbow:baselines/checkpoints/dqn-duel-seed0", CONFIG)
    # ... and a Rainbow run is neither a DQN nor a PPO run: refused before anything is built.
    for kind in ("dqn", "ppo"):
        with pytest.raises(ValueError, match="is a rainbow run; write rainbow:<run dir>"):
            check_game(parse_agent(f"{kind}:{tmp_path}"), CONFIG)
    r = record_games(CONFIG, [name, "random_legal"], jax.random.key(0), 2, max_turns=6)
    assert r["games"][0]["seats"][0] == parse_agent(name).name


def test_reinforce_agent_plays_from_observations(tmp_path):
    pytest.importorskip("optax")
    reinforce = agents_mod.load_reinforce_module()
    cfg = reinforce.ReinforceConfig(conv_channels=(4,), conv_strides=(2,), hidden=16)
    reinforce.save_config(str(tmp_path), cfg)
    reinforce.save_checkpoint(str(tmp_path), reinforce.init_network(jax.random.key(0), cfg))
    for mode, how in (("", "greedy"), ("-sample", "sampled")):
        name = f"reinforce{mode}:{tmp_path}"
        agent = make_agent(name, CONFIG)
        assert agent.needs_obs and f"{how} self-play REINFORCE" in agent.description
        spec = parse_agent(name)
        assert spec.kind == "reinforce" and spec.greedy == (how == "greedy")
        info = agent_config(spec)
        assert info["type"] == "reinforce" and info["network"]["hidden"] == 16
        r = record_games(CONFIG, [name, "random_legal"], jax.random.key(0), 2, max_turns=6)
        assert r["games"][0]["seats"][0] == spec.name
    with pytest.raises(ValueError, match="trained on 11x11"):
        make_agent(f"reinforce:{tmp_path}", GameConfig(width=7, height=7))
    with pytest.raises(FileNotFoundError, match="no REINFORCE checkpoint"):
        make_agent("reinforce:/nonexistent/run", CONFIG)
    if agents_mod.DEFAULT_PPO_CHECKPOINT.is_dir():  # a PPO run is not a REINFORCE run
        with pytest.raises(ValueError, match="not a REINFORCE run"):
            make_agent("reinforce:baselines/checkpoints/ppo-duel-seed0", CONFIG)
    # ... and a REINFORCE run is not a PPO run: refused before anything is built.
    with pytest.raises(ValueError, match="is a reinforce run; write reinforce:<run dir>"):
        check_game(parse_agent(f"ppo:{tmp_path}"), CONFIG)


def test_az_names():
    spec = parse_agent("az-64:c_puct=0.75")
    assert (spec.kind, spec.sims, spec.overrides) == ("az", 64, (("c_puct", 0.75),))
    assert spec.checkpoint == agents_mod.DEFAULT_AZ_CHECKPOINT.resolve()
    # The name fixes the search's structure (enough for the benchmark's cost and memory models).
    assert (spec.mcts.selection, spec.mcts.prior, spec.mcts.leaf) == ("puct", "network", "network")
    assert spec.mcts.num_simulations == 64 and spec.mcts.c_puct == 0.75
    raw = parse_agent("az-0")
    assert raw.mcts is None and raw.sims == 0 and raw.overrides == ()


def _tiny_az_checkpoint(run_dir: Path, **kw) -> Path:
    az = agents_mod.load_az_module()
    cfg = az.AZConfig(conv_channels=(4,), conv_strides=(2,), hidden=16, **kw)
    run_dir.mkdir()
    az.save_config(str(run_dir), cfg)
    az.save_checkpoint(str(run_dir), az.init_network(jax.random.key(0), cfg))
    return run_dir


def test_az_agents_play_raw_or_with_search(tmp_path):
    pytest.importorskip("optax")
    run = _tiny_az_checkpoint(tmp_path / "az-run", c_puct=0.3, draw_value=-0.25)
    raw, searched = make_agent(f"az-0:{run}", CONFIG), make_agent(f"az-4:{run}", CONFIG)
    assert not raw.needs_obs and not searched.needs_obs  # observations come from the state
    assert "raw AlphaZero network" in raw.description
    assert "4 simulations" in searched.description
    assert make_agent(f"az-4:{run}", CONFIG) is searched
    # The search uses the run's constants, and the name's settings over them.
    info = agent_config(parse_agent(searched.name))
    assert info["type"] == "az" and info["simulations"] == 4 and info["network"]["hidden"] == 16
    assert (info["search"]["c_puct"], info["search"]["draw_value"]) == (0.3, -0.25)
    assert info["search"]["root_noise_frac"] == 0.0 and info["search"]["final"] == "max"
    assert agent_config(parse_agent(f"az-4:{run}:c_puct=0.7"))["search"]["c_puct"] == 0.7
    assert agent_config(parse_agent(raw.name))["search"] is None
    assert info["params_sha256"] == agent_config(parse_agent(raw.name))["params_sha256"]
    with pytest.raises(ValueError, match="trained on 11x11"):
        make_agent(f"az-4:{run}", GameConfig(width=7, height=7))
    with pytest.raises(ValueError, match="at most 4 snakes"):
        check_game(parse_agent(f"az-4:{run}"), GameConfig(num_snakes=5))
    with pytest.raises(FileNotFoundError, match="no AlphaZero checkpoint"):
        make_agent("az-4:/nonexistent/run", CONFIG)
    # An AlphaZero run is not a PPO run, and a PPO run is not an AlphaZero run.
    with pytest.raises(ValueError, match="is an az run; write az-<n>:<run dir>"):
        check_game(parse_agent(f"ppo:{run}"), CONFIG)
    ppo_run = _tiny_ppo_checkpoint(tmp_path / "ppo-run")
    with pytest.raises(ValueError, match="not an AlphaZero run"):
        make_agent(f"az-0:{ppo_run}", CONFIG)
    r = record_games(CONFIG, [searched, raw], jax.random.key(0), 2, max_turns=6)
    assert r["agents"] == [searched.name, raw.name]


def test_az_without_its_default_checkpoint(tmp_path, monkeypatch):
    script, module, _ = agents_mod.BASELINES["az"]
    monkeypatch.setitem(agents_mod.BASELINES, "az", (script, module, tmp_path / "missing"))
    spec = parse_agent("az-8")
    assert spec.name == "az-8" and spec.checkpoint == (tmp_path / "missing").resolve()
    with pytest.raises(FileNotFoundError, match="no AlphaZero checkpoint in .*missing"):
        make_agent("az-8", GameConfig(max_turns=7))


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


# --- ppo_modes: the multi-mode PPO baseline ------------------------------------------

TINY_MODES = dict(conv_channels=(4,), conv_strides=(2,), hidden=16)


def _tiny_modes_run(run_dir: Path, game_modes=modes.MODE_NAMES, seed: int = 0, **kw) -> Path:
    """A saved ``ppo_modes`` run with a tiny network and *non-zero* ``feat/w`` and policy head.

    (A fresh network has zeros in ``feat/w``, so its features would change nothing and a
    registry agent that dropped them could not be told from a correct one.)
    """
    pm = agents_mod.load_ppo_modes_module()
    cfg = pm.ModesConfig(game_modes=tuple(game_modes), **TINY_MODES, **kw)
    params = pm.init_network(jax.random.key(seed), cfg)
    k_feat, k_policy = jax.random.split(jax.random.key(seed + 1))
    params["policy"] = {
        **params["policy"],
        "w": jax.random.normal(k_policy, params["policy"]["w"].shape) * 0.3,
    }
    if "feat" in params:
        params["feat"] = {"w": jax.random.normal(k_feat, params["feat"]["w"].shape) * 0.5}
    run_dir.mkdir()
    pm.save_config(str(run_dir), cfg)
    pm.save_checkpoint(str(run_dir), params)
    return run_dir


def test_ppo_modes_names():
    checkpoint = agents_mod.DEFAULT_PPO_MODES_CHECKPOINT.resolve()
    spec = parse_agent("ppo_modes")
    assert (spec.name, spec.kind, spec.checkpoint) == ("ppo_modes", "ppo_modes", checkpoint)
    assert spec.greedy is True and spec.game_mode is None
    assert checkpoint.name == "ppo-modes-seed0"
    assert agents_mod.BASELINES["ppo_modes"][1] == "baselines_ppo_modes"
    # The default play mode and the default checkpoint are left out of the canonical name.
    assert parse_agent("ppo_modes-greedy") == spec
    assert parse_agent(f"ppo_modes-greedy:{checkpoint}") == spec
    sample = parse_agent("ppo_modes-sample")
    assert sample.name == "ppo_modes-sample" and sample.greedy is False
    told = parse_agent("ppo_modes:/nonexistent/run:game_mode=duel")
    assert told.game_mode == "duel" and told.checkpoint == Path("/nonexistent/run")
    assert told.name == "ppo_modes:/nonexistent/run:game_mode=duel"
    assert parse_agent("ppo_modes:game_mode=royale").name == "ppo_modes:game_mode=royale"
    for s in (spec, sample, told, parse_agent("ppo_modes-sample:game_mode=constrictor")):
        assert parse_agent(s.name) == s  # canonical names read back as the same agent
    # Distinct agents, distinct names: the told mode and the play mode are part of the name.
    names = {parse_agent(f"ppo_modes{h}{t}").name for h in ("", "-sample") for t in
             ("", *(f":game_mode={m}" for m in modes.MODE_NAMES))}  # fmt: skip
    assert len(names) == 2 * (1 + len(modes.MODE_NAMES))
    # The name belongs to this kind alone, and the other PPO names mean what they did.
    assert parse_agent("ppo").kind == "ppo" and parse_agent("ppo").game_mode is None
    assert parse_agent("ppo-sample:/nonexistent/run").kind == "ppo"
    assert parse_agent("ppo:/nonexistent/run").name == "ppo:/nonexistent/run"


def test_ppo_modes_without_its_default_checkpoint(tmp_path, monkeypatch):
    script, module, _ = agents_mod.BASELINES["ppo_modes"]
    monkeypatch.setitem(agents_mod.BASELINES, "ppo_modes", (script, module, tmp_path / "missing"))
    spec = parse_agent("ppo_modes")
    assert spec.name == "ppo_modes" and spec.checkpoint == (tmp_path / "missing").resolve()
    # A clear error, naming the missing directory, from every way in.
    for call in (
        lambda: make_agent("ppo_modes", GameConfig(max_turns=7)),  # a config no test caches
        lambda: make_agent("ppo_modes:game_mode=duel", GameConfig(max_turns=7)),
        lambda: agent_config(spec),
        lambda: check_game(spec, CONFIG),
    ):
        with pytest.raises(FileNotFoundError, match="no multi-mode PPO checkpoint in .*missing"):
            call()


def test_ppo_modes_agent_config(tmp_path):
    ppo_run = _tiny_ppo_checkpoint(tmp_path / "ppo-run")
    run = _tiny_modes_run(tmp_path / "modes-run")
    cfg = agent_config(parse_agent(f"ppo_modes:{run}"))
    digest = hashlib.sha256((run / "params.npz").read_bytes()).hexdigest()[:16]
    assert cfg["type"] == "ppo_modes" and cfg["greedy"] is True and cfg["game_mode"] is None
    assert cfg["params_sha256"] == digest and cfg["checkpoint"] == str(run.resolve())
    assert cfg["network"]["hidden"] == 16 and cfg["network"]["game_modes"] == list(modes.MODE_NAMES)
    assert json.loads(json.dumps(cfg)) == cfg
    # The told mode and the play mode are part of the identity; the parameters are shared.
    duel = agent_config(parse_agent(f"ppo_modes:{run}:game_mode=duel"))
    sample = agent_config(parse_agent(f"ppo_modes-sample:{run}"))
    assert duel["game_mode"] == "duel" and duel != cfg
    assert sample["greedy"] is False and sample != cfg
    assert {c["params_sha256"] for c in (cfg, duel, sample)} == {digest}
    assert {c["game_mode"] for c in (cfg, sample)} == {None}

    # No other kind has the key, and their identities are exactly what they were.
    ppo_cfg = agent_config(parse_agent(f"ppo:{ppo_run}"))
    ppo = agents_mod.load_ppo_module()
    assert ppo_cfg == {
        "type": "ppo",
        "checkpoint": str(ppo_run.resolve()),
        "greedy": True,
        "params_sha256": hashlib.sha256((ppo_run / "params.npz").read_bytes()).hexdigest()[:16],
        "network": json.loads(json.dumps(dataclasses.asdict(ppo.load_config(str(ppo_run))))),
    }
    assert set(ppo_cfg) == {"type", "checkpoint", "greedy", "params_sha256", "network"}
    az_run = _tiny_az_checkpoint(tmp_path / "az-run")
    az_keys = {"type", "checkpoint", "greedy", "params_sha256", "network", "simulations", "search"}
    for name in (f"az-0:{az_run}", f"az-2:{az_run}"):
        assert set(agent_config(parse_agent(name))) == az_keys
    for name in ("random_legal", "random", "heuristic", "mcts-4"):
        assert "game_mode" not in agent_config(parse_agent(name))


def test_ppo_modes_snapshots(tmp_path, monkeypatch):
    """``:snapshot=<steps>`` plays ``params-<steps>.npz``: its parameters, hash and name."""
    pm = agents_mod.load_ppo_modes_module()
    run = _tiny_modes_run(tmp_path / "run", seed=0)
    cfg = pm.load_config(str(run))
    other = pm.init_network(jax.random.key(5), cfg)  # a snapshot differing from the final params
    other["policy"] = {
        **other["policy"],
        "w": jax.random.normal(jax.random.key(6), other["policy"]["w"].shape) * 0.3,
    }
    other["feat"] = {"w": jax.random.normal(jax.random.key(7), other["feat"]["w"].shape) * 0.5}
    pm.save_checkpoint(str(run), other, pm.snapshot_name(1024))

    spec = parse_agent(f"ppo_modes:{run}:snapshot=1024")
    assert spec.snapshot == 1024 and spec.name == f"ppo_modes:{run}:snapshot=1024"
    assert parse_agent(spec.name) == spec and parse_agent(f"ppo_modes:{run}").snapshot is None
    # The file is the trainer's snapshot file, and the identity follows it.
    assert agents_mod._checkpoint_files(spec)[0] == run / pm.snapshot_name(1024)
    assert agents_mod._checkpoint_files(parse_agent(f"ppo_modes:{run}"))[0] == run / "params.npz"
    final, snap = agent_config(parse_agent(f"ppo_modes:{run}")), agent_config(spec)
    digest = hashlib.sha256((run / "params-1024.npz").read_bytes()).hexdigest()[:16]
    assert snap["params_sha256"] == digest != final["params_sha256"]
    assert (snap["snapshot"], final["snapshot"]) == (1024, None)
    assert {k: v for k, v in snap.items() if k not in ("params_sha256", "snapshot")} == {
        k: v for k, v in final.items() if k not in ("params_sha256", "snapshot")
    }
    assert agent_config(parse_agent(f"ppo_modes:{run}:snapshot=1024:game_mode=duel")) != snap

    # The agent plays the snapshot's parameters, not the final ones.
    network, recorded = _recording_network(pm, monkeypatch)
    params = {
        name: pm.load_params(str(run), cfg, name=name)
        for name in ("params.npz", pm.snapshot_name(1024))
    }
    group = pm.make_groups(cfg)[0]
    state, ts = _states_of(group.game, {6})[0]
    states = jax.tree.map(lambda x: x[None], state)
    boards, feats, _ = pm.group_inputs(cfg, group, states)
    want = {n: np.asarray(network(p, boards, feats, cfg)[0])[0] for n, p in params.items()}
    assert not np.allclose(*want.values(), atol=1e-3)  # the two sets of parameters differ
    for given, name in ((f"ppo_modes:{run}:snapshot=1024", pm.snapshot_name(1024)),
                        (f"ppo_modes:{run}", "params.npz")):  # fmt: skip
        make_agent(given, group.game).policy(jax.random.key(0), state, ts)
        np.testing.assert_allclose(recorded[-1], want[name], rtol=1e-4, atol=1e-5)
    described = make_agent(spec.name, group.game).description
    assert (
        "snapshot 1024" in described
        and "snapshot" not in make_agent(f"ppo_modes:{run}", group.game).description
    )

    # A run directory that has only the snapshots plays them; a missing one is a clear error.
    only = tmp_path / "only"
    only.mkdir()
    for name in ("config.json", pm.snapshot_name(1024)):
        (only / name).write_bytes((run / name).read_bytes())
    assert make_agent(f"ppo_modes:{only}:snapshot=1024", group.game).needs_obs
    with pytest.raises(FileNotFoundError, match="no multi-mode PPO checkpoint in .*only"):
        agent_config(parse_agent(f"ppo_modes:{only}"))  # the final params.npz is not there
    for call in (
        lambda: agent_config(parse_agent(f"ppo_modes:{run}:snapshot=7")),
        lambda: check_game(parse_agent(f"ppo_modes:{run}:snapshot=7"), CONFIG),
        lambda: make_agent(f"ppo_modes:{run}:snapshot=7", GameConfig(max_turns=9)),
    ):
        with pytest.raises(FileNotFoundError, match=r"expected params-7\.npz and config\.json"):
            call()


def test_ppo_modes_runs_and_other_runs_are_not_mixed_up(tmp_path):
    run = _tiny_modes_run(tmp_path / "modes-run")
    ppo_run = _tiny_ppo_checkpoint(tmp_path / "ppo-run")
    az_run = _tiny_az_checkpoint(tmp_path / "az-run")
    # ppo:<ppo_modes run> (and every other kind) is refused before anything is built ...
    for kind in ("dqn", "ppo", "ppo-sample", "rainbow"):
        with pytest.raises(ValueError, match="is a ppo_modes run; write ppo_modes:<run dir>"):
            check_game(parse_agent(f"{kind}:{run}"), CONFIG)
    with pytest.raises(ValueError, match="is a ppo_modes run; write ppo_modes:<run dir>"):
        make_agent(f"ppo:{run}", CONFIG)
    with pytest.raises(ValueError, match="is a ppo_modes run; write ppo_modes:<run dir>"):
        make_agent(f"az-0:{run}", CONFIG)
    # ... and a ppo_modes agent refuses the runs of the others.
    with pytest.raises(ValueError, match="not a ppo_modes run"):
        make_agent(f"ppo_modes:{ppo_run}", CONFIG)
    with pytest.raises(ValueError, match="is an az run; write az-<n>:<run dir>"):
        check_game(parse_agent(f"ppo_modes:{az_run}"), CONFIG)
    with pytest.raises(FileNotFoundError, match="no multi-mode PPO checkpoint"):
        make_agent("ppo_modes:/nonexistent/run", CONFIG)


def test_ppo_modes_check_game(tmp_path):
    general = _tiny_modes_run(tmp_path / "general")
    royale = _tiny_modes_run(tmp_path / "royale", game_modes=("royale",))
    pair = _tiny_modes_run(tmp_path / "pair", game_modes=("duel", "constrictor"))
    # A generalist plays every mode, and is told the mode of the game it plays.
    for mode in modes.MODE_NAMES:
        game = modes.mode_config(mode)
        agent = make_agent(f"ppo_modes:{general}", game)
        assert agent.needs_obs and f"told {mode}" in agent.description
        assert make_agent(f"ppo_modes:{general}", game) is agent  # cached per (name, game)
    assert make_agent(f"ppo_modes:{general}", modes.mode_config("duel")) is not make_agent(
        f"ppo_modes:{general}", modes.mode_config("standard")
    )  # one policy, so one compile, per mode
    # ... also with 3 snakes (still standard) and other snake counts of a mode.
    for game in (GameConfig(num_snakes=3), modes.mode_config("constrictor", 2)):
        check_game(parse_agent(f"ppo_modes:{general}"), game)

    # A specialist is refused every other mode's game ...
    check_game(parse_agent(f"ppo_modes:{royale}"), modes.mode_config("royale"))
    for mode in ("standard", "duel", "constrictor"):
        with pytest.raises(ValueError, match=rf"trained on royale only and can't play a {mode}"):
            check_game(parse_agent(f"ppo_modes:{royale}"), modes.mode_config(mode))
    with pytest.raises(ValueError, match="can't play a standard game; write :game_mode="):
        make_agent(f"ppo_modes:{pair}", modes.mode_config("standard"))
    check_game(parse_agent(f"ppo_modes:{pair}"), modes.mode_config("constrictor"))
    # ... unless it is told its own mode, the cross-mode matrix.
    told_royale = parse_agent(f"ppo_modes:{royale}:game_mode=royale")
    for mode in ("standard", "duel", "constrictor"):
        check_game(told_royale, modes.mode_config(mode))
    cross = make_agent(f"ppo_modes:{royale}:game_mode=royale", modes.mode_config("duel"))
    assert (
        cross.name == f"ppo_modes:{royale}:game_mode=royale" and "told royale" in cross.description
    )
    # A told mode outside the run's modes is refused whatever the game is.
    for game in (modes.mode_config("royale"), modes.mode_config("duel")):
        with pytest.raises(ValueError, match="can't be told game_mode=standard"):
            check_game(parse_agent(f"ppo_modes:{royale}:game_mode=standard"), game)
    with pytest.raises(ValueError, match="can't be told game_mode=royale"):
        check_game(parse_agent(f"ppo_modes:{pair}:game_mode=royale"), modes.mode_config("duel"))
    # The wrong-mode ablation: a generalist told another mode.
    check_game(parse_agent(f"ppo_modes:{general}:game_mode=duel"), modes.mode_config("standard"))

    # Games outside the four modes, and other board sizes, are refused.
    with pytest.raises(ValueError, match="no game mode for ruleset='standard' map='empty'"):
        check_game(parse_agent(f"ppo_modes:{general}"), GameConfig(map="empty"))
    check_game(parse_agent(f"ppo_modes:{general}:game_mode=standard"), GameConfig(map="empty"))
    with pytest.raises(ValueError, match="trained on 11x11"):
        make_agent(f"ppo_modes:{general}", GameConfig(width=7, height=7))
    with pytest.raises(ValueError, match="no game mode"):
        make_agent(f"ppo_modes:{general}", GameConfig(num_snakes=1))


FOOD = CHANNEL_NAMES.index("food")
# Turn 0, a few turns in, and (royale) past the first shrink at turn 25: so the turn, the
# shrink phase and the share of opponents alive take different values.
STATE_TURNS = (0, 6, 40)


def _states_of(game, turns):
    """The ``(state, timestep)`` of one game (random legal play) at each of ``turns``."""
    env = agents_mod.make_env(game, True)
    step = jax.jit(env.step)
    random_legal = make_agent("random_legal", game).policy
    state, ts = jax.jit(env.reset)(jax.random.key(7))
    out = []
    for t in range(max(turns) + 1):
        if t in turns:
            out.append((state, ts))
        key = jax.random.key(100 + t)
        state, ts = step(key, state, random_legal(key, state, ts))
    return out


def _recording_network(pm, monkeypatch):
    """Wrap ``pm.network`` (which the policies look up at call time) to record the logits.

    Returns ``(original network, list of the logits of every call)``. Only calls that run
    un-jitted record concrete values, so use it with a policy called directly.
    """
    original, recorded = pm.network, []

    def recording(params, obs, feats, cfg):
        out = original(params, obs, feats, cfg)
        recorded.append(np.asarray(out[0]))
        return out

    monkeypatch.setattr(pm, "network", recording)
    return original, recorded


def test_ppo_modes_logits_equal_the_trainers_network(tmp_path, monkeypatch):
    """The registry agent computes what the trainer's network does on the same states.

    The trainer's inputs come from ``group_inputs`` (egocentric boards with the food
    hidden in constrictor, and the mode features) and its network from ``network``; the
    agent reaches the same functions through ``make_policy``. The agent's logits are
    recorded by wrapping ``network`` while its (un-jitted) policy runs, so what is compared
    is the logits themselves, not only the moves made from them.
    """
    pm = agents_mod.load_ppo_modes_module()
    run = _tiny_modes_run(tmp_path / "run")
    cfg = pm.load_config(str(run))
    params = pm.load_params(str(run), cfg)
    network, recorded = _recording_network(pm, monkeypatch)
    for group in pm.make_groups(cfg):
        game, mode = group.game, group.name
        greedy = make_agent(f"ppo_modes:{run}", game)
        sample = make_agent(f"ppo_modes-sample:{run}", game)
        pairs = _states_of(game, STATE_TURNS)
        states = jax.tree.map(lambda *xs: jnp.stack(xs), *(s for s, _ in pairs))
        boards, feats, mask = pm.group_inputs(cfg, group, states)  # the trainer's inputs
        expected = np.asarray(network(params, boards, feats, cfg)[0])  # [E, N, 4]
        assert len({tuple(np.round(f[0, 4:], 4)) for f in np.asarray(feats)}) > 1  # they vary
        for i, (state, ts) in enumerate(pairs):
            del recorded[:]
            actions = np.asarray(greedy.policy(jax.random.key(i), state, ts))
            np.testing.assert_allclose(recorded[-1], expected[i], rtol=1e-4, atol=1e-5)
            masked = np.where(np.asarray(mask[i]), expected[i], -np.inf)
            np.testing.assert_array_equal(actions, masked.argmax(axis=-1))
            for k in range(4):  # sampled play draws from the same distribution, same keys
                key = jax.random.key(10 * i + k)
                want = pm.sample_actions(key, jnp.asarray(expected[i]), mask[i])
                np.testing.assert_array_equal(sample.policy(key, state, ts), want)

        # The check would pass if the agent dropped the features or kept the food: it must not.
        blind = np.asarray(network(params, boards, jnp.zeros_like(feats), cfg)[0])
        assert not np.allclose(blind, expected, atol=1e-3), mode  # the features matter
        raw = jnp.stack([ts.obs for _, ts in pairs])  # the boards as the env reports them
        assert float(jnp.abs(raw[..., FOOD]).max()) > 0.0, mode  # (food is on every board)
        if mode == "constrictor":
            assert float(jnp.abs(boards[..., FOOD]).max()) == 0.0  # hidden from the network
            shown = np.asarray(network(params, raw, feats, cfg)[0])
            assert not np.allclose(shown, expected, atol=1e-3)  # and hiding it matters
        else:
            np.testing.assert_array_equal(np.asarray(boards), np.asarray(raw))


def test_ppo_modes_food_follows_the_game_and_features_the_told_mode(tmp_path, monkeypatch):
    """The food follows the game's ruleset and the features the told mode, never the other way."""
    pm = agents_mod.load_ppo_modes_module()
    run = _tiny_modes_run(tmp_path / "run")
    cfg = pm.load_config(str(run))
    params = pm.load_params(str(run), cfg)
    network, recorded = _recording_network(pm, monkeypatch)
    for game_mode, told in (("standard", "constrictor"), ("constrictor", "standard")):
        game = cfg.games[game_mode]
        state, ts = _states_of(game, {5})[0]
        agent = make_agent(f"ppo_modes:{run}:game_mode={told}", game)
        assert f"told {told}" in agent.description
        agent.policy(jax.random.key(0), state, ts)
        # Hidden exactly when the game is constrictor; the features say the told mode.
        board = ts.obs.at[..., FOOD].set(0.0) if game_mode == "constrictor" else ts.obs
        assert float(jnp.abs(ts.obs[..., FOOD]).max()) > 0.0
        want = np.asarray(network(params, board, modes.mode_features(state, game, told), cfg)[0])
        np.testing.assert_allclose(recorded[-1], want, rtol=1e-4, atol=1e-5)
        # The told mode is not the game's own: the logits differ from an untold agent's.
        own = make_agent(f"ppo_modes:{run}", game)
        own.policy(jax.random.key(0), state, ts)
        assert not np.allclose(recorded[-1], want, atol=1e-3)
        assert own.name != agent.name and own.policy is not agent.policy


def test_ppo_modes_agent_plays_and_records(tmp_path):
    run = _tiny_modes_run(tmp_path / "run")
    for mode in ("duel", "royale"):
        game = dataclasses.replace(modes.mode_config(mode), max_turns=8)
        names = [f"ppo_modes:{run}"] + ["heuristic"] * (game.num_snakes - 1)
        r = record_games(game, names, jax.random.key(0), 2, max_turns=8, rotate=True)
        assert r["games"][0]["seats"][0] == f"ppo_modes:{run}"
        assert r["ruleset"] == game.ruleset.value and r["map"] == game.map
        assert all(len(g["frames"]) > 1 for g in r["games"])
