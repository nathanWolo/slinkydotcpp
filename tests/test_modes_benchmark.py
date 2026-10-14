"""Tests for the mode benchmark (``benchmarks/modes.py``): tiny cells with cheap agents."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from slinky import agents as registry
from slinky import modes
from slinky.evaluate import play_table

pytest.importorskip("optax")

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


M = _load("benchmarks_modes", "modes.py")

# Cheap agents (no network, no heuristic): policies compile in about a second.
CHEAP = ["--agent", "G=random_legal", "--agent", "S=random", "--agent", "R=random_legal"]
CHEAP_NAMES = {"G": "random_legal", "S": "random", "R": "random_legal", "H": "random"}


def run(out: Path, *args: str) -> int:
    return M.main([*args, "--out", str(out)])


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def cell(mode: str, seats: str, games: int = 8, names=None):
    return M.make_cell(mode, seats.split(","), names or CHEAP_NAMES, games)


SETTINGS = M.Settings(max_turns=1000, seed=0, slots=32)


def direct(c, games: int, seed: int = 0):
    """The cell's table played straight through ``play_table``, with its per-game values."""
    built = {x: registry.make_agent(c.agents[x].spec, c.game) for x in c.policies}
    env = registry.make_env(c.game, any(a.needs_obs for a in built.values()))
    return play_table(
        env, tuple(built[x].policy for x in c.policies), c.lineup, M.cell_key(seed, c.game_mode),
        games, max_turns=1000, per_game=True,
    )  # fmt: skip


def _tiny_modes_run(run_dir: Path, game_modes=modes.MODE_NAMES, seed: int = 0) -> Path:
    pm = registry.load_ppo_modes_module()
    cfg = pm.ModesConfig(
        game_modes=tuple(game_modes), conv_channels=(4,), conv_strides=(2,), hidden=16
    )
    params = pm.init_network(jax.random.key(seed), cfg)
    k_feat, k_policy = jax.random.split(jax.random.key(seed + 1))
    params["policy"] = {
        **params["policy"],
        "w": jax.random.normal(k_policy, params["policy"]["w"].shape) * 0.3,
    }
    params["feat"] = {"w": jax.random.normal(k_feat, params["feat"]["w"].shape) * 0.5}
    run_dir.mkdir()
    pm.save_config(str(run_dir), cfg)
    pm.save_checkpoint(str(run_dir), params)
    return run_dir


# --- Identity, keys, encoding ----------------------------------------------------------


def test_cell_identity_covers_everything_but_the_games():
    base = cell("standard", "G,R,R,R", games=8)
    same = cell("standard", "G,R,R,R", games=4096)
    ident = M.cell_id(base, SETTINGS)
    assert M.cell_id(same, SETTINGS) == ident  # the number of games is not part of it
    assert len(ident) == 16 and M.cell_id(base, M.Settings(1000, 0, 4)) == ident  # nor the slots
    other_r = {**CHEAP_NAMES, "R": "random"}
    others = {
        "seats": M.cell_id(cell("standard", "R,G,R,R"), SETTINGS),
        "lineup size": M.cell_id(cell("standard", "G,R,R"), SETTINGS),
        "mode": M.cell_id(cell("royale", "G,R,R,R"), SETTINGS),
        "agent": M.cell_id(cell("standard", "S,R,R,R"), SETTINGS),
        "other agent": M.cell_id(cell("standard", "G,R,R,R", names=other_r), SETTINGS),
        "max_turns": M.cell_id(base, M.Settings(500, 0, 32)),
        "seed": M.cell_id(base, M.Settings(1000, 1, 32)),
    }  # fmt: skip
    assert len({ident, *others.values()}) == 1 + len(others), others
    # An agent that is not in the line-up does not matter.
    spare = cell("standard", "G,R,R,R", names={**CHEAP_NAMES, "S": "heuristic"})
    assert M.cell_id(spare, SETTINGS) == ident


def test_cell_identity_follows_the_agents_full_config(tmp_path):
    a = _tiny_modes_run(tmp_path / "a", seed=0)
    b = _tiny_modes_run(tmp_path / "b", seed=1)
    names = {"G": f"ppo_modes:{a}", "R": "random_legal"}
    c1 = M.make_cell("duel", ["G", "R"], names, 8)
    c2 = M.make_cell("duel", ["G", "R"], {**names, "G": f"ppo_modes:{b}"}, 8)
    c3 = M.make_cell("duel", ["G", "R"], {**names, "G": f"ppo_modes:{a}:game_mode=standard"}, 8)
    assert len({M.cell_id(c, SETTINGS) for c in (c1, c2, c3)}) == 3  # parameters, told mode
    assert (
        c1.agents["G"].config["game_mode"] is None
        and c3.agents["G"].config["game_mode"] == "standard"
    )


def test_cells_of_a_mode_share_their_games(tmp_path):
    def data(seed, mode):
        return np.asarray(jax.random.key_data(M.cell_key(seed, mode)))

    assert (data(0, "duel") == data(0, "duel")).all()
    assert not (data(0, "duel") == data(0, "royale")).all()
    assert not (data(0, "duel") == data(1, "duel")).all()
    # The key does not depend on who plays or on the symbols: two cells of one mode with the same
    # agents under other names are the same games, play for play.
    out = tmp_path / "modes.jsonl"
    same = ["--agent", "G=random_legal", "--agent", "S=random_legal", "--agent", "R=random_legal"]
    for seats in ("G,R,R,R", "S,R,R,R"):
        assert run(out, "--game-mode", "standard", "--seats", seats, *same, "--games", "12") == 0
    first, second = lines(out)
    assert first["cell_id"] != second["cell_id"]
    for key in ("share", "place2"):
        assert first["per_game"][key]["G"] == second["per_game"][key]["S"]


def test_per_game_encoding_round_trip():
    values = np.array([0, 1, 11, 12, 35, 36, 1295])
    for width in (1, 2, 3):
        v = values[values < 36**width]
        assert M.unbase36(M.base36(v, width), width).tolist() == v.tolist()
    # One character per game and value for 4 snakes (shares in twelfths, twice the placement
    # summed over the seats at most 32), two for 5 snakes (shares in sixtieths).
    for seats, scale, width in (("G,S,G,S", 12, 1), ("G,S,R,R,R", 60, 2)):
        c = cell("standard", seats, games=24)
        r = direct(c, 24)
        pg = M.per_game_record(c, r)
        assert pg["share_scale"] == scale and pg["width"] == width and pg["encoding"] == "base36"
        assert all(
            len(text) == 24 * width for text in [*pg["share"].values(), *pg["place2"].values()]
        )
        held = {x: seats.split(",").count(x) for x in c.policies}
        back = M.decode_per_game({"per_game": pg, "results": {x: {"seats": held[x]} for x in held}})
        for i, x in enumerate(c.policies):
            np.testing.assert_allclose(back[x]["share"], r.game_share[:, i])
            np.testing.assert_allclose(back[x]["placement"], r.game_placement[:, i])
        total = sum(back[x]["share"] for x in c.policies)
        np.testing.assert_allclose(total, 1.0)  # the shares of a game sum to 1


# --- Running cells ---------------------------------------------------------------------


def test_single_cell_is_recorded_skipped_and_extended(tmp_path, capsys):
    out = tmp_path / "modes.jsonl"
    args = ["--game-mode", "duel", "--seats", "G,S", *CHEAP, "--seed", "3"]
    assert run(out, *args, "--games", "16") == 0
    (record,) = lines(out)
    assert record["schema"] == M.SCHEMA and record["game_mode"] == "duel"
    assert record["lineup"] == "G,S" and record["seats"] == ["G", "S"]
    assert record["seat_names"] == ["random_legal", "random"]
    assert record["agents"]["G"]["name"] == "random_legal" and record["agents"]["S"]["config"]
    assert record["game"]["num_snakes"] == 2 and record["game"]["ruleset"] == "standard"
    assert record["max_turns"] == 1000 and record["seed"] == 3 and record["early_stop"]
    assert record["games"] == record["games_requested"] == 16
    for field in ("git_commit", "git_dirty", "code_fingerprint", "heuristic_fingerprint", "date"):
        assert field in record, field
    assert len(record["heuristic_fingerprint"]) == 16

    # The result is play_table's, game for game, with the key of the mode and the seed.
    c = M.make_cell("duel", ["G", "S"], CHEAP_NAMES, 16)
    r = direct(c, 16, seed=3)
    for i, x in enumerate(c.policies):
        got = record["results"][x]
        assert got["share"] == r.share[i] and got["placement"] == r.placement[i]
        assert (got["wins"], got["ties"], got["losses"]) == (r.wins[i], r.ties[i], r.losses[i])
        assert got["share_wilson"] == list(r.share_wilson[i]) and got["deaths"] == r.deaths[i]
        assert got["seats"] == 1
    assert record["truncated"] == r.truncated and record["mean_turns"] == r.mean_turns
    decoded = M.decode_per_game(record)
    np.testing.assert_allclose(decoded["G"]["share"], r.game_share[:, 0])
    np.testing.assert_allclose(decoded["S"]["placement"], r.game_placement[:, 1])
    assert json.loads(json.dumps(record)) == record

    # Finished: not played again (the line count stays); the log says so.
    capsys.readouterr()
    assert run(out, *args, "--games", "16") == 0
    assert len(lines(out)) == 1 and "1 finished, 0 to play" in capsys.readouterr().err
    assert run(out, *args, "--games", "8") == 0 and len(lines(out)) == 1  # fewer: done too
    # --rerun plays it again; more games, another seed or another line-up are new work.
    assert run(out, *args, "--games", "16", "--rerun") == 0 and len(lines(out)) == 2
    assert run(out, *args, "--games", "24") == 0 and len(lines(out)) == 3
    assert lines(out)[-1]["games"] == 24
    assert run(out, "--game-mode", "duel", "--seats", "S,G", *CHEAP, "--games", "8") == 0
    assert len(lines(out)) == 4
    assert run(out, "--game-mode", "duel", "--seats", "G,S", *CHEAP, "--games", "8") == 0
    assert len(lines(out)) == 5  # the same cell with another seed (the default 0)
    # The first 16 games of the 24-game line are the 16-game line's (nothing depends on the count).
    first, big = lines(out)[0], lines(out)[2]
    assert M.decode_per_game(big)["G"]["share"][:16].tolist() == (
        M.decode_per_game(first)["G"]["share"].tolist()
    )


def test_table_of_a_four_snake_cell(tmp_path):
    out = tmp_path / "modes.jsonl"
    assert run(out, "--game-mode", "royale", "--seats", "G,S,G,S", *CHEAP, "--games", "12") == 0
    (record,) = lines(out)
    c = M.make_cell("royale", ["G", "S", "G", "S"], CHEAP_NAMES, 12)
    assert record["game"]["ruleset"] == "royale" and record["game"]["map"] == "royale"
    assert record["results"]["G"]["seats"] == 2 and record["lineup"] == "G,S,G,S"
    r = direct(c, 12)
    assert record["results"]["G"]["share"] == r.share[0]
    assert record["results"]["G"]["share"] + record["results"]["S"]["share"] == pytest.approx(1.0)
    assert record["results"]["G"]["placement"] + record["results"]["S"]["placement"] == (
        pytest.approx(5.0)  # placements of the seats average to (N + 1) / 2 over the table
    )


def test_table_printer(tmp_path, capsys):
    out = tmp_path / "modes.jsonl"
    assert M.main(["--table", "--out", str(out)]) == 1  # no file
    assert "no results" in capsys.readouterr().err
    for seats in ("G,R,R,R", "S,R,R,R", "G,S,R,R"):
        assert run(out, "--game-mode", "standard", "--seats", seats, *CHEAP, "--games", "12") == 0
    assert run(out, "--game-mode", "duel", "--seats", "G,S", *CHEAP, "--games", "12") == 0
    capsys.readouterr()
    assert M.main(["--table", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "## standard" in text and "## duel" in text and "## royale" not in text
    assert "| G,R,R,R | 12 |" in text and "| G,S (primary) | 12 |" in text
    assert "(co-headline)" not in text  # G,S,R,R is not one
    assert "Agents: G = random_legal; R = random_legal; S = random" in text
    c = M.make_cell("standard", ["G", "R", "R", "R"], CHEAP_NAMES, 12)
    r = direct(c, 12)
    lo, hi = r.share_wilson[0]
    assert f"G {r.share[0]:.3f} [{lo:.3f}, {hi:.3f}]" in text  # the Wilson interval
    assert f"G {r.placement[0]:.2f} +- {r.placement_ci95[0]:.2f}" in text
    assert f"G {r.wins[0]}/{r.ties[0]}/{r.losses[0]}" in text
    # Filters, and the newest of the lines of a cell (the one with the most games).
    assert M.main(["--table", "--out", str(out), "--game-mode", "duel"]) == 0
    only_duel = capsys.readouterr().out
    assert "## duel" in only_duel and "## standard" not in only_duel
    assert M.main(["--table", "--out", str(out), "--seed", "9"]) == 1
    assert run(out, "--game-mode", "duel", "--seats", "G,S", *CHEAP, "--games", "20") == 0
    capsys.readouterr()
    assert M.main(["--table", "--out", str(out), "--game-mode", "duel"]) == 0
    assert "| G,S (primary) | 20 |" in capsys.readouterr().out


# --- Paired differences --------------------------------------------------------------------


def test_paired_differences(tmp_path, capsys):
    out = tmp_path / "modes.jsonl"
    games = 32
    # The two suicidal random snakes (R) are soon out; G (random legal moves) or S (a small
    # network) then plays H (random legal moves): shares that vary from game to game and depend
    # on the game (same start, same food), so the games of two cells are correlated.
    run_dir = _tiny_modes_run(tmp_path / "net", seed=3)
    names = {"G": "random_legal", "S": f"ppo_modes:{run_dir}", "R": "random", "H": "random_legal"}
    agent_flags = [x for k, v in names.items() for x in ("--agent", f"{k}={v}")]
    for seats in ("G,R,R,H", "S,R,R,H", "G,S,R,R", "G,S,S,R", "G,H,R,R", "S,R,H,R"):
        assert run(out, "--game-mode", "standard", "--seats", seats, *agent_flags,
                   "--games", str(games)) == 0  # fmt: skip
    # The same line-up in another seed is other games: it never pairs with these.
    assert run(out, "--game-mode", "standard", "--seats", "S,R,R,H", *agent_flags, "--seed", "1",
               "--games", str(games)) == 0  # fmt: skip
    records = M.best_records(M.read_records(out)[0], seed=0, mode="standard")
    pairs = dict(M.pair_records(records))
    # G,R,R,H pairs with S,R,R,H (the same seats once G becomes S, the same others); G,H,R,R has no
    # partner, and neither has S,R,H,R (another seat order); G,S,R,R and G,S,S,R compare the two
    # inside their cell.
    assert set(pairs) == {"G,R,R,H vs S,R,R,H", "G,S,R,R (within)", "G,S,S,R (within)"}
    everything = M.best_records(M.read_records(out)[0], seed=None, mode="standard")
    assert len(everything) == len(records) + 1
    # With several seeds in view every comparison says which one (and seed 1 has no partner).
    assert set(dict(M.pair_records(everything))) == {f"{t} [seed 0]" for t in pairs}

    def table(seats):
        return direct(M.make_cell("standard", seats.split(","), names, games), games)

    z = 1.959963984540054
    a, b = table("G,R,R,H"), table("S,R,R,H")
    x, y = a.game_share[:, 0], b.game_share[:, 0]
    assert x.std() > 0.1 and y.std() > 0.05 and not np.allclose(x, y)  # shares vary by game
    p = pairs["G,R,R,H vs S,R,R,H"]
    assert p.games == games and p.mean == pytest.approx((x - y).mean())
    assert p.half == pytest.approx(z * (x - y).std(ddof=1) / np.sqrt(games))
    assert p.unpaired_half == pytest.approx(
        z * np.sqrt(x.var(ddof=1) + y.var(ddof=1)) / np.sqrt(games)
    )
    assert p.half != pytest.approx(p.unpaired_half, rel=1e-3)  # pairing changes the interval
    pa, pb = a.game_placement[:, 0], b.game_placement[:, 0]
    assert p.place_mean == pytest.approx((pa - pb).mean())
    assert p.place_half == pytest.approx(z * (pa - pb).std(ddof=1) / np.sqrt(games))

    # Within a cell, shares per seat: G holds one seat and S two in G,S,S,R.
    for seats, held in (("G,S,R,R", 1), ("G,S,S,R", 2)):
        c, within = table(seats), pairs[f"{seats} (within)"]
        d = c.game_share[:, 0] - c.game_share[:, 1] / held
        assert within.mean == pytest.approx(d.mean()) and within.games == games
        assert within.half == pytest.approx(z * d.std(ddof=1) / np.sqrt(games))
        assert within.unpaired_half is None

    assert M.main(["--table", "--out", str(out), "--seed", "0"]) == 0
    text = capsys.readouterr().out
    assert "G - S, paired game by game" in text and "G,R,R,H vs S,R,R,H" in text
    assert f"{p.mean:+.3f} +- {p.half:.3f}" in text and f"+- {p.unpaired_half:.3f}" in text


def test_further_specialists_are_compared(tmp_path, capsys):
    """A specialist of another seed or setting (S16) is compared with G and with S."""
    out = tmp_path / "modes.jsonl"
    agent_flags = [*CHEAP, "--agent", "S16=random", "--agent", "S_duel=random"]
    for seats in ("G,S16", "S16,S", "G,R", "S16,R", "S,R", "S_duel,R"):
        assert run(out, "--game-mode", "duel", "--seats", seats, *agent_flags, "--games", "8") == 0
    capsys.readouterr()
    assert M.main(["--table", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "G - S16, paired game by game" in text and "S16 - S, paired game by game" in text
    assert "| G,S16 (within) | 8 |" in text and "| G,R vs S16,R | 8 |" in text
    assert "| S16,S (within) | 8 |" in text and "| S16,R vs S,R | 8 |" in text
    assert "| G,R vs S,R | 8 |" in text  # the G-versus-S table as before
    assert "S_duel - S" not in text and "G - S_duel" not in text  # the matrix's symbols are not


def test_deterministic_constrictor_line_ups_are_marked(tmp_path, capsys):
    """Greedy networks alone in constrictor play games fixed by their start: few distinct games."""
    out = tmp_path / "modes.jsonl"
    a, b = _tiny_modes_run(tmp_path / "a", seed=5), _tiny_modes_run(tmp_path / "b", seed=6)
    names = {"A": f"ppo_modes:{a}", "B": f"ppo_modes:{b}", "As": f"ppo_modes-sample:{a}"}
    names |= {"Bs": f"ppo_modes-sample:{b}", "H": "heuristic"}
    agent_flags = [x for k, v in names.items() for x in ("--agent", f"{k}={v}")]
    games = 48
    for mode, seats in (("constrictor", "A,B,A,B"), ("constrictor", "As,Bs,As,Bs"),
                        ("constrictor", "A,H,H,H"), ("standard", "A,B,A,B")):  # fmt: skip
        assert run(out, "--game-mode", mode, "--seats", seats, *agent_flags,
                   "--games", str(games)) == 0  # fmt: skip
    recs = M.read_records(out)[0]
    flagged = {(r["game_mode"], r["lineup"]) for r in recs if M.deterministic(r)}
    assert flagged == {("constrictor", "A,B,A,B")}
    # It really is: the games repeat (2 start layouts x 6 seatings at most), the sampled ones don't.
    greedy = direct(M.make_cell("constrictor", ["A", "B", "A", "B"], names, games), games)
    sampled = direct(M.make_cell("constrictor", ["As", "Bs", "As", "Bs"], names, games), games)
    distinct = lambda r: len({tuple(x) for x in np.round(r.game_placement, 6)})  # noqa: E731
    assert distinct(greedy) <= 12 < games
    assert distinct(sampled) > distinct(greedy)
    capsys.readouterr()
    assert M.main(["--table", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "| A,B,A,B (deterministic) |" in text and "Deterministic: only greedy ppo_modes" in text
    assert "| As,Bs,As,Bs |" in text and "| A,H,H,H |" in text
    # Paired rows built from a deterministic line-up say so too.
    titles = {t for t, _ in M.pair_records(recs, a="A", b="B")}
    assert "A,B,A,B (within, deterministic)" in titles and "A,B,A,B (within)" in titles
    assert {t for t, _ in M.pair_records(recs, a="As", b="Bs")} == {"As,Bs,As,Bs (within)"}


def test_pairing_needs_the_same_other_agents_and_the_same_games(tmp_path):
    out = tmp_path / "modes.jsonl"
    g, spec_s = ["--agent", "G=random_legal"], ["--agent", "S=random"]
    common = ["--game-mode", "standard", "--games", "8", "--seats"]
    assert run(out, *common, "G,R,R,R", *g, "--agent", "R=random_legal") == 0
    # Each of these differs from the first in one way and has no partner.
    assert run(out, *common, "S,R,R,R", *spec_s, "--agent", "R=random") == 0  # another R
    assert run(out, "--game-mode", "royale", "--games", "8", "--seats", "S,R,R,R", *spec_s,
               "--agent", "R=random_legal") == 0  # another mode  # fmt: skip
    assert (
        run(out, *common, "S,R,R,R", *spec_s, "--agent", "R=random_legal", "--max-turns", "50") == 0
    )
    assert run(out, *common, "S,R,R,R", *spec_s, "--agent", "R=random_legal", "--seed", "1") == 0
    assert run(out, *common, "R,S,R,R", *spec_s, "--agent", "R=random_legal") == 0  # another seat
    records = M.read_records(out)[0]
    assert len(records) == 6 and M.pair_records(records) == []
    # With the same everything but the policy it pairs.
    assert run(out, *common, "S,R,R,R", *spec_s, "--agent", "R=random_legal") == 0
    pairs = M.pair_records(M.read_records(out)[0])
    # (The file has seed 1 lines too, so the comparison names its seed.)
    assert [t for t, _ in pairs] == ["G,R,R,R vs S,R,R,R [seed 0]"]


def test_pairing_needs_the_same_code(tmp_path, capsys):
    """Lines from before and after a change of the heuristic (or of any code) are not paired."""
    out = tmp_path / "modes.jsonl"
    # As in test_paired_differences: shares that vary from game to game in both cells.
    net = _tiny_modes_run(tmp_path / "net", seed=3)
    names = {"G": "random_legal", "S": f"ppo_modes:{net}", "R": "random", "H": "random_legal"}
    agents = [x for k, v in names.items() for x in ("--agent", f"{k}={v}")]
    for seats in ("G,R,R,H", "S,R,R,H"):
        args = ["--game-mode", "standard", "--seats", seats, *agents, "--games", "32"]
        assert run(out, *args) == 0
    first, second = M.read_records(out)[0]
    assert first["code_fingerprint"] == second["code_fingerprint"]
    ((_, same),) = M.pair_records([first, second])
    assert same.note == "" and same.unpaired_half is not None
    da, db = M.decode_per_game(first), M.decode_per_game(second)
    x, y = da["G"]["share"], db["S"]["share"]
    assert x.std() > 0.1 and y.std() > 0.05
    assert not np.isclose(same.half, same.unpaired_half, rtol=1e-3)  # pairing changes the interval
    z = 1.959963984540054
    independent = z * np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / len(x))

    old = "0" * 16  # a fingerprint of other code
    for field, note in (
        ("heuristic_fingerprint", "different heuristic version"),
        ("code_fingerprint", "different code version"),
    ):
        for changed in ([first, {**second, field: old}], [{**first, field: old}, second]):
            ((title, p),) = M.pair_records(changed)
            assert title == "G,R,R,H vs S,R,R,H" and p.note == note
            assert p.mean == pytest.approx(same.mean) and p.games == same.games
            # Only the unpaired intervals: the lines are not taken as the same games.
            assert p.half == pytest.approx(independent) and p.unpaired_half == p.half
            pa, pb = da["G"]["placement"], db["S"]["placement"]
            assert p.place_half == pytest.approx(
                z * np.sqrt((pa.var(ddof=1) + pb.var(ddof=1)) / len(pa))
            )
    # A changed heuristic changes the whole code fingerprint too: the heuristic is named.
    both = {**second, "heuristic_fingerprint": old, "code_fingerprint": old}
    assert M.pair_records([first, both])[0][1].note == "different heuristic version"
    # The same code on both sides pairs, whatever the fingerprint is.
    same_old = [
        {**r, "heuristic_fingerprint": old, "code_fingerprint": old} for r in (first, second)
    ]
    assert M.pair_records(same_old)[0][1].note == ""

    # The table says so, and shows the unpaired comparison.
    with open(out, "w") as f:
        for r in (first, {**second, "heuristic_fingerprint": old}):
            f.write(json.dumps(r) + "\n")
    assert M.main(["--table", "--out", str(out)]) == 0
    text, err = capsys.readouterr()
    line = next(x for x in text.splitlines() if x.startswith("| G,R,R,H vs S,R,R,H |"))
    assert "(unpaired)" in line and "not paired: different heuristic version" in line
    assert f"{same.mean:+.3f} +- {independent:.3f} (unpaired)" in line
    assert "different versions of the heuristic" in err


def test_several_seeds_are_labelled(tmp_path, capsys):
    out = tmp_path / "modes.jsonl"
    for seed in ("0", "1"):
        for seats in ("G,R", "S,R"):
            args = ["--game-mode", "duel", "--seats", seats, *CHEAP, "--games", "12"]
            assert run(out, *args, "--seed", seed) == 0
    assert M.main(["--table", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    for seed in (0, 1):
        assert f"| G,R [seed {seed}] | 12 |" in text and f"| S,R [seed {seed}] | 12 |" in text
        assert f"| G,R vs S,R [seed {seed}] | 12 |" in text
    assert text.count("| G,R vs S,R [seed") == 2  # one comparison per seed, told apart
    # One seed: no tags.
    assert M.main(["--table", "--out", str(out), "--seed", "1"]) == 0
    text = capsys.readouterr().out
    assert "[seed" not in text and text.count("| G,R vs S,R |") == 1
    # The rows of a mode with one seed are untagged even if the file has several.
    assert run(out, "--game-mode", "royale", "--seats", "G,R,R,R", *CHEAP, "--games", "8") == 0
    assert M.main(["--table", "--out", str(out), "--game-mode", "royale"]) == 0
    assert "[seed" not in capsys.readouterr().out


def test_wrong_mode_ablation_is_read_out(tmp_path, capsys):
    """G against G told another mode, same games: the difference says whether the mode is used."""
    out = tmp_path / "modes.jsonl"
    net = _tiny_modes_run(tmp_path / "net", seed=4)
    games = 32
    names = {"G": f"ppo_modes:{net}", "G@standard": f"ppo_modes:{net}:game_mode=standard"}
    names["R"] = "random_legal"
    agents = [x for k, v in names.items() for x in ("--agent", f"{k}={v}")]
    for seats in ("G,R", "G@standard,R"):
        assert (
            run(out, "--game-mode", "duel", "--seats", seats, *agents, "--games", str(games)) == 0
        )
    recs = M.best_records(M.read_records(out)[0], seed=0, mode="duel")
    ((title, p),) = M.pair_records(recs, a="G", b="G@standard")
    assert title == "G,R vs G@standard,R" and p.games == games and p.note == ""
    right = direct(M.make_cell("duel", ["G", "R"], names, games), games)
    wrong = direct(M.make_cell("duel", ["G@standard", "R"], names, games), games)
    x, y = right.game_share[:, 0], wrong.game_share[:, 0]
    assert not np.allclose(x, y)  # the told mode changes play: the network uses it
    assert p.mean == pytest.approx((x - y).mean())
    assert p.half == pytest.approx(1.959963984540054 * (x - y).std(ddof=1) / np.sqrt(games))
    assert M.pair_records(recs) == []  # the G-versus-S table has nothing here

    assert M.main(["--table", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Wrong-mode ablation, G - G@m" in text and "G - S, paired" not in text
    assert (
        "| Comparison | Games | Share G - G@m | Unpaired half-width | Placement G - G@m |" in text
    )
    line = next(x for x in text.splitlines() if x.startswith("| G,R vs G@standard,R |"))
    assert f"{p.mean:+.3f} +- {p.half:.3f}" in line and f"+- {p.unpaired_half:.3f}" in line


def test_rough_cost_model_counts_ppo_modes(tmp_path):
    run_dir = _tiny_modes_run(tmp_path / "net")
    ppo = M.strength.ROUGH_MS["ppo"]
    heuristic = M.strength.load_agent("heuristic")
    cost = {}
    for name in ("heuristic", f"ppo_modes:{run_dir}:game_mode=duel"):
        matchup = M.strength.Matchup(M.strength.load_agent(name), heuristic, 100)
        cost[name] = matchup.rough_seconds()
    base = cost["heuristic"]  # no network: the fixed cost per move only
    turns = M.strength.ROUGH_TURNS_PER_GAME
    assert cost[f"ppo_modes:{run_dir}:game_mode=duel"] == pytest.approx(
        base + 100 * turns * ppo / 1e3
    )


# --- The protocol ------------------------------------------------------------------------


def test_protocol_cells(tmp_path):
    general = _tiny_modes_run(tmp_path / "g")
    duel = _tiny_modes_run(tmp_path / "s-duel", ("duel",))
    royale = _tiny_modes_run(tmp_path / "s-royale", ("royale",))
    specialists = {"duel": str(duel), "royale": str(royale)}
    cells = M.protocol_cells(str(general), specialists)
    by_mode = {m: [(c.label, c.games) for c in cells if c.game_mode == m] for m in modes.MODE_NAMES}
    wrong = lambda m, tail: [(f"G@{o}{tail}", 256) for o in modes.MODE_NAMES if o != m]  # noqa: E731

    # The duel: G,S first (the primary endpoint) and its sampled twin, then each of G and S
    # against H and P, floors.
    assert by_mode["duel"] == [
        ("G,S", 1024), ("Gs,Ss", 1024),
        ("G,H", 1024), ("S,H", 1024), ("G,P", 1024), ("S,P", 1024),
        ("G,R", 256), ("S,R", 256), *wrong("duel", ",H"),
    ]  # fmt: skip
    # A 4-snake mode: the primary endpoint, its sampled twin and the co-headline first, then the
    # mid-strength and zero-shot references, the floors and the generalist told wrong modes.
    four = [
        ("G,S,G,S", 1024), ("Gs,Ss,Gs,Ss", 1024), ("G,S,H,H", 1024),
        ("G,H,H,H", 1024), ("G,P,P,P", 1024), ("G,H,R,R", 1024),
        ("S,H,H,H", 1024), ("S,P,P,P", 1024), ("S,H,R,R", 1024),
        ("P,H,H,H", 512), ("G,R,R,R", 256), ("S,R,R,R", 256), *wrong("royale", ",H,H,H"),
    ]  # fmt: skip
    assert by_mode["royale"] == four
    # No specialist, no S cells (G and P keep theirs).
    assert [label for label, _ in by_mode["standard"]] == [
        "G,H,H,H",
        "G,P,P,P",
        "G,H,R,R",
        "P,H,H,H",
        "G,R,R,R",
        *(x for x, _ in wrong("standard", ",H,H,H")),
    ]
    assert not [c for c in cells if "S" in c.seats and c.game_mode in ("standard", "constrictor")]

    # Agents: G is the generalist told its game; G@m is told m; P is the duel checkpoint.
    c = next(c for c in cells if c.game_mode == "royale" and c.label == "G,S,H,H")
    assert (
        c.agents["G"].name == f"ppo_modes:{general}" and c.agents["S"].name == f"ppo_modes:{royale}"
    )
    assert c.agents["H"].name == "heuristic"
    sampled = next(c for c in cells if c.game_mode == "royale" and c.label == "Gs,Ss,Gs,Ss")
    assert sampled.agents["Gs"].name == f"ppo_modes-sample:{general}"
    assert sampled.agents["Ss"].name == f"ppo_modes-sample:{royale}"
    assert not sampled.agents["Gs"].spec.greedy and not sampled.agents["Ss"].spec.greedy
    p = next(c for c in cells if c.label == "P,H,H,H" and c.game_mode == "royale")
    assert p.agents["P"].name == "ppo"
    told = next(c for c in cells if c.label == "G@duel,H,H,H" and c.game_mode == "royale")
    assert told.agents["G@duel"].name == f"ppo_modes:{general}:game_mode=duel"
    assert told.agents["G@duel"].spec.game_mode == "duel"

    # --only, --games and --matrix.
    only = M.protocol_cells(str(general), specialists, ("duel",), games=4)
    assert {c.game_mode for c in only} == {"duel"} and {c.games for c in only} == {4}
    matrix = M.protocol_cells(str(general), specialists, ("royale",), matrix=True)
    extra = [c for c in matrix if "S_duel" in c.seats]
    assert [c.label for c in extra] == ["S_duel,H,H,H"] and extra[0].games == 256
    assert extra[0].agents["S_duel"].name == f"ppo_modes:{duel}:game_mode=duel"
    assert len(matrix) == len(by_mode["royale"]) + 1
    # Each other specialist has a symbol, and so a row, of its own (not one shared symbol
    # that prints identical rows): in a 4-snake mode and in the duel.
    constrictor = _tiny_modes_run(tmp_path / "s-constrictor", ("constrictor",))
    three = {**specialists, "constrictor": str(constrictor)}
    for mode, tail in (("standard", ",H,H,H"), ("duel", ",H")):
        cells = M.protocol_cells(str(general), three, (mode,), matrix=True)
        labels = [c.label for c in cells]
        assert len(set(labels)) == len(labels), labels
        told = {c.label: c.agents[c.seats[0]].name for c in cells if c.seats[0].startswith("S_")}
        want = {m: f"ppo_modes:{three[m]}:game_mode={m}" for m in three if m != mode}
        assert told == {f"S_{m}{tail}": name for m, name in want.items()}
    # A generalist that does not know every mode: its wrong-mode cells for the others are skipped,
    # and a mode it was not trained on is an error.
    narrow = _tiny_modes_run(tmp_path / "narrow", ("duel", "constrictor"))
    cells = M.protocol_cells(str(narrow), {}, ("duel",))
    assert [c.label for c in cells if c.label.startswith("G@")] == ["G@constrictor,H"]
    with pytest.raises(M.SpecError, match="can't play a royale game"):
        M.protocol_cells(str(narrow), {}, ("royale",))
    with pytest.raises(M.SpecError, match="unknown game mode"):
        M.protocol_cells(str(general), {"arena": str(duel)})
    with pytest.raises(M.SpecError, match="unknown game mode"):
        M.protocol_cells(str(general), {}, ("arena",))


def test_protocol_command_line(tmp_path, capsys):
    general = _tiny_modes_run(tmp_path / "g")
    duel = _tiny_modes_run(tmp_path / "s-duel", ("duel",))
    out = tmp_path / "modes.jsonl"
    flags = ["--protocol", "--generalist", str(general), "--specialist", f"duel={duel}"]
    assert run(out, *flags, "--only", "duel", "--dry-run") == 0
    err = capsys.readouterr().err
    assert "11 cell(s), 0 finished, 11 to play" in err and "G,S" in err and not out.exists()
    # Two cells of the protocol, played; a repeat finds them finished.
    cells = M.protocol_cells(str(general), {"duel": str(duel)}, ("duel",), games=6)
    assert M.sweep([cells[0], cells[6]], SETTINGS, out, rerun=False) == 0
    assert [r["lineup"] for r in lines(out)] == ["G,S", "G,R"]
    assert run(out, *flags, "--only", "duel", "--games", "6", "--dry-run") == 0
    err = capsys.readouterr().err
    assert "11 cell(s), 2 finished, 9 to play" in err and err.count("  done  ") == 2
    # Bad uses.
    for bad, message in (
        (["--protocol"], "--protocol needs --generalist"),
        ([*flags, "--seats", "G,S"], "not a cell"),
        (["--protocol", "--generalist", str(general), "--specialist", "duel"], "NAME=VALUE"),
        (["--protocol", "--generalist", str(general), "--specialist", f"duel={duel}",
          "--specialist", f"duel={duel}"], "given twice"),
        (["--game-mode", "duel", "--seats", "G,S", "--generalist", str(general)], "for --protocol"),
        (["--game-mode", "duel"], "name a cell"),
    ):  # fmt: skip
        assert run(out, *bad) == 2
        assert message in capsys.readouterr().err


def test_bad_cells_are_refused(tmp_path, capsys):
    out = tmp_path / "modes.jsonl"
    duel = _tiny_modes_run(tmp_path / "s-duel", ("duel",))
    cases = (
        (["--game-mode", "duel", "--seats", "G,S,R", *CHEAP], "cannot have 3 snakes"),
        (["--game-mode", "standard", "--seats", "G,S", *CHEAP], "cannot have 2 snakes"),
        (["--game-mode", "duel", "--seats", "G,X", *CHEAP], "no agent for seat symbol 'X'"),
        (["--game-mode", "duel", "--seats", "G,1x", *CHEAP], "bad seat symbol '1x'"),
        (["--game-mode", "duel", "--seats", "G,S", "--agent", "G=bogus"], "unknown agent"),
        (["--game-mode", "duel", "--seats", "G,S", "--agent", "G"], "NAME=VALUE"),
        (["--game-mode", "duel", "--seats", "G,S", *CHEAP, "--agent", "G=random"], "given twice"),
        (["--game-mode", "duel", "--seats", "G,S", *CHEAP, "--games", "0"], "must be >= 1"),
        # A specialist can't play the game of another mode (told its own: it can).
        (["--game-mode", "standard", "--seats", "G,R,R,R", "--agent", f"G=ppo_modes:{duel}"],
         "trained on duel only"),
        (["--game-mode", "duel", "--seats", "G,R", "--agent", "G=ppo_modes:/nonexistent/run"],
         "no multi-mode PPO checkpoint"),
    )  # fmt: skip
    for args, message in cases:
        assert run(out, *args) == 2, args
        assert message in capsys.readouterr().err, args
    assert not out.exists()
    told = ["--agent", f"G=ppo_modes:{duel}:game_mode=duel"]
    assert run(out, "--game-mode", "standard", "--seats", "G,R,R,R", *told, "--games", "4") == 0
    assert lines(out)[0]["agents"]["G"]["config"]["game_mode"] == "duel"


def test_results_file_reading(tmp_path):
    out = tmp_path / "modes.jsonl"
    assert M.read_records(out) == ([], 0) and M.finished_games(out) == {}
    assert run(out, "--game-mode", "duel", "--seats", "G,S", *CHEAP, "--games", "4") == 0
    with open(out, "a") as f:  # a torn line, a foreign line, a blank, and a line cut short
        f.write('{"schema": 1, "cell_id": "x"\n{"schema": 99}\n\n[1]')
    records, bad = M.read_records(out)
    assert len(records) == 1 and bad == 3
    assert M.finished_games(out) == {records[0]["cell_id"]: 4}
    # The next line is appended after the damage, and reading still finds both.
    assert run(out, "--game-mode", "duel", "--seats", "S,G", *CHEAP, "--games", "4") == 0
    assert len(M.read_records(out)[0]) == 2
