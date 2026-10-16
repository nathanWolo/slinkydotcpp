"""Checks of the Rust match runner (native/crates/referee, ``naz-match``) against slinky's Python.

Run (CPU, memory capped, through the gate):
    bash <scratchpad>/memgate.sh /home/nathan/venvs/slinky-rocm10/bin/python \\
        /home/nathan/slinky-native/native/golden/referee_golden.py [--only seq,scoring] \\
        [--out DIR] [--results FILE.jsonl ...] [--cases N]

``seq``: golden vectors for the ported statistics. The functions come from the eval-speed
branch's ``benchmarks/strength.py`` (``git show eval-speed:benchmarks/strength.py`` of
``--evalspeed``, imported with that worktree's ``src`` first on the path): ``seq_test`` on
random per-round W/D/L sequences (duel margins from Elo and 4-snake-like ``p0``/margins,
several alpha/beta, draw rates 0 to 40%, true scores 0.3 to 0.7), ``score_cs``,
``wilson_eff_interval``, ``score_interval``, ``outcome_stats``, ``kelly_bet`` and
``score_to_elo``. Written to ``<out>/referee/seq.json``:

    {"seq": [{"rounds": [[w, d, l], ...], "p0", "margin", "alpha", "beta",
              "log_e": {"e_a", "f_a", "e_b", "f_b"}, "ended": {"a", "b"}, "decision",
              "stop_round", "games"}, ...],
     "intervals": [{"wdl": [w, d, l], "cs": [lo, hi], "eff": [lo, hi],
                    "wilson": [score, lo, hi], "wald": [n, score, ci]}, ...],
     "kelly": [[m_null, m_alt, draw, lam], ...], "elo": [[score, elo], ...]}

The Rust test ``crates/referee/tests/golden.rs`` reads it from ``$NAZ_GOLDEN`` (default
``/dev/shm/naz-golden``) and skips when it is missing.

``scoring``: replays the games of ``naz-match`` result files (``--results``, each with its
``.games.jsonl``) through slinky's own scoring: ``slinky.evaluate.run_match`` (W/D/L,
truncations, total turns) and ``play_table`` with line-up ``(0, 1)`` or ``(0, 1, 1, 1)``
(``per_game=True``: each game's share and placement for A). The env is a stand-in whose states
follow each recorded game's eliminations (alive, elimination turn and cause per seat, the turn,
``done`` at the recorded last turn), so slinky's code scores exactly the games the referee
played, with A in seat ``g % N`` by slinky's own rotation. Prints the comparison and exits 1 on
any mismatch.
"""
import argparse, collections, importlib.util, json, math, os, random, subprocess, sys, tempfile
from pathlib import Path
from typing import Any, NamedTuple


def load_strength(evalspeed: Path):
    """The eval-speed strength.py as a module (its src first on the path)."""
    sys.path.insert(0, str(evalspeed / "src"))
    sys.path.insert(1, str(evalspeed))
    text = subprocess.run(
        ["git", "-C", str(evalspeed), "show", "eval-speed:benchmarks/strength.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / "strength_evalspeed.py"
    tmp.write_text(text)
    spec = importlib.util.spec_from_file_location("strength_evalspeed", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["strength_evalspeed"] = mod  # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


def seq_part(out: Path, evalspeed: Path, cases: int, seed: int) -> None:
    st = load_strength(evalspeed)
    rng = random.Random(seed)
    seq = []
    for i in range(cases):
        duel = i % 3 != 2
        if duel:
            p0 = 0.5
            margin = st.score_margin(rng.choice([20.0, 35.0, 50.0, 100.0]))
        else:
            p0 = rng.uniform(0.2, 0.4)
            margin = rng.uniform(0.02, 0.1)
        alpha, beta = rng.choice([(0.05, 0.05), (0.01, 0.1), (0.1, 0.02)])
        true = min(max(p0 + rng.uniform(-0.2, 0.2), 0.02), 0.98)
        draw = rng.uniform(0.0, min(0.4, 2 * min(true, 1 - true) * 0.9))
        win = true - draw / 2
        size = rng.choice([8, 32, 64, 64, 128])
        rounds = []
        for _ in range(rng.randint(1, 40)):
            w = d = lo = 0
            for _ in range(size):
                u = rng.random()
                if u < win:
                    w += 1
                elif u < win + draw:
                    d += 1
                else:
                    lo += 1
            rounds.append((w, d, lo))
        s = st.seq_test(rounds, p0, margin, alpha, beta)
        seq.append({
            "rounds": [list(r) for r in rounds], "p0": p0, "margin": margin,
            "alpha": alpha, "beta": beta, "log_e": s.log_e, "ended": s.ended,
            "decision": s.decision, "stop_round": s.stop_round, "games": s.games,
        })
    intervals = []
    for _ in range(cases):
        n = rng.choice([1, 2, 7, 64, 300, 1024, 4096])
        w = rng.randint(0, n)
        d = rng.randint(0, n - w)
        lo = n - w - d
        intervals.append({
            "wdl": [w, d, lo], "cs": list(st.score_cs(w, d, lo)),
            "eff": list(st.wilson_eff_interval(w, d, lo)),
            "wilson": list(st.score_interval(w, d, lo)),
            "wald": list(st.outcome_stats(w, d, lo)),
        })
    kelly = []
    for _ in range(cases):
        m_null = rng.uniform(0.05, 0.95)
        m_alt = min(max(m_null + rng.uniform(-0.1, 0.1), 0.01), 0.99)
        draw = rng.uniform(0, 0.6)
        kelly.append([m_null, m_alt, draw, st.kelly_bet(m_null, m_alt, draw)])
    elo = [[s, st.score_to_elo(s)] for s in [0.0, 1e-12, 0.1, 0.3, 0.5, 0.527, 0.9, 1.0]]
    path = out / "referee"
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "seq.json", "w") as f:
        json.dump({"seq": seq, "intervals": intervals, "kelly": kelly, "elo": elo}, f)
    decisions = collections.Counter(c["decision"] for c in seq)
    print(f"seq: {len(seq)} sequential cases (decisions {dict(decisions)}), "
          f"{len(intervals)} interval cases, {len(kelly)} bets -> {path / 'seq.json'}")


class FState(NamedTuple):
    game: Any
    turn: Any
    alive: Any
    elim_turn: Any
    elim_cause: Any
    done: Any


def scoring_part(results: list[Path]) -> int:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from slinky.evaluate import run_match, run_table

    class Config(NamedTuple):
        solo: bool

    class ReplayEnv:
        """Stands in for BattlesnakeEnv: game g's states follow the recorded eliminations."""

        def __init__(self, n, elim, cause, last, key_table):
            self.num_agents = n
            self.config = Config(False)
            self.elim = jnp.asarray(elim, jnp.int32)  # [G, N]; 0 = not eliminated
            self.cause = jnp.asarray(cause, jnp.int8)
            self.last = jnp.asarray(last, jnp.int32)  # [G]
            self.keys = jnp.asarray(key_table, jnp.uint32)  # [G, 2]

        def reset(self, key):
            data = jax.random.key_data(key)
            g = jnp.argmax(jnp.all(self.keys == data, axis=-1)).astype(jnp.int32)
            n = self.num_agents
            return FState(g, jnp.int32(0), jnp.ones(n, bool), jnp.zeros(n, jnp.int32),
                          jnp.zeros(n, jnp.int8), jnp.bool_(False)), None

        def step(self, key, s, actions):
            t = s.turn + 1
            e = self.elim[s.game]
            dead = (e > 0) & (e <= t)
            return FState(s.game, t, ~dead, jnp.where(dead, e, 0),
                          jnp.where(dead, self.cause[s.game], 0).astype(jnp.int8),
                          t >= self.last[s.game]), None

    bad = 0
    for res in results:
        games_path = res.with_name(res.name + ".games.jsonl")
        records = [json.loads(x) for x in open(res) if x.strip()]
        lines = []
        for x in open(games_path):
            try:
                lines.append(json.loads(x))
            except ValueError:
                pass
        for rec in records:
            if rec.get("tool") != "naz-match":
                continue
            n = rec["game"]["num_snakes"]
            g_count = rec["games"]
            by_id = {}
            for line in lines:
                if line["matchup_id"] == rec["matchup_id"] and line["code_fingerprint"] == rec["code_fingerprint"]:
                    by_id[line["game"]] = line
            games = [by_id[g] for g in range(g_count)]
            elim = [[x["elim_turn"][s] if not x["alive"][s] else 0 for s in range(n)] for x in games]
            cause = [x["elim_cause"] for x in games]
            last = [x["turns"] for x in games]
            key = jax.random.key(0)
            table = np.stack([
                np.asarray(jax.random.key_data(jax.random.split(jax.random.fold_in(key, g))[0]))
                for g in range(g_count)
            ])
            assert len({tuple(r) for r in table.tolist()}) == g_count
            env = ReplayEnv(n, elim, cause, last, table)
            zero = lambda k, s, ts: jnp.zeros(n, jnp.int32)  # noqa: E731
            zero_b = lambda k, s, ts: jnp.zeros(n, jnp.int32)  # noqa: E731
            m = run_match(env, zero, zero_b, key, g_count, rec["max_turns"], min(g_count, 256)).result
            lineup = (0,) + (1,) * (n - 1)
            t = run_table(env, (zero, zero_b), lineup, key, g_count, rec["max_turns"],
                          min(g_count, 256), early_stop=True, per_game=True).result
            rust = (rec["wins"], rec["draws"], rec["losses"], rec["truncated"])
            py = (m.wins, m.draws, m.losses, m.truncated)
            turns_ok = math.isclose(m.mean_turns, rec["mean_turns"], rel_tol=1e-12)
            share_err = max(abs(float(t.game_share[g, 0]) - games[g]["share"]) for g in range(g_count))
            place_err = max(abs(float(t.game_placement[g, 0]) - games[g]["placement"]) for g in range(g_count))
            # The Rust share/placement are exact rationals in f64; slinky's per-game values are f32.
            ok = rust == py and turns_ok and share_err < 1e-6 and place_err < 1e-6
            ok = ok and abs(t.share[0] - rec["share"]) < 1e-9 and abs(t.placement[0] - rec["placement"]) < 1e-9
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {rec['mode']:12s} {g_count:5d} games: W/D/L/trunc rust {rust} "
                  f"slinky {py}; mean turns {rec['mean_turns']:.3f} vs {m.mean_turns:.3f}; share "
                  f"{rec['share']:.6f} vs {t.share[0]:.6f}, placement {rec['placement']:.6f} vs "
                  f"{t.placement[0]:.6f}; per game max |dshare| {share_err:.1e} |dplace| {place_err:.1e}")
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="seq,scoring")
    p.add_argument("--out", type=Path, default=Path(os.environ.get("NAZ_GOLDEN", "/dev/shm/naz-golden")))
    p.add_argument("--evalspeed", type=Path, default=Path("/home/nathan/slinky-evalspeed"))
    p.add_argument("--results", type=Path, nargs="*", default=[])
    p.add_argument("--cases", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    parts = set(a.only.split(","))
    bad = 0
    if "seq" in parts:
        seq_part(a.out, a.evalspeed, a.cases, a.seed)
    if "scoring" in parts and a.results:
        bad = scoring_part(a.results)
        print(f"scoring: {bad} matchups disagree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
