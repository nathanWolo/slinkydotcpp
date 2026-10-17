# slinky native engine

A Rust port of slinky's multi-mode AlphaZero agent (`az_modes`) that runs without JAX, Python
or a BLAS library. Give it a Battlesnake `/move` request and it returns the move the Python
server would play for `az_modes:<run dir>:simulations=256`. The target is AWS Lambda (arm64),
where the JAX stack is too heavy.

This is a faithful port. Given the same position, the same weights and the same random key, it
runs the same 256-simulation decoupled-PUCT search as `slinky.mcts.search` and plays the same
move. A time-budget search (`Budget::until`) exists but is not the default and has no parity
claim.

The porting spec, with `file:line` references into the Python, is
`/home/nathan/rocdxg-work/native-az/SPEC.md`. Each crate's module docs name the Python
functions they port.

## Crates

| crate | package | what it does | Python it ports |
|---|---|---|---|
| `crates/rules` | `slinky-rules` | Game state (up to 4 seats, boards up to 11x11 = 121 cells as `u128` bitboards). The rules step for standard, royale and constrictor without food spawns. Action masks, legal moves, certain starvation, royale zones and placement values | `rules.py`, `types.py`, parts of `mcts.py` |
| `crates/obs` | `slinky-obs` | Egocentric 21x21x13 observations, the 7 mode features and constrictor food hiding | `observations.py`, `modes.py`, `unified.policy_inputs` |
| `crates/net` | `slinky-net` | Loads `params.npz` and `config.json` (its own npy/zip reader), then runs the forward pass: config-driven convs, the hidden layer, features, policy and value heads. Sparse and padding-aware, with no `unsafe` | `az_modes.network`, `unified.torso`, `unified.load_params_like` |
| `crates/api` | `slinky-api` | Battlesnake API types. Converts a request into a config, a mode and a state (our snake in seat 0), and computes the safe-move fallback. Requests are read with Python's semantics (`py.rs`) | `server.py` (`game_config`, `request_state`, `royale_sides`, `safe_move`), `engine_json.py` |
| `crates/search` | `slinky-search` | The PUCT search, JAX threefry noise, the `Agent` API (request in, move out) and the `naz` CLI | `mcts.search`, `az_modes.agent_search` / `make_evaluator` / `make_policy`, `agents.check_game` |
| `crates/lambda` | `slinky-lambda` | An AWS Lambda handler skeleton (routing, the info response and the `/start` snake-count cache). It is **not yet wired to the engine** and answers "up"; see its README | `deploy/lambda` |
| `crates/referee` | `slinky-referee` | `naz-match`: strength tests between native agents in the four ladder modes (multi-threaded, resumable, sequential stopping) | `benchmarks/strength.py`, `evaluate.run_match` / `play_table` scoring, eval-speed `--sequential` |

Dependency order: `rules`, then `obs` and `api`, then `search`. `net` has no engine dependencies,
and `search` uses all four.

## Build and test

```bash
cd native
source .cargo-env.sh                          # rustup toolchain on PATH
export CARGO_TARGET_DIR=/dev/shm/naz-target   # build output in RAM: the SSD is worn
cargo build --release
cargo test --workspace --release              # golden tests skip without vectors (below)
cargo clippy --workspace --all-targets        # kept at zero warnings
cargo fmt --all --check
```

Play one move from the command line:

```bash
$CARGO_TARGET_DIR/release/naz --run /home/nathan/slinky-serve/az-modes-M-s0 -v < move.json
# options: --sims N (default 256)  --seed S (tie noise of jax.random.key(S))
#          --starting N (snakes at /start)  --time-ms T (stop early; not the faithful mode)
```

From Rust:

```rust
let agent = slinky_search::Agent::load(run_dir)?;            // config.json + params.npz
let (response, result) = agent.respond(&body, starting, 0);  // never fails: safe move on error
```

`Agent` is immutable and can be shared across threads. Each thread needs its own `Search`
workspace.

## How parity is checked

The Python scripts in `golden/` run slinky's jitted JAX code and write golden vectors to
`/dev/shm/naz-golden/<crate>/` (RAM; never committed). Each script's docstring documents its
file format. The Rust tests read them from `$NAZ_GOLDEN` (default `/dev/shm/naz-golden`). When
the files are missing, a test prints a SKIP line and passes. Run the scripts CPU-only with a
memory cap (the gate script sets `JAX_PLATFORMS=cpu` and the slinky `PYTHONPATH`):

```bash
G=native/golden
for s in rules_golden obs_golden net_golden api_golden review_rules \
         review_network_and_observations review_api_conversion; do
  python $G/$s.py; done
for p in noise mech agent e2e; do python $G/search_golden.py --only $p; done
for p in crafted random ladder variants; do python $G/review_search.py --only $p; done
```

Regenerating everything takes about 12 minutes on 4 CPU cores and uses about 770 MB of
`/dev/shm`. Then `cargo test --workspace --release` takes about 1 minute.

Each layer is held to the strictest standard that is achievable:

* **Rules, observations, features, request conversion, search mechanics: bit-exact.** These
  are integer code or f32 code in a fixed order, and the tests compare bit patterns.
* **Network: within float tolerance.** XLA's convolution summation order and its `tanh`/`exp`
  approximations are not reproducible, so logits and values are compared to about 1e-5.
* **Whole agent: same move.** The search is checked bit-exact with Python's network outputs
  replayed into it. With the Rust network it is checked for move agreement, because a 1e-6
  network difference can flip a near-tie in the visit counts.

## Measured parity (final verification, 2026-10-15)

All golden vectors were regenerated from an empty `/dev/shm/naz-golden` (15 generator runs, all
succeeded). `cargo test --workspace --release` passed 140 tests, with 0 failed, 0 ignored and no
skips. With `NAZ_GOLDEN` pointing at an empty directory, all 140 still pass and the golden parts
print SKIP. Clippy (`--all-targets -D warnings`), `cargo fmt --check` and rustdoc with
`-D warnings` are clean.

| part | cases | result |
|---|---|---|
| rules: transitions (full next state), masks, legality, starvation, game over, placement values | about 142k transitions over 13 configs (4 ladder modes, 3- and 2-seat variants, fast royale, hazard damages 0 to 101, healing hazards, 7x7, 9x11), plus 3,000 royale zones | 0 mismatches |
| rules: adversarial review (head-ons of 2 to 4 snakes, stacked tails, necks, hazard kill boundaries, crowded small boards, royale shrink edges, last-two draws) | about 507k transitions, 21,600 + 15,000 placement records | 0 mismatches |
| obs: boards and features, compared as f32 bits | 13,000 states, 43,000 seat boards (about 243M floats), 4 told modes | 0 mismatches |
| net: M (deployed), fresh S and L, S with `features` = `no_mode` and `none` | 940 cases each | real states: max abs dlogit 1e-6 to 9e-6, max abs dvalue up to 6.9e-6. Random inputs (M): max abs dlogit 2.7e-4 at abs(logit) up to 209 (relative 6.6e-5) |
| net: odd architectures (strides 1/3/4, five stride-2 layers, odd widths) | 21 end-to-end states each | max relative dlogit under 1e-5 |
| api: request to config, mode, state and safe move | 3,659 golden cases + 4,893 review cases (including a 4,000-case differential fuzz) | 0 problems |
| search: JAX threefry noise | 27 key, split, fold_in and bit streams | bit-exact |
| search: mechanics (uniform priors, zero values), 20 to 256 simulations, several constants | 2,980 searches | bit-exact (visits, q, policy, value, priors, moves, node count, depth) |
| search: real agent with Python's network outputs replayed | 2,300 searches | bit-exact |
| search: adversarial (crafted, random play, other constants) | 680 + 840 + 340 searches | bit-exact, root priors included |
| search: real agent with the Rust network, 256 simulations | 2,300 positions (golden seeds) | move agrees 100%. Visits equal for seat 0 in 99.91% of positions and for all seats in 99.87% |
| agent end to end (`Agent::play` on API requests) | 210 golden requests + 466 ladder-shaped requests | move agrees 100%, seat-0 visits equal 100% |

### Fresh positions (new seeds, not used during development)

The agent-with-Rust-network check was rerun on positions and search keys that no earlier stage had
seen. A scratch driver called `search_golden.py`'s `positions`/`agent_fns` with new rollout seeds
(bases 70000, 90000 and 110000) and new search-seed streams:

| run dir | positions | move agreement | visits equal (all seats) |
|---|---|---|---|
| M (deployed): duel 600, standard 4p 300, royale 4p 300, constrictor 4p 300 | 1,500 | **100%** (1,500/1,500) | 99.73% (4 cases) |
| M: standard3, royale2, constrictor2 (100 each) | 300 | 100% | 100% |
| M: `Agent::play` on fresh API requests (all 4 modes, 25% without the `/start` count) | 210 | 100% | 100% (seat 0) |
| `rocdxg-work/cmp-test/az-modes-S-s0` (a different architecture: 2 convs [4, 8], hidden 32): duel 150, 4p modes 100 each | 450 | 100% | 99.78% (1 case) |

In each fresh set, the replay with Python's network outputs was bit-exact (1,500 + 300 + 450
searches). The 5 positions whose visits differ also replay bit-exact. In those positions, the
Rust and Python networks differ by at most 6.6e-7 in value and 6e-7 in prior over Python's tree.
These are float near-ties, not search bugs. Root priors from the same logits differ from XLA's
softmax by at most 2 ulp (in 559 of 4,800 rows), because XLA uses a polynomial `exp`.

## Speed

Release build, one core of a Ryzen 7 5800XT (`taskset -c 7`, while other jobs were running on the
machine), M network, 256 simulations. Each config used 300 fresh realistic positions
(`examples/bench.rs`):

| config | mean ms/move | median | p90 | network rows/move |
|---|---|---|---|---|
| duel | 10.5 to 11.3 | 11.0 to 11.5 | 11.6 to 12.3 | 491 |
| standard 4p | 17.5 to 17.6 | 16.9 to 17.1 | 22.4 | 817 |
| royale 4p | 16.8 | 16.8 to 16.9 | 22.2 | 784 |
| constrictor 4p | 14.5 to 14.7 | 16.4 to 16.9 | 22.0 to 22.4 | 687 |
| standard3 / royale2 / constrictor2 | 14.8 / 11.6 / 8.9 | | | 678 / 512 / 413 |

The earlier development run (golden positions, quieter machine) measured 9.8 ms for a duel and
13 to 16 ms for 4p. Nearly all the time is in the network, at about 21 us per seat row (the
forward pass alone takes about 18 us). Dead seats are skipped, which saves about 25% of rows
in 4p. For comparison, Python/XLA takes 21 ms (duel) and 30 ms (4p) per move on **4** cores.
Loading the M run dir takes about 1.5 ms. The Battlesnake limit is 500 ms per move, including
the network round trip.

Run it with:
`NAZ_GOLDEN=... taskset -c N cargo run --release -p slinky-search --example bench -- <run dir> [cases per config]`.

## Strength tests (naz-match)

`naz-match` (crate `referee`) plays native agents against each other in the four ladder modes,
the native counterpart of `benchmarks/strength.py`. Duel: A and B alternate seats. 4 snakes: A
holds seat `g % 4` of game `g` and B the other three, which is the line-up of `evaluate.run_match`
and of `play_table` with `(0, 1, 1, 1)`. By default each seat searches separately with its own
noise, as independent ladder snakes do (see "Per-seat searches" below). Games run one per
thread. Each game is fixed by `(seed, game id)`, so the thread count does not
change any game, as long as both agents use simulation budgets.

```bash
B=$CARGO_TARGET_DIR/release/naz-match            # cargo build --release -p slinky-referee
R=/home/nathan/slinky-serve/az-modes-M-s0
# 1024 against 256 simulations: the duel stops sequentially, then 256 games of each 4p mode
$B --a "az_modes:$R:simulations=1024" --b $R --mode duel --games 1024 --sequential --out s.jsonl
$B --a "az_modes:$R:simulations=1024" --b $R --mode standard,royale,constrictor --games 256 --out s.jsonl
$B --a "az_modes:$R:time_ms=350" --b $R --mode duel --games 256 --view request --out t.jsonl
$B --table s.jsonl                               # markdown table of the summary lines
```

Agents:

* `random_legal`;
* `az_modes:<run dir>` with `:simulations=N` (default 256), or `:time_ms=T[:max_simulations=N]`
  (not deterministic), and optionally `:threads=K`, `:early_stop=1` and `:reuse=1` (the
  deployed engine's speed-ups that keep the moves) and `:int8=1` (the int8 network,
  `Network::quantize_int8`). A bare run dir also works.

Options:

* `--mode` (default `all`), `--games` (default 256, rounded up to whole seat rotations),
  `--round-games` (default 64), `--seed`, `--threads` (default: all cores);
* `--max-turns` (default 500): a harness cut-off, which counts as a draw if A is still alive;
* `--view request`: each seat gets the ladder's `/move` request and searches it, one search per
  seat. The default `state` view searches the true state, as slinky's strength tests do, and so
  it knows the royale shrink sides;
* `--full-games`: by default a 4p game stops once A is dead or alone. All the statistics are
  already fixed at that point;
* `--per-agent`: B plays its three seats from one search per turn, as slinky's `run_match` does.
  This is biased; see below.

Output is resumable. `--out F` gets one summary line per matchup. `F.games.jsonl` gets one line
per game, written as soon as the game ends. It holds the final alive flags, elimination turns
and causes, A's result, and each agent's search time and simulations per move. `F.rounds.jsonl`
gets one line per round, with strength.py's W/D/L columns first. Running the same command again
plays only the missing games. Raising `--games` continues a finished matchup and reuses its
games. A summary that is already in `F` is skipped unless `--rerun`.

Statistics, following slinky's conventions:

* W/D/L and the score, where a draw counts 1/2. A draw is a cut-off with A alive, or A dying on
  the last turn together with every other snake. The score has the Wald half-width of
  `slinky.evaluate` and the Wilson interval of strength.py;
* for the duel, Elo with its interval;
* for 4 snakes, A's share of the win (parity 0.25) and its placement (parity 2.5), each with a
  95% interval;
* per agent and mode, the search time per move (p50, p90, p99, max) and the simulations per
  move.

Stopping:

* `--target-ci W` stops once the Wilson half-width is at most `W`, after at least 64 games
  (`--target-ci-draws` uses the draw-aware interval instead);
* `--sequential` is the eval-speed branch's anytime-valid test. Four Kelly betting e-processes
  run on the per-round W/D/L counts, and the test reports an always-valid 95% confidence
  sequence (`cs95_*`, `elo_cs95_*`). Its parameters are `--seq-elo` (default 35) for the duel,
  or `--seq-p0` and `--seq-margin` for 4 snakes, plus `--seq-alpha` and `--seq-beta` (default
  0.05).

The test is ported to Rust rather than read back from Python. It is about a hundred lines of pure
functions, and the port keeps `naz-match` one binary that decides its stop between rounds on
machines without the JAX stack (the laptops, AWS).

Checks (`cargo test -p slinky-referee`, and `native/golden/referee_golden.py`):

* **Sequential test against the eval-speed Python.** 400 random cases (decisions, stop rounds,
  test ends) agree exactly. The log e-values agree to 4e-14 (Python 3.12's compensated `sum`).
  The confidence sequence agrees to 1e-16, since `lgamma` is CPython's.
* **Scoring against slinky's own code.** `referee_golden.py --only scoring --results F...`
  replays the referee's games through `evaluate.run_match` and `play_table`, using a stand-in env
  that follows each recorded game. W/D/L, truncations, mean turns, and each game's share and
  placement agree on 32 matchups (10,096 games: random and network agents, the laptop runs below,
  cut-offs at 25, 120 and 500 turns, full and early-stopped 4p games).
* **Other tests:** hand-checked final states, identical games with 1 and 4 threads in all modes,
  resume after a kill (torn line), continuing 32 to 64 games, `--target-ci`, a sequential stop,
  the request view, and per-seat against per-agent searches.

### Per-seat searches

slinky's `run_match` asks each policy once per turn and gives B's three seats the moves of one
search. Those three snakes then move consistently with one tree's predictions of each other,
which A's separate tree does not share. A 4p self-match is therefore biased against A. Measured
with M at 256 simulations, constrictor, 512 games on the same seed:

| searches | A's share (parity 0.25) | A's placement (parity 2.5) |
|---|---|---|
| per agent (`--per-agent`, slinky's) | 0.187 +- 0.028 | 2.74 +- 0.08 |
| per seat (default) | 0.255 +- 0.034 | 2.46 +- 0.09 |

Royale showed the same bias (0.201 +- 0.049 per agent). This bias also affects 4-snake
results from `benchmarks/strength.py` for searching agents. In the duel each agent has one seat,
so both settings play the same games.

### Measured (2026-10-15, M network, laptops, view `state`)

| matchup | mode | games | result |
|---|---|---|---|
| 256 vs 256 (self) | duel | 512 | score 0.479 [0.436, 0.522], Elo -15 [-45, +15] |
| 256 vs 256 (self), per seat | standard / royale / constrictor | 256 / 256 / 512 | share 0.236 +- 0.051 / 0.275 +- 0.054 / 0.255 +- 0.034; placement 2.55 / 2.44 / 2.46 |
| 1024 vs 256 | duel, `--sequential` | 64 (stopped: A>B) | score 0.758 [0.640, 0.846], Elo +198 [+100, +296], always-valid CS [0.533, 0.909] |
| 1024 vs 256, per seat | standard | 256 | share 0.490 +- 0.060, placement 1.96 +- 0.14 |
| 1024 vs 256, per seat | royale | 256 | share 0.408 +- 0.060, placement 2.12 +- 0.14 |
| 1024 vs 256, per seat | constrictor | 256 | share 0.319 +- 0.051, placement 2.35 +- 0.12 |
| `time_ms=200` vs 256 | duel, `--sequential` | 64 (stopped: A>B) | score 0.805 [0.692, 0.883], Elo +246 [+140, +352]; 1,968 simulations per move (p50) |

More search wins clearly in every mode. Each laptop core ran one game, with 7 to 9 games at once
on 8 or 12 hyperthreads, so the per-move times are slower than one quiet core. Ms per move
(p50 / p99) on the ThinkPad with 9 games at once:

* duel: 1024 simulations 92 / 212, 256 simulations 24 / 72;
* standard 4p: 1024 simulations 137 / 456;
* `time_ms=200`: 200.1 / 200.3, max 204.0. The deadline holds under load.

On pop-os-2 with 7 games at once, 256 simulations took 44 (duel) to 70 (4p) ms at p50.

## Weight-loading contract

`Network::load(dir)` and `Agent::load(dir)` read `dir/config.json` and `dir/params.npz`. Those are
the files `unified.save_params` and the az_modes trainer write, so a retrained run dir can be
used as-is.

* `config.json` must say `"algorithm": "az_modes"` and `"compute_dtype": "float32"`. A
  bfloat16 config is refused, because only f32 is implemented.
* The architecture comes from `conv_channels`, `conv_strides` (same length, non-zero; any count
  of 3x3 SAME convs), `hidden` and `features` (`all`, `no_mode` or `none`). S, M and L
  (`unified.SIZES`) and any other sizes work. Tested: S, M, L, a 2-conv [4, 8] / 32 net, strides
  1, 3 and 4, five stride-2 layers, channel counts 5 to 64 and hidden widths 16 to 100.
* `game_modes` (non-empty, known names, no duplicates) decides which modes the agent plays.
  `hide_constrictor_food`, `c_puct`, `fpu_reduction` and `draw_value` are read too. Missing fields
  take the `az_modes.Config` defaults. Training-only fields are ignored.
* `params.npz` is matched by name (`conv<i>/w|b`, `hidden/w|b`, `feat/w` unless `features` is
  `none`, `policy/w|b`, `value/w|b`), and every shape is checked against the config. Extra
  arrays and key order do not matter. The input is always the 11x11 board's 21x21x13 window.
* Every failure is an error (`NetError`), never a panic. Snapshot files such as `params-256.npz`
  are not loaded, only `params.npz`.

## Known gaps and risks

1. **Lambda packaging is not done.** `crates/lambda` builds and routes requests, but its engine hook
   (`engine_choose_move`) is still a stub that answers "up". Its `slinky-search` dependency is
   commented out, and nothing has been deployed (that needs AWS credentials). The remaining work:
   * enable the dependency;
   * call `Agent::respond` with the cached `/start` count;
   * load the weights once per container from `WEIGHTS_DIR`;
   * optionally answer HTTP 400 on `ApiError::Json`, as Python does.
2. **No time-budget search yet.** The faithful mode always runs 256 simulations. `Budget::until`
   can stop at a deadline, but it is untested for strength and has no parity claim. At about
   11 to 18 ms per move on one core, there is plenty of headroom for more simulations on Lambda.
3. **The snake count is ambiguous.** The API does not send the starting snake count. The engine
   follows the Python server: `max(count seen at /start, snakes listed)`, otherwise the
   *living* snakes. A stateless or restarted Lambda therefore reads a 4p standard game with two
   snakes left as a **duel** (duel mode, 2 seats), and a 4p royale or constrictor game keeps its
   mode but loses seats. This matches Python exactly, but it may not be what you want. The
   Lambda's `/start` cache is per process, so a cold container mid-game falls back to the live
   count. A durable store keyed by game id, or a policy such as "4 seats unless it started as a
   duel", is a product decision (SPEC.md section 8, point 1).
4. **Royale's future shrinks are unknown.** The shrink sides are rebuilt from the hazards in the
   request, and sides not shown yet are UNKNOWN (they move no edge). So the search assumes the
   zone stays put, and a shrink due next turn is not anticipated. This matches Python. A better
   model would be a strength change, not a port.
5. **No heuristic fallback.** Where Python's server plays its `heuristic` fallback agent, Rust
   answers the safe move. That covers boards other than 11x11, a single seat, and a mode the run
   was not trained on. Ladder games never get there. Porting `slinky.heuristic` (about 950 lines,
   boards up to 25x25) is follow-up work.
6. **Small API differences on inputs no engine sends** are documented in `crates/api/src/convert.rs`
   and `py.rs`. Examples: a head mixing a string and a float coordinate, integers beyond i128,
   lone UTF-16 surrogates and a list `turn`.
7. **Exact visit counts are not guaranteed.** Moves agreed in all 5,436 positions and requests checked with
   the Rust network (golden, fresh and ladder-shaped). Still, about 0.1 to 0.3% of positions have slightly different visit
   counts, from 1e-6 network rounding near ties, so a rare move flip in a near-tie is possible.
   The test threshold is 99%.
8. **Tie-noise seed.** `Agent::play(.., seed)` uses `jax.random.key(seed)`. The Python server
   uses `fold_in(key(0), move_counter)`, with a process-global counter. `JaxKey::fold_in` can
   reproduce it, but the Lambda has to choose a seed policy. Either choice gives a valid move;
   only bit-for-bit equality with a particular server process depends on it.
9. **Build warning.** Cargo warns about an output filename collision: four crates have an
   example named `bench`. Rename them before Cargo makes this a hard error.
10. **Architecture coverage.** Speed was measured on x86-64 (an SSE2 baseline build), not on the
    Lambda's arm64 (Graviton). Expect a different per-row cost there. The code has no
    architecture-specific paths.
