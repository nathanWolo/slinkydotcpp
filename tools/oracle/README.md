# Rules-engine oracle (test-only)

`oracle` is a small Go program that drives the **official Battlesnake rules
engine** ([`github.com/BattlesnakeOfficial/rules`](https://github.com/BattlesnakeOfficial/rules),
pinned in `go.mod` to commit `87e094e`, pseudo-version
`v1.2.4-0.20251224171327-87e094e2e1c2`). The slinky test-suite uses it as
ground truth for the JAX reimplementation.

**Licensing:** the engine is AGPL-3.0. This tool *links* it (it is fetched as
a Go module, not vendored or copied). The tool is test-only: it is not part
of the `slinky` Python package, never imported by it, and not distributed
with it. The built binary (`bin/`) and generated data (`.oracle-cache/`) are
gitignored.

## Build

Needs Go ≥ 1.21 and network access to the Go module proxy on first build.

```sh
cd tools/oracle && go build -o bin/oracle .
```

The pytest bridge (`tests/oracle.py`, `build_oracle()`) does this
automatically. It rebuilds when any `.go`/`go.mod`/`go.sum` file is newer
than the binary. It looks for Go in `$ORACLE_GO`, then `PATH`, then
`/usr/local/go/bin/go`. Oracle tests are skipped when the oracle cannot be
built. Run them with `pytest -m oracle`.

## `oracle step`: one ruleset step per request

Reads JSON Lines on stdin and writes one JSON line per request, flushed after
each line. Python can therefore drive it interactively or pipeline many
requests.

```json
{"ruleset": "standard", "solo": false, "seed": 1,
 "settings": {"foodSpawnChance": "15", "minimumFood": "1", "damagePerTurn": "14", "shrinkEveryNTurns": "25"},
 "state": STATE, "moves": ["up", "left"]}
```

- The request runs
  `rules.NewRulesetBuilder().WithParams(settings).WithSeed(seed).WithSolo(solo).NamedRuleset(ruleset).Execute(state, moves)`.
  It runs the ruleset pipeline **only**: no map pre- or post-update, no food
  spawning, and no `turn += 1`.
- `ruleset` must be one of `standard`, `solo`, `royale`, `wrapped`,
  `constrictor` or `wrapped_constrictor`. An unknown name is an error. The
  engine itself would silently fall back to standard.
- `settings` uses the engine's parameter keys (see `constants.go`) and is
  passed through unchanged. Numbers and booleans are converted to strings.
- `seed` defaults to 1 when absent. Only the royale ruleset's hazard stage
  uses randomness. Seed `0` means "unseeded" to the engine, which then uses
  global `math/rand`.
- `moves[i]` is the move for snake `i`, sent with that snake's id.
  - Send `""` for eliminated snakes; the engine ignores them.
  - Any string other than `up`/`down`/`left`/`right` makes a living snake use
    the engine's default move.
  - Fewer moves than snakes is allowed: a living snake with no move is an
    engine error.
  - `"moves": null`, or no `moves` key, calls `Execute(state, nil)`, which is
    the CLI's turn-0 initialisation call.

Response: `{"game_over": bool, "state": STATE | null, "error": ""}`. On an
engine error, a malformed request or an engine panic, `state` is `null` and
`error` is non-empty, and the process keeps serving requests.

## `oracle games`: full games with the real map and ruleset

Plays complete games exactly like `cli/commands/play.go`:

1. `maps.SetupBoard`, then `ruleset.Execute(state, nil)`.
2. Then, until game over or `--max-turns`, each turn runs `PreUpdateBoard`,
   then `Execute`, then `PostUpdateBoard`, then `Turn += 1`.

Moves come from a built-in seeded policy instead of HTTP snakes.

```sh
bin/oracle games --ruleset royale --map royale --snakes 4 --games 100 --seed 1 > royale.jsonl
```

| flag | default | |
|---|---|---|
| `--ruleset` | `standard` | ruleset name |
| `--map` | `standard` | map ID (`standard`, `royale`, `empty`, ...) |
| `--width`, `--height` | 11 | board size |
| `--snakes` | 2 | snake ids are `s0`..`s{N-1}`; `snakes == 1` enables the solo game-over rule, like the CLI |
| `--games` | 10 | |
| `--seed` | 1 | game `g` (0-based) uses a hash of `(seed, g)` for the engine seed and for the policy RNG (consecutive Go `math/rand` seeds give correlated streams) |
| `--max-turns` | 500 | maximum transitions per game |
| `--food-spawn-chance` | 15 | engine `foodSpawnChance` |
| `--minimum-food` | 1 | engine `minimumFood` |
| `--hazard-damage` | 14 | engine `damagePerTurn` |
| `--shrink-every-n-turns` | 25 | engine `shrinkEveryNTurns` |
| `--invalid-move-prob` | 0 | chance that a living snake sends `""`, `"bogus"`, `"UP"` or `"none"` |
| `--p-random` | 0.03 | policy: uniformly random move |
| `--p-aggressive` | 0.3 | policy: greedy towards the closest other head |
| `--p-food` | 0.3 | policy: greedy towards the closest food |
| `--p-food-averse` | 0.15 | per snake per game: never seeks food, and avoids it when possible |
| `--p-careful` | 0.75 | per snake per game: avoids dead ends (flood fill) and squares that a not-shorter head can also reach |

Policy, per living snake per turn:

1. With probability `p-random`, play a random move.
2. Otherwise the snake considers only *safe* moves: moves that stay on the
   board (or wrap) and do not enter `body[:-1]` of any living snake. A tail
   vacates unless its last two segments are equal.
3. Among the safe moves, it chases the closest other head with probability
   `p-aggressive`, the closest food with probability `p-food`, and otherwise
   picks uniformly.
4. If no move is safe, it plays a random move.

Lower `--p-careful` and raise `--p-aggressive` for more collisions, including
3- and 4-snake head-ons on 7x7. Raise `--p-careful` for longer games, with
more starvation and royale hazard deaths.

Output is JSON Lines:

```json
{"kind": "initial", "game": 0, "config": {...flags..., "solo": false, "settings": {...}, "game_seed": 1}, "state": STATE}
{"kind": "transition", "game": 0, "pre": STATE, "moves": ["up", "left"], "post_rules": STATE, "post_map": STATE, "game_over": false}
```

- `initial` is the state after `SetupBoard` and the initial
  `Execute(state, nil)`.
- `pre` is the state the snakes saw, after `PreUpdateBoard`.
- `post_rules` is the output of `Execute`.
- `post_map` is the state after `PostUpdateBoard` and `Turn += 1`.
- `moves[i]` is snake `i`'s move, or `""` if the snake was already
  eliminated. Like the CLI, only living snakes' moves are passed to `Execute`.
- `config.settings` and `config.game_seed` are exactly what the engine was
  given. Pass them to `oracle step` to replay a transition.
- Because the engine checks game over *before* moving, the last transition
  of a finished game has `game_over: true` and `post_rules == pre`. The CLI
  still runs `PostUpdateBoard` and `Turn += 1` on that step, so `post_map` may
  gain food.

## STATE

```json
{"width": 11, "height": 11, "turn": 5,
 "snakes": [{"id": "s0", "body": [[x, y], ...], "health": 90,
             "eliminated_cause": "", "eliminated_on_turn": 0, "eliminated_by": ""}],
 "food": [[x, y], ...],
 "hazards": [[x, y], ...]}
```

- Snakes are listed in engine (index) order, and eliminated snakes are kept
  with their bodies as the engine holds them.
- Food and hazards are listed in engine order, with duplicates. Stacked
  hazards each deal damage.
- The engine's point `TTL`/`Value` fields and the map-only `GameState` and
  `PointState` fields are not exposed. The standard and royale maps do not
  use them.
