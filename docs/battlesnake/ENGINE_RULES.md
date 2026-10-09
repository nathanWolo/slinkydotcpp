# Battlesnake Engine Rules (precise reference)

The official prose rules (`official/rules.md`) are a good overview, but they
are vague or slightly inaccurate in places. This file records the exact
behaviour of the official rules engine, which is the real source of truth for
how a turn resolves. Use it when writing a simulator or search.

Source: <https://github.com/BattlesnakeOfficial/rules> at commit `87e094e`
(2025-12-24). This file describes the engine's behaviour in prose. No engine
code is copied here, because the engine is AGPL-3.0. File names are given so
you can check the details yourself.

---

## 1. Board and coordinates

- `(0,0)` is the **bottom-left** corner. `x` increases to the right and `y`
  increases **upward**.
  - `up` = `y+1`, `down` = `y-1`, `right` = `x+1`, `left` = `x-1`.
- Standard sizes are 7×7, 11×11 (the most common), 19×19, 21×21 and 25×25.
  - The standard map accepts any odd size from 7 to 25.
  - Always read `board.width` and `board.height` from the request instead of
    hard-coding them.
- `body[0]` is the head and `body[len-1]` is the tail.
  - **Body segments can be stacked on the same square.** This happens at the
    start of the game and right after eating.
- Max health is 100. Snakes start with length 3 and health 100.

## 2. Turn numbering and the request loop

1. The initial board is **turn 0**.
   - The first `/move` request is sent with `turn: 0`.
2. Every living snake is sent the same state in parallel. Then:
   1. the ruleset pipeline runs (section 3);
   2. the map's post-update runs (section 4: food and hazard spawning);
   3. `turn += 1`.
3. A snake eliminated while resolving the moves for request turn `T` is
   recorded with `eliminatedOnTurn = T+1`.

### Invalid or missing moves (`standard.go`, `getDefaultMove`)

If a move is not one of `up/down/left/right`, the engine picks a default move:

- It continues in the **direction from the neck (`body[1]`) to the head
  (`body[0]`)**. Wrapped boards are handled.
- If the head and neck are on the same square, as on turn 0 when the body is
  stacked, the default is **`up`**.

In the local CLI, a timeout or bad response reuses the snake's previous valid
move. In practice you keep going straight, which may be into a wall.

## 3. Ruleset pipeline (the order matters)

Standard order, from `standard.go`, `standardRulesetStages`:

1. **Game-over check** (runs *before* the moves are applied).
   - Standard: the game is over when **≤ 1** snakes remain.
   - Solo: the game is over when **0** snakes remain.
2. **Move.** For each living snake: put the new head at the front, then
   **remove one segment from the tail**.
3. **Starvation.** Every living snake loses 1 health.
4. **Hazard damage** (section 5). Hazard damage can eliminate a snake
   **immediately**, during this stage.
5. **Feed.** For each food, *every* living snake whose head is on that square
   eats it:
   - health is set to 100;
   - the tail segment is **duplicated** (appended again), so length +1 right
     away in the body array;
   - the food is removed.
   - If two heads land on the same food, **both** eat.
6. **Eliminate** (section 6).

The variants change the pipeline as follows:

| Ruleset | Difference from standard |
|---|---|
| `standard` | The baseline above. |
| `solo` | The game ends only when every snake is dead. |
| `royale` | Standard rules plus a hazard map that shrinks the board (section 7). |
| `wrapped` | After moving, a head that went past an edge wraps to the opposite edge. Nobody goes out of bounds. |
| `constrictor` | After elimination: **all food is removed**, health is reset to 100, and every snake grows by 1 every turn, so tails never move. |
| `wrapped_constrictor` | Wrapped and constrictor combined. |

## 4. Health and food: consequences of the order

- **Starvation happens before feeding, and the out-of-health check happens
  after feeding.** A snake at health 1 that moves onto food goes 1 → 0 → 100
  and survives. A snake at health 1 that does *not* eat dies this turn.
- In general, a snake with health `h` that does not eat dies after `h` more
  moves (ignoring hazards).
- **Hazard damage happens before feeding**, but **food on a hazard square
  cancels that square's damage completely** (section 5).

## 5. Hazards (`standard.go`, `DamageHazardsStandard`)

- Hazard damage only applies when the snake's **head** ends the move on a
  hazard square.
- The amount is `hazardDamagePerTurn`. The CLI default is **14**, so 15 in
  total with starvation.
- If the head's square also contains food, the snake takes **no** hazard
  damage. It then eats normally.
- Health is clamped to `[0, 100]` after the damage.
  - Negative damage would heal; some maps use this.
- If health reaches 0, the snake is eliminated **immediately**, with cause
  `hazard`, so food later in the turn cannot save it.
  - This only matters when there is no food on that square, because food
    there would have cancelled the damage.
- **Hazards stack.** The `hazards` list can contain the same square more than
  once, and each copy deals damage separately. Maps such as Sinkholes and
  Hazard Pits layer hazards this way.

## 6. Elimination (`standard.go`, `EliminateSnakesStandard`)

The checks run in two passes.

**Pass 1.** These snakes are eliminated immediately:

- **Out of health:** `health <= 0`.
- **Out of bounds:** any body point lies off the board. In practice only the
  head can be off the board.

Snakes removed in pass 1 are gone *before* collisions are checked. Their
bodies cannot kill anyone, and they cannot win or lose head-to-heads.

**Pass 2.** Collisions are evaluated against the snakes that survived pass 1,
using the bodies **after** moving and feeding. The results are collected
first and applied together at the end. Each snake is checked in this order,
stopping at the first hit:

1. **Self collision:** the head is on any of the snake's own segments at
   index ≥ 1.
2. **Body collision:** the head is on any segment at index ≥ 1 of another
   living snake.
   - Heads are not "body" for this check.
3. **Head-to-head:** the head is on another snake's head, and
   **`my_length <= other_length`**.
   - With equal lengths, both die.
   - With 3 or more snakes on one square, a snake dies if *any* other snake
     there is at least as long. For example, with lengths 5, 5, 4 on one
     square, all three die.
   - Lengths are compared **after feeding**. If both heads ate the same
     food, both grew by 1, so the comparison is unchanged.

Because collision results are applied together, a snake that dies from a
body collision this turn can still win a head-to-head against a shorter
snake on the same turn. That shorter snake also dies.

### The tail rule (important for move generation)

The engine removes the tail **before** checking collisions. So:

- Moving into the square where **any** snake's tail currently is, including
  your own, is **safe**, *unless that snake ate on the previous turn*.
  - You can tell a snake ate last turn when its last two body segments are
    the same square. In that case the tail does not leave its square.
- If that snake eats **this** turn, the tail square is still vacated. The
  duplicate segment is added to the *new* tail (the old second-to-last
  segment), not to the old tail square.
- Exception: in **constrictor**, snakes grow every turn, so tails never move.
- At game start all 3 segments are stacked. The tail effectively stays put
  for the first 2 moves while the snake uncoils.

## 7. Maps: setup and spawning (`board.go`, `maps/standard.go`, `maps/royale.go`)

### Starting positions (standard map, square board ≥ 7×7, ≤ 8 snakes)

Let `W` be the board width, `mn = 1`, `md = (W-1)/2` and `mx = W-2`.

- **Corners:** `(mn,mn)`, `(mn,mx)`, `(mx,mn)`, `(mx,mx)`.
- **Cardinals:** `(mn,md)`, `(md,mn)`, `(md,mx)`, `(mx,md)`.
- Each group is shuffled. Then, with 50/50 odds, either all corners or all
  cardinals are used first.
- All 3 segments of each snake are stacked on its start square, at health
  100.
- With more than 8 snakes, square boards ≥ 11×11 use spread-out quadrant
  placement instead. Other boards place snakes randomly on squares where
  `x + y` is even.

### Starting food (square board ≥ 7×7)

If there are ≤ 4 snakes, or the board is at least 11×11:

- Each snake gets one food on a **diagonal** neighbour of its head.
  - The food is always on the side **away from the board centre**.
  - It is never on a corner square, and never on the centre square.
- One food is always placed **on the centre square**.

On a small board with more than 4 snakes, only the centre food is placed.

### Food spawning each turn (standard map)

This runs after the ruleset pipeline, on the board after eliminations.

1. If `food_count < minimumFood`, enough food is added to reach
   `minimumFood`. The CLI default `minimumFood` is **1**.
2. Otherwise, one food is added with roughly `foodSpawnChance`% probability.
   - The CLI default is **15**.
   - The implementation actually gives `(foodSpawnChance - 1)`%, so about
     14%.
3. Valid spawn squares are squares that:
   - are not occupied by food or by a living snake's body;
   - are **not orthogonally adjacent to any living snake's head**.
   - Hazard squares **are** allowed.

So new food never appears right next to a head, and you can never be
surprised by food appearing in your next square.

### Royale

- The board uses standard food spawning.
- Every `shrinkEveryNTurns` turns (CLI default 25; your request contains the
  real value), the safe zone shrinks by one row or column on a randomly
  chosen side.
- Every square outside the safe zone is a hazard.

### Other maps in the engine

These are given for reference. Read the `hazards` list in the request to see
where hazards actually are.

`empty`, `arcade_maze`, `solo_maze`, `snail_mode` (snakes leave hazard
trails), `healing_pools`, `sinkholes`, `hz_hazard_pits`, `hz_inner_wall`,
`hz_rings`, `hz_columns`, `hz_spiral`, `hz_scatter`, `hz_grow_box`,
`hz_expand_box`, `hz_expand_scatter`, `hz_castle_wall*`,
`hz_rivers_bridges*`, `hz_islands_bridges*`.

## 8. Timing

- The default move timeout is **500 ms**. Read the real value from
  `game.timeout` in each request.
- `you.latency` reports how long your previous response took.
- Leave a safety margin for network latency. Many competitive snakes stop
  searching around 350–400 ms on a 500 ms timeout.

## 9. Quick checklist for a simulator

A simulator that matches the engine needs to do all of the following:

- [ ] Move all snakes at the same time: add the new head, then remove the
      tail.
- [ ] Apply health −1, then hazard damage. Cancel hazard damage on squares
      with food, and eliminate immediately at 0.
- [ ] Feed every head on a food square: health = 100, duplicate the tail,
      remove the food.
- [ ] Eliminate out-of-health and out-of-bounds snakes first, and leave them
      out of the collision checks.
- [ ] Collisions: check self, then other bodies (index ≥ 1), then
      head-to-head (`<=` loses). Collect all results before applying any of
      them.
- [ ] Track "ate last turn" through the duplicated tail segment, so the tail
      rule works correctly.
