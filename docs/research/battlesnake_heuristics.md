# How practical Battlesnake bots play: heuristics and search

Research notes for building a strong heuristic opponent (and an MCTS leaf
evaluator) for slinky's 1v1 duel on 11x11 with the standard rules. The last
section is a concrete spec.

How to read this file:

- **[src]** marks a claim taken from a source listed in section 7. Every URL
  in section 7 was opened while writing this note, except two that are marked
  there as seen only in search results. Where a page could not be opened, that
  is said in section 7, and nothing here depends on it.
- **[measured]** marks a result from a throwaway JAX prototype of the spec
  written for this note (not committed). Games were played with
  `slinky.evaluate.play_match`: seats alternate, win = 1, draw = 1/2, and ±
  is the 95% half-width. Typical sample size is 512 games, so differences
  under about 0.05 are noise.
- Everything else is reasoning from the rules in `docs/battlesnake/ENGINE_RULES.md`.

Contents:

1. Summary
2. Rules that drive the heuristics
3. Heuristics in practice
4. Search methods
5. Evidence from slinky
6. Recommended heuristic snake for slinky
7. Sources

---

## 1. Summary

- Almost every strong open-source snake is a depth-limited search plus a
  hand-written static evaluation. [src] The common evaluation ingredients are
  reachable area (flood fill), territory (Voronoi-style "who gets there
  first"), length advantage, and food/health pressure.
- In 1v1 the search is nearly always paranoid minimax with alpha-beta and
  iterative deepening. MCTS snakes exist but are rarer. [src]
- The field's most repeated lesson (from Tron, the closest ancestor game) is
  that a better evaluation beats deeper search, and that MCTS needs a good
  heuristic to compete with alpha-beta. [src]
- Territory, food control and length advantage carry most of the strength.
  Time-aware reachability (a body cell is free once its tail has passed) is
  worth a lot: in the prototype, treating bodies as permanent walls cost 0.23
  of score. [measured]
- Avoiding head-to-heads with equal-length snakes is *not* free. Treating an
  equal-length meeting as forbidden makes the snake timid and loses ground.
  Treating it as a draw with a small penalty ("contempt") works better.
  [measured]
- Recommended for slinky: a one-ply simultaneous-move search over the exact
  `rules_step` (all 4x4 joint moves), scored with a static evaluator built
  from time-aware Voronoi territory, food-weighted territory, length
  difference, and starvation/trap pressure. The same evaluator serves as the
  MCTS leaf value. In the prototype it scores 0.999 vs `random_legal`, 0.954 vs
  the shipped DQN, and 0.75 vs a feature-only one-ply snake. [measured]

---

## 2. Rules that drive the heuristics

Details are in `docs/battlesnake/ENGINE_RULES.md`. The parts that matter here:

- **Head-to-head.** Same square, the longer snake survives; equal lengths both
  die. Lengths are compared after feeding. [src: official rules, engine notes]
- **Tail rule.** The tail leaves before collisions are checked, so moving into
  any snake's tail square is safe unless that snake's tail is stacked (it ate
  last turn). In slinky's countdown grid this is simply "a cell with countdown
  `k` is enterable at step `d >= k`".
- **Starvation order.** Health drops, then food is eaten, then the health check
  happens: a snake on health 1 that eats survives.
- **Hazards.** Damage applies only if the head ends the move on the hazard
  square, and food on that square cancels it. [src: Royale map doc] slinky's
  current maps (`standard`, `empty`) have no hazards and `royale` is not
  implemented, so hazard handling below is a documented extension.
- **Food is scarce in a duel.** With the default settings there is usually
  one to three food items on the board, so food control is a strategic
  resource, not just a health refill.

---

## 3. Heuristics in practice

### 3.1 Flood fill, reachable area, traps

- The official docs describe flood fill as "a simple way to detect and avoid
  enclosed spaces that Battlesnakes may not be able to escape from" and
  suggest counting the squares reachable after each candidate move. [src: 1]
- Winning snakes use it as an evaluation term: Orion's Fang (2nd place, RBC
  Spring 2023) scores positions with flood fill and edge control under 7-ply
  paranoid minimax [src: 9]; Calvin Lin's snake (1st place, RBC Winter 2022)
  lists A* and flood fill as the heuristics behind its 5-ply minimax [src: 10].
- Pure counts overestimate: they ignore bodies that will move away and
  corridors that waste space. The better flood fills delete tails as the fill
  proceeds. The 2024 Albatross paper (which includes a Battlesnake benchmark)
  describes a flood fill that "dynamically deletes the current tail of all
  snakes in every iteration", with one exception: a snake that ate on the last
  turn keeps its tail for the first iteration [src: 6]. This is exactly the
  countdown-grid reachability used in section 6.
- A trap penalty on `reachable < own length` is common (FusionSnake subtracts
  `trap_penalty * (length - reachable)`) [src: 12]. Redbrick's 2017 bounty
  snake weighted the final score by open area reachable from the position
  [src: 7].
- Tron-era refinements, in case traps still cost games: an upper bound on the
  moves available in a region is `cells - |black - white|` (checkerboard
  parity), and the "tree of chambers" splits a region at articulation points
  [src: 5, 14]. The prototype tried the parity bound and saw no gain
  [measured: 0.479 ± 0.041 vs the plain count], so it is left out.

### 3.2 Voronoi territory and area control

- A cell belongs to whoever can reach it first. a1k0n's Google AI Challenge
  (Tron) post-mortem: "for each spot on the map, find whether player 1 can
  reach it before player 2 does or vice versa". He later replaced raw counts
  with a linear model over nodes and edges fitted on 11,691 games by top-100
  players (K1 ≈ 0.055 per node, K2 ≈ 0.194 per edge), and when the players
  were separated the value became 1000 x the difference in component sizes
  [src: 5].
- For Battlesnake, the 2019 Victoria paper evaluates "board control" with a
  diamond-shaped flood filler [src: 3]; Albatross uses the same flood fill
  started from every head with ties broken by length, "according to the
  head-to-head collision rule" [src: 6]; FusionSnake runs a lockstep BFS from
  every head and marks cells reached by two snakes as contested [src: 12];
  Hovering Hobbs and a "Jump Flooding" snake in coreyja's collection use
  area-control scores [src: 2]; Sandworm (a C minimax snake ranked 4th of 500+
  on the standard ladder and 5th of 400+ on duels) is described as "minimax
  over Voronoi" [src: 17].
- Counterpoint: Redbrick's 2017 post-mortem says "Tron games typically use a
  voronoi diagram here, but that wouldn't work very well in a snake game"
  [src: 7]. And Typhon's authors measured their Voronoi and "opponent
  confinement" terms as costing games, in *four-snake* games with a 4,000-node
  budget and an expensive implementation (Voronoi was 74% of run time)
  [src: 11]. Their benchmark file lists whether the result holds in duel play
  as an open question. In the slinky prototype, removing the territory or the
  food-weighted territory term were among the largest losses in a duel
  [measured, section 5].

### 3.3 Head-to-heads: avoid, and hunt

- Rule of thumb: never step where a longer-or-equal snake can step at the
  same time, and look for such squares when you are the longer snake.
  FusionSnake encodes it directly: a penalty when an enemy is within distance
  1 and is at least as long, a bonus when an enemy is within distance 2 and
  shorter [src: 12].
- Son of Robosnake (Redbrick, 2018) first played "too defensively. It would
  never move into a square that the enemy snake could also move into", then
  modelled collisions as valid moves for both and added an aggression term
  that rises as the snake nears a square the enemy can enter next turn,
  weighted toward the square ahead of the enemy's head [src: 8].
- Devious Devin (paranoid minimax) "hunts food when short, stalks opponents
  when long" [src: 2].
- Search handles this exactly if the simultaneous move is modelled: the
  opponent cannot see your move, so the outcome of a meeting is a function of
  both moves. See section 4.1.
- Equal lengths need care. A meeting at equal length kills both snakes, which
  is a draw in a duel (worth 1/2). [measured] A snake that treats such cells
  as forbidden yields territory (variant without the equal-length avoidance
  scored 0.595 ± 0.041 against the version with it), while two snakes that
  ignore the risk trade evenly at the start (the tuned variant against a near
  copy of itself: 83% draws, 28 turns on average). A small "contempt" penalty
  on drawn terminal states was the best compromise found (section 5).

### 3.4 Food: when and how much

- The usual rule is "ignore food until it matters". Son of Robosnake: ignore
  food unless health ≤ 40, or length < 3, or ≤ 8 food items remain [src: 8].
  Redbrick 2017: reward proximity to food with a weight that rises as health
  falls [src: 7]. FusionSnake multiplies the food term by 3 when health is
  below a "desperate" threshold and doubles the area and length terms when it
  is healthy [src: 12].
- Length is strength: "Long Length = Strength" (Rusty always seeks food that
  has a safe path back to its tail) [src: 16]. Famished Frank (grow, then
  retreat to corners) is a solo-challenge snake [src: 2].
- Contestation matters. FusionSnake treats a food item as contested if an enemy
  at least as long is at least as close, and discounts it [src: 12].
  Territory-weighted food (count food items that lie inside your Voronoi
  region minus theirs) gets this for free. [measured] Removing that single
  term dropped the one-ply search's score against default P0 from 0.728 to
  0.422.
- Over-eating is not clearly bad in 1v1; the prototype's best weights favour
  eating adjacent safe food (food bonus x4 improved score to 0.536 ± 0.042).
  [measured]
- Typhon measured a "hunger vs distance to food" term and a reachable-space
  term as the only two that helped (both over 400 four-snake games on two
  seed blocks); a larger food weight "looked like a win at p=0.0003 and then
  reversed on a second seed block" [src: 11]. Expect flat optima.

### 3.5 Tail chasing and following

- Tail chasing is the standard way to stay alive in a closed region: the tail
  square frees up as you move. Rusty's priority list is food (if safe) > own
  tail > nearest enemy tail > "panic" [src: 16]. Eremetic Eric survives solo
  games "mostly by chasing its tail" [src: 2], and Nettogrof's challenge notes
  use "go in a corner, then tail-chasing until low health" [src: 15].
- Son of Robosnake counts enemy tails as safe squares unless the enemy just
  ate [src: 8], which is the same tail rule as in section 2.
- Typhon's "tail reachability" term (the largest weight, 40) was measured as
  harmful at weight 40 and "useless rather than harmful" at 10 [src: 11]. The
  slinky spec therefore uses tail reachability only as a safety test
  ("can the fill reach my own tail?") and gets the rest from time-aware
  reachability.

### 3.6 Edges, corners, centre

- Edge avoidance is common: Son of Robosnake weights outer squares "very
  unfavorably" [src: 8]; FusionSnake subtracts a penalty per axis on which
  the head touches the border [src: 12]; Redbrick 2017 favoured the board
  centre, both for food access and to push the opponent to the walls [src: 7].
- Typhon measured centre control as the worst of its removed terms ("worst of
  them and carried the smallest weight"), harmful in both directions (weight
  +1 and -1), in four-snake games [src: 11].
- Territory already prefers central squares: they are closer to more cells. The
  spec adds no explicit edge or centre term; add one only if a benchmark shows
  a need. (This is untested.)

### 3.7 Hazards

- Official behaviour (Royale map doc): entering a hazard square costs 14
  health in addition to the 1 per turn; food on a hazard square gives its full
  benefit without the penalty; a hazard that appears under a snake does no
  damage that turn [src: 4]. Snail mode stacks hazards [src: 4].
- Community write-ups in the sources above do not describe hazard heuristics
  in detail (Typhon lists hazard-pit maps as unsupported [src: 11]). Sensible
  handling follows from the rules: count a hazard step as `1 + damage` health
  in the starvation/hunger terms, treat it as passable but costly in the
  fill, and let food-on-hazard count as safe. See section 6.9.

### 3.8 How the terms are weighted

No source publishes a tuned duel recipe; all say weights were tuned
empirically.

| Source | Weighting |
|---|---|
| Typhon [src: 11] | Active: space 6, food 4, length 30 (not separated from zero). Measured harmful and switched off: centre 1, tail reachability 40, Voronoi 10 + confinement 6. 400 four-snake games per run, two seed blocks. "Barely tuned." |
| FusionSnake [src: 12] | Health tiers scale terms: desperate (area x1, food x3, length x1), balanced (all x1), healthy (area x2, food x1, length x2). Terms: area `2*mine - total`, trap penalty, body-proximity penalty, edge penalty, health (x3 under 20), contested-food distance, length difference vs longest enemy, aggression and head-to-head danger. Numeric defaults live in `heuristic_params.rs`/`params.json` (not retrieved). |
| a1k0n, Tron [src: 5] | Linear in nodes (0.055) and edges (0.194) of the Voronoi partition, fitted to 11,691 expert games; separated: 1000 x component-size difference. |
| Albatross baseline [src: 6] | Value for player i = 1/2 (area advantage relative to board size + own relative health - mean relative health). |
| Schier & Wustenbecker [src: 3] | Heuristic parameters tuned with grid search and a genetic algorithm. |
| snork "Tree" and "Flood" [src: 13] | Configurable terms (health, food distance, space, space advantage, size advantage, centrality) tuned with Bayesian optimization. |

Takeaway for weights: lexicographic tiers for hard safety (a legal move that
cannot be starved or trapped), then a weighted sum of smooth terms. Make the
sum a function of *differences* between the two snakes so it is antisymmetric
and bounded.

---

## 4. Search methods

### 4.1 Minimax, alpha-beta, iterative deepening

- **Paranoid minimax with alpha-beta** is the workhorse. "Paranoid Minimax gets
  its name by being 'paranoid' and thinking all the opponents are out to get
  you specifically": max at your nodes, min at every opponent's. Alpha-beta
  prunes cleanly; MaxN (each snake maximises its own score) cannot use the same
  pruning [src: 2b]. Snakes: Devious Devin and Hovering Hobbs [src: 2],
  Orion's Fang (depth 7) [src: 9], Calvin Lin's winner (depth 5) [src: 10],
  FusionSnake (iterative deepening, "4xN branching, not 4^N" by treating each
  enemy independently) [src: 12], Typhon [src: 11].
- **Iterative deepening** with a time check and "keep the best move from the
  last completed depth" is standard [src: 5, 11, 12]. a1k0n: when time runs out
  "you have to throw away the ply you're in the middle of searching" [src: 5].
  Typhon reports a mean depth of 8 in a 500 ms duel [src: 11].
- **Simultaneous moves** are the awkward part. Options seen:
  - Alternate plies (me, then opponent) and accept an omniscient opponent.
    Typhon's authors say alternation "makes the opponent omniscient" and
    instead advance the board only after every snake has committed [src: 11].
  - Son of Robosnake searches the opponent's reply from the board *before*
    your move and evaluates the heuristic only at even depths, so both
    snakes' moves have been applied [src: 8].
  - Treat the 4x4 joint moves as a matrix game. Tron research calls this the
    "stacked matrix" model; alternating plies is the "sequential tree" model
    and "clearly favours one player", though in Tron it still played well
    (Sequential UCT 51.4% overall) [src: 18].
  - In slinky the exact rules are available, so one-ply of the matrix game is
    cheap: 16 `rules_step` calls. Maximin over the opponent's replies is a
    safe (lower bound) value; a pure-strategy maximin cannot represent mixing,
    but it is the standard cheap choice. [measured: section 5]
- **Locality.** Search only nearby opponents. Schier & Wustenbecker restrict
  search to players in the acting snake's locality and combine iterative
  deepening with alpha-beta and max^n; their agent placed second in the
  intermediate division at the Victoria competition [src: 3]. Not needed for a
  duel.
- **Evaluation matters more than depth.** a1k0n: "a better evaluation heuristic
  will always beat deeper minimax searches" [src: 5]. Typhon's profile says the
  evaluation took 85.6% of run time and that "the search machinery ... is
  noise" next to it, while depth still paid: full search beat one-ply 192-0
  (8 splits) in duels [src: 11].

### 4.2 Evaluation functions inside search

- Terminal positions get exact values (win/loss, sometimes ranked by speed:
  Son of Robosnake still evaluates the heuristic when victory is predicted,
  so "win in 3" beats "win in 4" [src: 8]).
- Non-terminal values combine the terms in section 3, often in tiers of
  health. When the players are separated, switch to a survival comparison
  (component sizes; a1k0n then used an iteratively deepened exhaustive search
  inside the chamber) [src: 5].
- Values should be bounded and antisymmetric for a duel, so the same function
  serves minimax and MCTS (Albatross's baseline value is in this form) [src: 6].

### 4.3 MCTS for simultaneous-move games

- **Decoupled UCT (DUCT)** keeps separate statistics per player at each node and
  picks one action per player independently with UCB1; the joint action
  selects the child. It "has been shown to work well in a variety of games",
  although it does not always converge to a Nash equilibrium [src: 6, 19].
- **Comparisons.** In nine simultaneous-move games, Decoupled UCT performed best
  overall despite its theoretical weakness [src: 19]. In Tron, Decoupled
  UCB1-Tuned (final move = most visited) won 62.3% overall against seven other
  variants (sequential UCT 51.4%, plain DUCT 49.1%, Exp3 35.5%, regret
  matching 53.1%), with 13x13 boards and 100,000 simulations per move; tuned
  `C = 1.5` [src: 18]. Convergence theory for Exp3 and regret matching:
  [src: 20].
- **MCTS vs alpha-beta in Tron.** Den Teuling's best MCTS programs won only
  8-18% (plain MCTS-UCT 14 ± 3%) against a1k0n's alpha-beta program with the
  tree-of-chambers evaluation. The reason given: "the reliability of the
  play-outs rapidly drops as the players get more distant from each other"
  [src: 14]. Takeaway: random play-outs are a weak evaluator in
  space-filling games; a static evaluator (or play-out cut-offs with a
  heuristic) is how MCTS stays competitive.
- **MCTS in Battlesnake.** Improbable Irene (coreyja) uses MCTS with
  UCB1-normal selection [src: 2]. The Albatross paper's baseline agents are
  simultaneous-move MCTS with a "handcrafted value heuristic adapted from
  Schier and Wustenbecker (2019)", DUCT selection with `c = sqrt(2)` and no
  policy prior; their strength is set by the *number of tree-search
  iterations* (Figure 9, log axis roughly 10^2 to 10^4), exactly the
  benchmark axis planned for slinky [src: 6].
- For AlphaZero-style search on simultaneous games the same paper found a fixed
  depth search with a logit-equilibrium backup the best of the variants it
  tried, with DUCT close behind in the four-player mode [src: 6]. Not needed
  for the baseline here.

### 4.4 Benchmark hygiene

- Typhon's finding 015: seat assignment and board seeds were aliased, so two
  identical bots scored 120-80. Starting squares decide many duels. The fix:
  play each board twice with the seats swapped and count only boards one side
  wins from both seats [src: 11b]. `play_match` draws independent boards per
  game, so it is not aliased, but pairing boards would cut variance.
- Replicate any "this term helps" result on a second seed block (Typhon's food
  result reversed) [src: 11].

---

## 5. Evidence from slinky

All numbers use a throwaway JAX prototype of section 6 (11x11 duel, standard
rules, seats alternated). "P0" is the feature-only one-ply snake, "P1" the
one-ply simultaneous maximin over the exact rules with the static evaluator
(section 6).

**Prototype checks.**

- The time-aware move test (`in bounds and tf[m] <= 1`) matched
  `rules.action_mask` on 233,954 snake-states with health > 1 (0 mismatches).
- The evaluator was bounded in [-1, 1] and exactly antisymmetric on all
  sampled states.
- Cost on a 4-core CPU: the evaluator runs at about 104,000 states/s when
  vmapped over 1024 states; P0 about 21,000 and P1 about 9,300 decisions/s
  (both computing both seats at once).

**Which terms matter (P0, variant vs default P0, 512 games each; below 0.5
means the removed term was helping).**

| Variant | Score of variant | Reading |
|---|---|---|
| no territory term | 0.299 ± 0.040 | most important |
| no food-weighted territory | 0.314 ± 0.039 | |
| no eat bonus (step onto adjacent food) | 0.326 ± 0.038 | |
| no hunger (starvation pressure) | 0.468 ± 0.041 | small |
| no space-vs-length term | 0.478 ± 0.041 | small |
| no "lose the head-to-head" tier | 0.472 ± 0.041 | small, but cheap safety |
| no "fits" (trap) tier | 0.469 ± 0.041 | small, but cheap safety |
| no equal-length-meeting tier | 0.595 ± 0.041 | removing it *helped* |
| eat bonus x4 | 0.536 ± 0.042 | |
| territory weight x3 | 0.382 ± 0.039 | too much |

A tuned P0 (no equal-length tier, eat bonus 2, food-territory weight 2,
territory weight 1.5) scored 0.696 ± 0.039 against default P0, 0.942 ± 0.019
vs the DQN and 0.987 vs `random_legal`, but traded head-to-heads so freely that
the mirror match was 83% draws in 28 turns.

**Static evaluator ablations inside P1 (vs default P0, 512 games unless
noted).** The coefficient rows use the fitted logit coefficients with no
contempt (reference 0.728); the fill-depth and time-awareness rows use the
final rounded coefficients of section 6.5 with contempt 0.3 (reference 0.745 ±
0.037). Sharpness is a multiplier on the fitted logit scale.

| Variant | Score | Reading |
|---|---|---|
| full (fitted coefficients) | 0.728 ± 0.037 | reference |
| territory coefficient 0 | 0.539 ± 0.042 | |
| food-territory coefficient 0 | 0.422 ± 0.042 | largest |
| length coefficient 0 | 0.638 ± 0.040 | |
| hunger coefficient 0 | 0.735 ± 0.037 | no effect |
| trap-shortfall coefficient 0 | 0.715 ± 0.037 | no effect |
| health-difference coefficient 0 | 0.713 ± 0.037 | no effect |
| bodies as permanent walls (no countdown), reference 0.745 | 0.513 ± 0.043 | time-awareness is worth 0.23 |
| fill depth K = 8 / 12 / 16 / 24 / 40, reference 0.745 | 0.428 / 0.634 / 0.750 / 0.745 / 0.746 | K >= 16 (about H + W) is enough |
| sharpness 0.5 / 1 / 2 / 4 / 8 (the last two with 256 games) | 0.696 / 0.728 / 0.730 / 0.676 / 0.422 | flat optimum at 1-2 |
| aggregation `min + 0.25 mean` vs `min` vs `min + 0.5 mean` | 0.728 / 0.710 / 0.694 | |

**Fitting the evaluator.** 46,514 states were sampled every third turn from
games of two P0 variants with 0-25% random moves, labelled with the final
result (win 1, draw 1/2, loss 0) and fitted by logistic regression on the
antisymmetric features. Single-feature AUCs: length difference 0.65,
territory 0.64, food-territory 0.63; all terms together 0.68 (the noisy
sampling policy caps this). Fitted logit coefficients in the open positions
(per unit feature): territory 0.51, food-territory 0.45, length 0.16, hunger
-0.76, shortfall -0.62. With the rounded evaluator of section 6.5, deciles of
predicted value against mean outcome (scaled to [-1, 1]) were essentially
monotone (two adjacent deciles differ by under 0.015 in the wrong order): from
v = -0.59 (outcome -0.43) up to v = +0.79 (outcome +0.59); regression slope of
outcome on v was 0.69 with offset 0.15 (the offset reflects that the sampled
seat-0 policy was stronger). Separated positions (no overlap between reach
sets) were almost absent: 7 of 46,514 states, because bodies unwind, so no
separate "separated" mode is specified.

**Final policy P1 (512 games each).**

| Contempt (value of a mutual-elimination draw is minus this) | vs `random_legal` | vs DQN | vs default P0 | mirror (P1 vs P1, 256 games) |
|---|---|---|---|---|
| 0 | 0.981 ± 0.008 | 0.949 ± 0.017 | 0.693 ± 0.038 | 95% draws, 13 turns |
| 0.25 | 0.997 ± 0.003 | 0.952 ± 0.018 | 0.752 ± 0.037 | not run |
| 0.3 | not run | not run | 0.745 ± 0.037 | 73% draws, 58 turns |
| 0.4 | 0.999 ± 0.002 | 0.954 ± 0.018 | 0.751 ± 0.037 | 46% draws, 103 turns |
| 0.5 | 1.000 | 0.960 ± 0.017 | 0.715 ± 0.039 | not run |
| 0.6 | 1.000 | 0.958 ± 0.017 | 0.675 ± 0.040 | 9% draws, 173 turns |
| 0.8 | 1.000 | 0.958 ± 0.017 | 0.649 ± 0.041 | not run |

Reference: default P0 scored 0.995 ± 0.006 vs `random_legal` and 0.914 ±
0.024 vs the DQN.

---

## 6. Recommended heuristic snake for slinky

Goal: a strong, simple, deterministic-up-to-random-tie-breaks, fully
vectorised heuristic for the 1v1 duel, plus a leaf evaluator for MCTS. Shapes
are fixed, no data-dependent control flow, `jit`/`vmap` friendly. All numbers
below were prototyped on 11x11 only.

### 6.1 Design in one paragraph

Score each of my four moves by a one-ply simultaneous-move search: for every
joint move `(a, b)` apply the exact `rules_step`, evaluate the successor with
the static evaluator, and take `min_b M[a,b] + 0.25 * mean_b M[a,b]` over the
opponent's non-suicidal replies `b`. Draws are valued slightly negative
(contempt). The evaluator is a bounded antisymmetric score of time-aware
Voronoi territory, food-weighted territory, length difference and two small
safety terms. The exact rules supply head-to-head, starvation and wall/body
death for free, so no hand-written tiers are needed in this policy.
A cheaper feature-only policy (P0) with explicit tiers is specified too; it is
what a one-ply scorer or MCTS prior can use without simulating.

### 6.2 Primitives (all on the countdown grid)

Notation: `tf[y,x] = max over alive snakes of body[i,y,x]` (int, 0 = empty);
`INF = 1000`; seat `i` is "me", `j = 1 - i`; `L`, `hp` are length and health.

**Time-aware arrival.** A cell with countdown `k` is enterable on the `d`-th
move from now iff `tf <= d` (the tail has been popped `d` times; stacked tail
has `k = 2`). Earliest-arrival steps by frontier expansion:

```python
def dilate4(m):                       # bool[H, W] -> 4-neighbours, no wrap
    p = jnp.pad(m, 1)
    return p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]

def arrival(tf, start, start_step, iters):   # start: bool[H, W]
    arr = jnp.where(start, start_step, INF)
    def body(k, carry):
        arr, frontier = carry
        d = start_step + k + 1
        nb = dilate4(frontier) & (arr == INF) & (tf <= d)
        return jnp.where(nb, d, arr), nb
    return jax.lax.fori_loop(0, iters, body, (arr, start))[0]
```

- `iters = H + W` (22 on 11x11). K >= 16 was equivalent in the prototype; 12
  and 8 were clearly worse.
- A cell blocked at its first-arrival time is not retried later (no waiting).
  That is a pessimistic simplification, and it matches the usual
  "delete tails each iteration" flood fill.
- Unreached cells keep `INF`. Wrapped rulesets: replace the pad-shift with
  `jnp.roll` (not prototyped).

**Arrival fields.** For the opponent: start at its head cell, `start_step = 0`
(computed once per state). For me after candidate move `a`: start at the
cell `m_a = head + delta[a]`, `start_step = 1` (empty start mask if the move is
illegal). Using one clock (steps from now) for both is what makes the
comparison a fair model of simultaneous movement.

**Claims (Voronoi).** For arrival fields `A` (me) and `B` (opponent):

```
claim_me[c] = (A[c] < B[c]) | ((A[c] == B[c]) & (A[c] < INF) & (L_me > L_op))
claim_op[c] = (B[c] < A[c]) | ((A[c] == B[c]) & (B[c] < INF) & (L_op > L_me))
```

Equal arrival and equal length: nobody claims the cell. Cells reached by only
one snake are its own.

### 6.3 Per-candidate-move features (policy P0, also usable as MCTS priors)

For each action `a` of snake `i`, with `m_a`, `A_a = arrival(tf, onehot(m_a), 1)`
and the opponent field `B`:

| Feature | Definition |
|---|---|
| `legal` | `m_a` in bounds and `tf[m_a] <= 1` (equals `rules.action_mask` apart from the "certainly starving snake's body" nuance), and not certain starvation: not (`hp - 1 <= 0` and no food on `m_a`) |
| `h2h_lose` | `B[m_a] == 1` (opponent can step there) and `L_op > L_me` |
| `h2h_draw` | `B[m_a] == 1` and `L_op == L_me` |
| `h2h_win` (opportunity) | `B[m_a] == 1` and `L_op < L_me` |
| `p_meet` | `[B[m_a] == 1] / max(1, #cells with B == 1)` (uniform opponent) |
| `space` | `#(A_a < INF)` |
| `fits` | my tail cell reachable (`A_a[cell with minimum positive body[i]] < INF`) or `space >= L_me + food[m_a]` |
| `terr` | `(#claim_me - #claim_op) / (H*W)` with `A_a` and `B` |
| `foodt` | `(#food in claim_me - #food in claim_op) / max(1, #food)` |
| `dfood` | `min` of `A_a` over food cells (INF if none reachable) |
| `hun` | see 6.5: `clip((30 - margin) / 30, 0, 1)^2`, `margin = hp - dfood` (or `min(hp, space) - 5` if no food reachable) |
| `eat` | `food[m_a]` |

Priority (lexicographic tiers, then a weighted sum):

```
dist    = dfood if a food item is reachable else 30
score_a = 1000*legal + 100*(not h2h_lose) + 50*fits
        + 1.5*terr + 2*foodt - 1.5*hun*dist/22
        + 2*eat*(1 + hun) + 0.25*min(space / L_me, 2)
```

Choose the argmax; add uniform noise of 1e-4 for random tie-breaks. Do **not**
add a tier against `h2h_draw` (it hurt, section 5). Opportunities
(`h2h_win`, a shorter opponent) are not scored separately here: a longer snake
already claims contested cells in `terr`, and food/space terms pull it toward
the opponent's area. If a dedicated aggression term is wanted, use
`+ w * p_meet * h2h_win` with `w` around 0.5, but this was not tested. Weights
are from one tuning pass (default P0: territory 3, food-territory 1, hunger
1.5, eat 0.5, space 0.5; tuned: territory 1.5, food-territory 2, eat 2, no
equal-length tier); expect flat optima.

### 6.4 Why the exact one-ply search is the main policy

The tiers above are a stand-in for what `rules_step` computes exactly: wall and
body deaths (including the tail rule), head-to-head outcomes by length,
starvation order, and the eating of food. Searching the 4x4 matrix with the
true rules removes the need for `h2h_*` features and makes "all opponent replies
lose" (a forced kill, value +1) visible. The cost is 16 `rules_step`s and 16
evaluations; the same 16 successors yield the values for both seats, so a policy
that must return actions for both snakes (as `play_match` expects) needs no
extra work.

### 6.5 Static evaluator (also the MCTS leaf value)

`evaluate(state) -> float32[2]`, value in `[-1, 1]` per snake, exactly
antisymmetric for non-terminal duel states (`v_1 = -v_0`).

Both snakes' arrival fields start at their own heads with `start_step = 0`;
`A_0, A_1`, `claim_0, claim_1` as in 6.2. Features are differences, snake 0
minus snake 1:

| Symbol | Definition |
|---|---|
| `T` | `(#claim_0 - #claim_1) / (H*W)` |
| `F` | `(#food in claim_0 - #food in claim_1) / max(1, #food)` |
| `dL` | `L_0 - L_1` (unclipped) |
| `hun_k` | `clip((30 - margin_k) / 30, 0, 1)^2`; `margin_k = hp_k - dfood_k` where `dfood_k = min A_k over food` if any food is reachable, else `min(hp_k, space_k) - 5`; `dHun = hun_0 - hun_1` |
| `short_k` | `clip(1 - space_k / L_k, 0, 1)` with `space_k = #(A_k < INF)`; `dShort = short_0 - short_1` |

```
z   = 0.40*T + 0.35*F + 0.15*dL - 0.50*dHun - 0.50*dShort
v_0 = clip(tanh(z), -0.99, 0.99)       v_1 = -v_0
```

Terminal states (`~alive`): one snake alive -> `+1` for it and `-1` for the
other; nobody alive -> `draw_value` for both (`0` for MCTS, `-0.4` inside P1;
this is the contempt). Expose `draw_value` as an argument. Stepping a done
state is a no-op, so `evaluate` of a done state must return the terminal
values rather than the heuristic. Truncated games with both alive fall through
to the heuristic.

Properties to unit-test: bounded; antisymmetric; swapping seats negates;
terminal values exact; invariant to dead snakes' body grids (use `alive` to
mask `body` before `tf`).

Calibration on sampled states (section 5): AUC about 0.67, deciles monotone,
slope 0.69. If an MCTS wants a calibrated value rather than a sharp one, scale
`z` by about 0.6; the one-ply policy preferred sharpness 1-2 relative to the
fitted logit scale (flat optimum).

Optional refinements (not measured): a larger hunger term when no food is
reachable; mixing in a separated-component term (a1k0n) if separated
positions turn out to matter; hazard cost (6.9).

### 6.6 Recommended policy P1 (one-ply simultaneous maximin)

```
for a in 0..3, b in 0..3:   s' = rules_step(state, (a, b), config)
                            M[a, b, :] = evaluate(s', draw_value=-0.4)       # both seats
mask = rules.action_mask(state, config)                                      # [2, 4]
for seat i, opponent j:
    score[i, a] = min_{b: mask[j, b]} M[a, b, i] + 0.25 * mean_{b: mask[j, b]} M[a, b, i]
    score[i, a] = -10 where not mask[i, a]
action[i] = argmax_a (score[i, a] + Uniform(0, 1e-4))
```

- Index the matrix by (own action, opponent action): seat 0 uses
  `M0[a, b] = M[a, b, 0]`, seat 1 uses `M1[a, b] = M[b, a, 1]`, where `M` is
  indexed `[action of seat 0, action of seat 1, seat]`.
- `rules_step` does not spawn food or advance the turn: that is fine for a
  one-ply evaluation (future food is ignored).
- Coefficients: `lam = 0.25` (min only: 0.710, `lam = 0.5`: 0.694, vs 0.728).
- Contempt: pick 0.3-0.4. It is the knob between "trades heads-on in a mirror
  match" (0) and "slightly more passive" (0.6+). The prototype table in
  section 5 shows scores vs the three opponents; 0.4 is a good default.
- Deterministic up to the 1e-4 tie-break noise. If an exact tie among legal
  moves should be uniform, replace the noise with `jax.random.categorical`
  over the tied maximisers.

### 6.7 Using it with simultaneous-move MCTS (the planned benchmark)

- **Selection:** Decoupled UCT: per node and per player, UCB1 (or
  UCB1-Tuned) over that player's four actions with separate statistics, then
  step with the joint action [src: 18, 19]. Rescale values to `[0, 1]` with
  `(v + 1) / 2` before UCB. Lanctot et al.'s best Tron configuration used
  UCB1-Tuned with tuned `C = 1.5` (for rewards in `[0,1]`) [src: 18]; Albatross's
  heuristic baseline used `c = sqrt(2)` [src: 6]. Tune on slinky.
- **Final move:** most-visited action ("max"); it beat sampling from the
  normalised visit counts ("mix") in Tron [src: 18].
- **Leaf value:** call `evaluate` at a new leaf instead of random play-outs
  (play-outs are unreliable once the snakes are apart [src: 14]). With
  `draw_value = 0` for the search itself. Cost [measured]: `evaluate` runs at
  about 104,000 states/s and `rules_step` at about 380,000 steps/s (README),
  both aggregate over a batch of 1024 on the 4-core CPU, so one iteration is
  about 12 us of aggregate throughput. Arithmetic from that: 256 iterations per
  decision for 1024 parallel games is about 3 s per turn for the whole batch,
  or roughly 8 minutes for a 150-turn batch. Size the iteration grid and
  batch accordingly.
- **Priors (optional):** softmax of the P0 `score_a` terms, or none (DUCT with
  no prior is what the Albatross baseline did [src: 6]).
- **Benchmark protocol:** MCTS with `N` iterations in {16, 64, 256, 1024, 4096}
  vs (a) `random_legal`, (b) the DQN, (c) P1, (d) MCTS with a different
  `N`; report score ± CI from `play_match`; for variance reduction replay
  each board with seats swapped [src: 11b]. Expect P1 as the scale for "how
  many iterations does MCTS need to match a one-ply heuristic".

### 6.8 Implementation notes

- Module suggestion: `src/slinky/heuristics.py` with `arrival`, `claims`,
  `evaluate`, `move_scores` (P0), `lookahead_policy` (P1); build policies once
  (`functools.lru_cache` on config) so `play_match`'s jit cache is hit.
- Everything is `[H, W]` boolean/int ops; the only loop is the fixed-length
  fill, which vmaps cleanly. Per state: 2 fills for `evaluate`; per P1
  decision: 16 steps + 16 `evaluate`s. On the 4-core CPU the prototype's P1 is
  about 9,300 decisions/s for a batch of 1024 games.
- Policies must return actions for all snakes, but only the seat's entry is
  used; computing both seats from one 4x4 matrix is free.
- Dead snakes: mask `body` by `alive` before `max`. In done states return
  terminal values.
- `rules_step` raises for the royale ruleset, so P1 supports only the rulesets
  slinky supports. Only `standard` was prototyped.

### 6.9 Limits and extensions (not validated)

- **Boards other than 11x11, and more than two snakes.** Only 11x11 duels were
  tested. For N > 2, compute one arrival field per snake and give each cell to
  the earliest arrival (ties to the strictly longest), use `value_i = tanh(z_i)`
  with `z_i` built from `i`'s features minus the mean (or max) over
  opponents, and consider restricting the matrix search to the nearest
  opponent (Typhon uses a full search for the nearest 2 and a cheap greedy
  move for the rest [src: 11]). Antisymmetry no longer holds.
- **Hazards.** For a hazard cell, charge `1 + damage * layers` health in the
  hunger margin (zero damage if food is on the cell), keep it passable, and
  treat a move whose resulting health would be <= 0 without food as illegal.
  `rules_step` already applies the damage exactly inside P1.
- **Wrapped boards:** neighbours wrap; use `roll` in `dilate4` and no edge
  penalty.
- **Not modelled by `arrival`:** an opponent that eats extends its body by one
  step (delays cell release by one); opponent food it is about to eat is not
  anticipated; snakes cannot "wait".
- **Known failure mode of this family:** deep traps beyond what one-ply +
  reachability sees. In 512 default-P0 mirror games, 140 of the 262
  eliminations of seat 0 were self-collision (69), starvation (45) or a wall
  hit when no legal move remained (26); another 98 were head-to-heads. Depth,
  not more terms, is the likely fix: this is what the MCTS is for. (The parity
  bound and a starvation tier did not help one-ply P0, section 5.)
- **Untested but plausible gains:** a dedicated aggression term; a separated-
  component mode; two-ply search over the three or four best moves; sampling
  from the equilibrium of the 4x4 matrix instead of maximin (cost: loses
  determinism).

---

## 7. Sources

All opened while writing this note unless marked.

1. Battlesnake docs, "Useful Algorithms" (flood fill, A*).
   https://docs.battlesnake.com/guides/useful-algorithms (local copy:
   `docs/battlesnake/official/guides/06-useful-algorithms.md`).
2. coreyja, battlesnake-rs (Rust): Amphibious Arthur, Devious Devin (paranoid
   minimax), Hovering Hobbs (minimax + flood fill), Improbable Irene (MCTS,
   UCB1-normal), a jump-flooding area-control snake, Eremetic Eric (tail
   chaser), Famished Frank, Gigantic George.
   https://github.com/coreyja/battlesnake-rs and the descriptive catalogue
   https://terrarium.coreyja.com/
   - 2b. "Minimax in Battlesnake" (paranoid vs MaxN, turn ordering, pruning):
     https://coreyja.com/posts/BattlesnakeMinimax/Minimax%20in%20Battlesnake/
3. M. B. Schier, N. Wustenbecker, "Adversarial N-player Search using Locality
   for the Game of Battlesnake", SKILL 2019 (Lecture Notes in Informatics S-15,
   pp. 109-120). https://dl.gi.de/items/43c71d90-3b16-4a1c-93c2-c1a98104a283/full
4. Battlesnake docs, Royale map (hazard damage 14, food in hazards): read from
   the local copy `docs/battlesnake/official/maps/02-royale.md`; the web URL
   https://docs.battlesnake.com/maps/royale appeared in search results only.
5. a1k0n, "Google AI Challenge post-mortem" (Tron: Voronoi, alpha-beta,
   chambers, "a better evaluation heuristic will always beat deeper minimax
   searches"). https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html
6. Y. Mahlau, F. Schubert, B. Rosenhahn, "Mastering Zero-Shot Interactions in
   Cooperative and Competitive Simultaneous Games" (Albatross), arXiv
   2402.03136 (2024). https://arxiv.org/abs/2402.03136 (PDF read for the
   Battlesnake baseline in Appendix C.1 and Figure 9).
7. Redbrick, "Building the Bounty Snake: a post-mortem" (2017).
   https://rdbrck.com/news/building-bounty-snake-post-mortem/
8. Redbrick, "Son of Robosnake: an Aggressive Bounty Snake" (2018/2019).
   https://rdbrck.com/news/son-robosnake-aggressive-bounty-snake/
9. ycheng11065, Orion's Fang (2nd place, RBC Spring 2023): paranoid minimax,
   alpha-beta, depth 7, flood fill and edge control.
   https://github.com/ycheng11065/2023-python-snake
10. calvinl4, battlesnake-winter-2022 (1st place, RBC): minimax with
    alpha-beta, 5 moves ahead. https://github.com/calvinl4/battlesnake-winter-2022
11. n0nuser, Typhon (Go): iterative-deepening alpha-beta with a transposition
    table, three evaluation terms, measured term ablations.
    https://github.com/n0nuser/typhon and its measurement record
    https://github.com/n0nuser/typhon/blob/main/BENCHMARK.md
    - 11b. Finding 015, seat/seed aliasing in benchmarks:
      https://github.com/n0nuser/typhon/blob/main/docs/findings/015-the-board-was-choosing-the-winner.md
12. FusionStreak, FusionSnake (Rust): iterative-deepening paranoid alpha-beta
    with a Voronoi-based evaluation.
    https://github.com/FusionStreak/FusionSnake and
    https://github.com/FusionStreak/FusionSnake/blob/main/src/evaluation.rs
    (`params.json` returned 404; numeric defaults not retrieved).
13. wrenger, snork (Rust/Python): configurable "Tree" (minimax) and "Flood"
    agents tuned with Bayesian optimization; 1st/2nd places in several arenas.
    https://github.com/wrenger/snork
14. N. G. P. Den Teuling, "Monte-Carlo Tree Search for the Simultaneous Move
    Game Tron" (2011). https://project.dke.maastrichtuniversity.nl/games/files/bsc/Denteuling-paper.pdf
15. Nettogrof, Battlesnake challenge tips (tail chasing, food use).
    https://github.com/Nettogrof/Battlesnake-Nessegrev-julia/blob/master/Challenge-Tips%26Tricks.md
16. "Battlesnake: The Rusty Tapeworm Chronicles" (food > own tail > enemy tail).
    https://sharkpillow.com/post/battlesnake/
17. Sandworm write-up (minimax over Voronoi; ladder ranks).
    https://emilien.ca/Sandworm/WRITEUP.md (the linked README and `move.c`
    returned 403, so the evaluation details are not known).
18. M. Lanctot, C. Wittlinger, M. H. M. Winands, N. G. P. Den Teuling, "Monte
    Carlo Tree Search for Simultaneous Move Games: A Case Study in the Game
    of Tron", BNAIC 2013.
    https://dke.maastrichtuniversity.nl/m.winands/documents/sm-tron-bnaic2013.pdf
19. M. J. W. Tak, M. Lanctot, M. H. M. Winands, "Monte Carlo Tree Search
    variants for simultaneous move games", IEEE CIG 2014.
    https://cris.maastrichtuniversity.nl/portal/en/publications/monte-carlo-tree-search-variants-for-simultaneous-move-games(5e4c8118-4db7-49bf-b12d-d16ee90fe167).html
20. V. Lisy, V. Kovarik, M. Lanctot, B. Bosansky, "Convergence of Monte Carlo
    Tree Search in Simultaneous Move Games", NIPS 2013. https://arxiv.org/abs/1310.8613

Also consulted, for context: the awesome-battlesnake index
(https://github.com/BattlesnakeOfficial/awesome-battlesnake), Kyle Poole's
tournament retrospective (https://www.kylepoole.me/blog/20191227_battlesnake_retrospective/),
TheApX's Caterpillar backstory
(https://github.com/TheApX/battlesnake-hungry/blob/main/docs/backstory.md),
and Bosansky et al., "Algorithms for computing strategies in two-player
simultaneous move games", Artificial Intelligence 237 (2016), pp. 1-40, cited
by the Albatross paper for simultaneous-move MCTS (its reference list was read;
the catalogue page
https://cris.maastrichtuniversity.nl/en/publications/algorithms-for-computing-strategies-in-two-player-simultaneous-mo
appeared in search results only).

Listed in awesome-battlesnake but not retrievable here (not relied on): Graeme
Hill's 2018 post-mortem (host did not resolve), the Medium posts (HTTP 403).
