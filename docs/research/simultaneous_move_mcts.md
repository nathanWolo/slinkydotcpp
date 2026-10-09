# Monte Carlo Tree Search for simultaneous-move games

A survey for building a simple, sound simultaneous-move MCTS (SM-MCTS) agent for
the slinky 1v1 duel, to benchmark against `random_legal` and the DQN baseline as
a function of the number of search iterations. It covers the variants, their
formulas and constants, how final moves and values are handled, what the Tron
literature says about playouts and evaluation functions, head-to-head results,
theory, and pitfalls. Recommendations for slinky are in the
[last section](#recommendations-for-slinky).

**Sources.** Every paper in the [references](#references) was opened and read,
except two:

- [Perick12]: only the abstract was read (via OpenAlex), because the PDF host
  refused connections.
- [Teytaud11]: only the bibliographic record was checked (via Crossref). Its
  claims are reported as later papers describe them.

Numbers quoted below are copied from the papers' tables. "A vs B 60%" means A
scores 60% against B, with draws counting ½.

## 1. The setting: stacked matrix games

A two-player game with simultaneous moves and otherwise perfect information is
called a *stacked matrix game*: in each state both players pick an action at the
same time, and each joint action `(a1, a2)` leads to a terminal state or to
another such game [Lisy13, Bosansky16]. The duel fits this model:

- **Players.** 2 snakes, each with up to 4 moves (usually 3, since the neck
  blocks one direction).
- **Zero-sum.** Win/draw/loss outcomes make the game zero-sum (constant-sum
  once rewards are scaled to `[0, 1]`).
- **Chance.** Food spawns are chance events.

There are two ways to build a search tree for such a game [Lanctot13Tron, Tak14]:

- **Sequential (serialized) tree.** The game is treated as turn-based inside
  the tree. One player moves, then the other replies *knowing* that move. Every
  Tron paper that does this lets the searching player move first. The opponent
  then best-responds to the searcher's move, so the searcher plays defensively
  (a "paranoid" view). In the minimax setting, the value of this serialized
  tree is a lower bound on the true value [Lanctot13Tron, Bosansky16].
- **Stacked matrix tree.** Each node stores a matrix of children indexed by
  joint actions. This is the faithful model: neither player can react to the
  other's current move [Lanctot13Tron].

In Tron, the simultaneous-move problem shows up when both players can reach
the same square. Battlesnake's version is the head-to-head collision.

- A sequential tree gets these cases wrong. In [DenTeuling12], the root player
  "moves first" into a contested square, which the opponent then cannot enter.
  This made the program play moves that lead to draws.
- The authors patched the expansion step: they add a draw terminal whenever
  the second mover could have entered the same square.
- A joint-action tree needs no such patch, because the real transition function
  resolves the collision.

## 2. The SM-MCTS template

All the variants share one loop [Lisy13 Alg. 1, Lanctot14Goof Alg. 1, Tak14
Alg. 1, Bosansky16 Alg. 5]. Each iteration starts at the root and does the
following:

1. **Terminal.** If the state is terminal, return its utility `u1` for player 1.
2. **Select.** If the node is in the tree, `Select(s)` returns a joint action
   `(a1, a2)`. Descend to the child `T(s, a1, a2)` and recurse.
3. **Update.** Call `Update(s, a1, a2, u1)` with the value returned by the
   recursion, then return a value to the parent. Usually this is the sample
   `u1`; see §8 for returning the mean instead.
4. **Expand.** If the state is not in the tree, add it. Estimate its value with
   a rollout (a playout to the end of the game) or an evaluation function, and
   return that value.

Two-player constant-sum games only need `u1`, because `u2 = k − u1` [Tak14].

When a node is expanded varies between papers:

- [Lanctot14Goof] and [Tak14] first try every unselected joint action at a node
  before running the selection policy.
- [Lanctot14Goof] notes that DUCT and Exp3 only need every *row and column*
  tried, which takes `|A1| + |A2|` visits instead of `|A1||A2|`. The decoupled
  methods need no joint statistics.
- [Bosansky16] runs `Select` at every in-tree node and adds the child when it
  is missing.

The method name gives the selection and update rule, and also the rule for
choosing the final move (§3).

## 3. Variants

### 3.1 Sequential UCT (SUCT)

SUCT runs ordinary UCT on the serialized tree. The searcher moves, then the
opponent responds knowing that move [Tak14, Lanctot13Tron].

- It was the first MCTS approach applied to Tron [Lanctot13Tron].
- It is easy to build and "learns to play safely": it picks the move whose
  worst best-reply is least bad [Lanctot13Tron].
- It cannot converge to a Nash equilibrium (NE) in general, because one player
  has an unrealistic information advantage [Lanctot14Goof, Bosansky16].
- In Tron, playouts are still simulated as simultaneous play [Lanctot13Tron].
- [Tak14] notes that the order of play could be reversed for aggressive play,
  or randomized.

### 3.2 Decoupled UCT (DUCT) and Decoupled UCB1-Tuned (DUCB1T)

Each player `i` keeps its own reward sum `X^i_{s,a}` and visit count `n^i_{s,a}`
for its *own* actions. Each player then picks a move with UCB1, independently
of the other player's choice [Tak14, Lanctot14Goof, Bosansky16]:

```
a_i = argmax_{a in A_i(s)}  X̄^i_{s,a} + C · sqrt( ln n_s / n^i_{s,a} ),   X̄^i_{s,a} = X^i_{s,a} / n^i_{s,a}
Update: X^i_{s,a_i} += u_i ;  n^i_{s,a_i} += 1        (n_s = Σ_a n^i_{s,a}, the node's visit count)
```

- DUCT first appeared in general game playing, in CadiaPlayer, which won the GGP
  competition in 2007–2009 and 2012 [Bosansky16, Lanctot14Goof].
- For rewards in `[0, 1]`, UCB1's own constant is `C = √2 ≈ 1.41` (Auer,
  Cesa-Bianchi & Fischer 2002, as cited by [Lanctot13Tron]).

**UCB1-Tuned.** UCB1-Tuned replaces `C` with a bound on each action's reward
variance [Lanctot13Tron Eq. 2]:

```
a = argmax_i  X̄_i + sqrt( min(1/4, V_i) · ln n_s / n_i ),   V_i = s̄²_i + sqrt( 2 ln n_s / n_i )
```

Here `s̄²_i` is the sample variance of action `i`'s rewards. The formula has
no tuned parameter, and needs only one extra sum of squared rewards per
(player, action).

**Final move.** There are two rules:

- **DUCT(max)** plays a single best move. [Lanctot13Tron] and [Bosansky16] take
  the move with the most visits. [Lanctot14Goof] and [Tak14] take the move with
  the highest mean `X̄`.
- **DUCT(mix)** samples a move from the normalized visit counts. [Bosansky16]
  uses only this mixed form: picking the most-visited move "certainly makes the
  algorithm not converge" to a Nash equilibrium, because the game may require a
  mixed strategy.

**Tie-breaking matters.** [Bosansky16, §4.4.1 and §6.3] studies how UCB ties
are broken:

- A deterministic rule, such as taking the first or last tied action, can make
  both players cycle in lock-step and never converge.
- The fix is to break ties at random among actions whose scores are within
  0.01 of the best.
- On Biased Rock-Paper-Scissors after 10⁸ iterations, this cut exploitability
  from 0.5–0.8 to 0.01–0.05. Exploitability is how much a best-responding
  opponent gains over the game value.
- The randomized version still does not provably converge.

### 3.3 Exp3

Each player samples its move from a softmax over importance-weighted reward
sums `x̂` [Lisy13 Eq. 4, Lanctot14Goof Eq. 3, Bosansky16 Eqs. 8–13]:

```
σ_i(a) = (1−γ) · exp(η w_a) / Σ_b exp(η w_b) + γ/|A_i|,   η = γ/|A_i|,   w_a = x̂_a − max_b x̂_b
Update: x̂_{a_i} += u_i / σ_i(a_i)
```

**Numerical stability.** Subtracting the maximum gives the same probabilities
and avoids overflow [Lisy13, Lanctot14Goof]. [Bosansky16, Eq. 9] uses the
equivalent form `(1−γ) / Σ_b exp(η(x̂_b − x̂_a)) + γ/|A|`, with rewards
normalized to `[0, 1]` first (Eq. 10).

**Final move.** The output strategy is the visit frequencies [Tak14,
Lanctot14Goof] or the average strategy [Bosansky16]. Either way, *the
exploration samples are removed first*:

- Visit counts: `n'_a = max(0, n_a − (γ/|A|) Σ_b n_b)` [Lanctot14Goof, Tak14].
- Average strategy: `σ̄(a) ← max(0, σ̄(a) − γ/|A|)`, then renormalize
  [Bosansky16].

The idea comes from [Teytaud11], who used Exp3 in UCT for the card game Urban
Rivals.

### 3.4 Regret matching (RM)

This variant was first used in [Lanctot14Goof] and, at the same time, in Tron
[Lanctot13Tron]. It applies regret matching
(Hart & Mas-Colell) to the matrix of estimated payoffs at each node
[Lisy13 §3.1, Tak14 §III-C, Bosansky16 §4.4.3].

**Statistics.** Each node stores:

- for each joint action, the visit count `n_{a1 a2}` and reward sum
  `X_{a1 a2}`, which give the estimated payoff matrix;
- for each player, cumulative regrets `r^i[a]`;
- for each player, the sum of the strategies played, `σ̄^i[a]`.

**Select.** Each player plays in proportion to its positive regrets:

```
σ_i(a) = r^i_+[a] / Σ_b r^i_+[b]   if that sum > 0, else uniform        (x_+ = max(x, 0))
```

The action is then sampled from `(1−γ) σ_i + γ · uniform`.

**Update.** Suppose joint action `(a1, a2)` returned `u1`. Let
`reward(b1, b2)` be the mean `X̄_{b1 b2}` for any other cell, and `u1` for the
cell `(a1, a2)` itself. Then:

```
∀ b1: r^1[b1] += reward(b1, a2) − u1
∀ b2: r^2[b2] += reward(a1, b2) − u2        (u2 = k − u1, rewards from player 2's view)
σ̄^i += σ_i        (the RM strategy, without the γ exploration)
```

[Bosansky16, Eq. 16] writes player 2's update with player 1's payoffs and a
minus sign. This is the same update.

**Final move.** Sample from the normalized average strategy σ̄ [Tak14,
Lanctot14Goof, Bosansky16]. [Lanctot13Tron, §6] suggests "purifying" the
output (zeroing low-probability actions) as future work.

**Relation to CFR.** This is close to Monte Carlo counterfactual regret
minimization (MCCFR), but the regrets are not counterfactual. Child means stand
in for the values of subtrees [Tak14].

### 3.5 Other variants

**SM-OOS (online outcome sampling).** This is an MCCFR-based search
[Lanctot14Goof, Bosansky16].

- It provably converges and gives the least exploitable strategies.
- In long games, few of its iterations update the root, and its regret updates
  have high variance.
- In 13×13 Tron with random simulations it was the weakest searcher:
  DOαβ beat it 78.2% and UCT 70.6% [Bosansky16 Table 4].

**Exact and depth-limited backward induction.** These methods solve the matrix
game at each node with a linear program, with alpha-beta pruning on serialized
bounds (BIαβ, DOαβ) [Bosansky16]. They are strong with a good evaluation
function, but they are not simple.

**Logit-equilibrium backup.** [Mahlau24] replaces MCTS in an AlphaZero-style
learner with a fixed-depth search that backs up a logit equilibrium. This needs
a learned value function, so it is out of scope for a heuristic benchmark.

**SM-MCTS-A.** This variant backs up the running mean instead of the sample;
§8 covers it.

### 3.6 Summary

| Variant | Stored per node | Selection | Final move | Converges to NE? |
|---|---|---|---|---|
| SUCT | standard UCT, on a serialized tree | UCB1 | most visits | No: the serialized model gives one player extra information |
| DUCT / DUCB1T | per player: `X^i[a]`, `n^i[a]` (+ `Σu²`) | per-player UCB1 / UCB1-Tuned, random tie-break | most visits (max) or visit distribution (mix) | No [Shafiei09]; randomized ties help a lot [Bosansky16] |
| Exp3 | per player: `x̂^i[a]`, `n^i[a]` | sample from softmax + γ | visit freqs or avg. strategy, exploration removed | Empirically yes; proven with extra conditions [Lisy13, Kovarik19] |
| RM | joint: `X[a1,a2]`, `n[a1,a2]`; per player: `r^i[a]`, `σ̄^i[a]` | sample from regret matching + γ | average strategy | Empirically yes; proven with extra conditions [Lisy13, Kovarik19] |

## 4. Constants used in the literature

| Paper | Domain | Value range | DUCT `C` | Exp3 `γ` | RM `γ` | Search budget |
|---|---|---|---|---|---|---|
| [Lanctot13Tron] | Tron 13×13, 4 boards | not stated | 1.5 tuned (reference values 10 and 3.52) | 0.3 (ref. 0.36) | 0.3 (ref. 0.025) | 100k simulations/move, ≈1 s |
| [Tak14] | 9 GGP games incl. Tron | rescaled so the two utilities sum to `k`; scale not stated | 1.4 reference; Tron tuned 1.8; SUCT Tron 2.0 | ref. 0.2; Tron 0.3 | ref. 0.025; Tron 0.3 | 2.5 s/move (2.2k–28k simulations/s) |
| [Lanctot14Goof] | Goofspiel(13) | win/loss {0, ½, 1}; point difference | 1.5 (win/loss), 150 (points) | 0.2 / 0.01 | 0.025 | 1 s/move, >100k simulations/s |
| [Bosansky16] | Tron 13×13, empty board | utilities ±1/0; Exp3 rescaled to [0, 1] | 0.6 (random rollouts), 2 (eval. function) | 0.5 / 0.1 (eval.) | 0.1 / 0.2 (eval.) | 1 s/move (≈11k UCT iterations/s) |
| [Mahlau24] | Battlesnake/Tron baseline | ±1 | `c = 2`, AlphaZero-style `sqrt(N)/n` bonus without a prior | — | — | varied (strength set by the iteration budget) |

Parameters are game-specific and must be tuned:

- [Tak14] finds that no single value works in every game, and that tuning is
  less smooth than in sequential games.
- For RM, `γ ∈ [0.2, 0.5]` "seems to be safe" across games [Tak14].
- [Bosansky16] finds the best values for RM and OOS stay within 0.1–0.3 in
  every domain. The best values for UCT and Exp3 vary widely: Exp3 wants 0.8 in
  Oshi-Zumo with an evaluation function but 0.1 in Tron with one.
- In [Bosansky16]'s offline runs, the best values were almost always 0.6
  (OOS), 0.1 (Exp3) and 0.1 (RM).

## 5. Value normalization

- **Use `[0, 1]`, constant-sum.** The theory assumes utilities in `[0, 1]` with
  `u2 = 1 − u1` [Lisy13, Kovarik19]. [Bosansky16, Eq. 10] rescales to the unit
  interval with `u1 ← (v1 − vmin)/(vmax − vmin)` and `u2 = 1 − u1`. [Tak14]
  rescales GGP payoffs so the two utilities always sum to the same constant
  `k`.
- **`C` depends on the scale.** The UCB constant only means something relative
  to the reward range: [Lanctot14Goof] needs `C = 1.5` for win/loss but
  `C = 150` for point-difference Goofspiel.
- **Exp3 is especially sensitive.** [Bosansky16] names "problematic
  normalization for wider ranges of payoffs" as a main weakness of Exp3.
- **Draws are worth ½.** Draws count as half a win in every head-to-head
  evaluation [Lanctot14Goof, Tak14, Bosansky16]. In the Tron searches, a draw
  is worth 0 on a ±1 scale, i.e. ½ on `[0, 1]` [DenTeuling12, Bosansky16].

## 6. Playouts, evaluation functions and cut-offs (the Tron evidence)

**Random playouts are viable, but get unreliable at a distance.**

- [DenTeuling12] found random playouts (excluding moves that lose at once)
  "surprisingly good", and the most robust of six playout policies across
  boards.
- The other policies were wall-following, offensive, defensive, mixed,
  move-category and ε-greedy. Each one's strength varied a lot from board to
  board. The offensive policy, for example, won only 4% on average against
  random [DenTeuling12 Table 4].
- But all the MCTS programs lost heavily to a1k0n, the winner of Google's 2010
  AI Challenge, an αβ searcher with a "tree of chambers" evaluation. The best
  MCTS variant scored only 18 ± 3% against it.
- The authors put this down to playouts whose "reliability … rapidly drops as
  the players get more distant from each other". They suggest replacing playouts
  with an evaluation function such as tree of chambers [DenTeuling12 §7.6, §8].

**Cut-offs help where they apply, and cost simulations.** In [DenTeuling12],
"play-out cut-off" stops a playout early once the players are separated and
predicts the winner by estimated space. The space estimate is evaluated only
every 5 moves.

- It cut throughput to 25k playouts/s.
- It won 54% and 56% on two boards but 33% on the third, 48 ± 2% overall
  against plain MCTS-UCT. On the third board players rarely become separated,
  so the heuristic was wasted time.
- "Predictive expansion" applies the same idea at expansion time: 53 ± 2%.
- An MCTS-Solver combined with both reached 61 ± 3% against MCTS-UCT.
- [Lanctot13Tron] used random playouts with cut-offs applied every 10 steps,
  plus predictive expansion.

**A good evaluation function in place of rollouts helps every variant.**
[Bosansky16 §5.2, §6.5.4] replaced `Rollout(s)` with `eval(s)` at new leaves.

- In Tron the evaluation is `tanh((owned1 − owned2)/5)`, where a cell is
  "owned" by the player that reaches it first. Ownership comes from a flood fill
  started from both heads.
- With random rollouts, DOαβ (which does use the evaluation function) beat
  every sampling algorithm. UCT scored 46.2% against it.
- With the evaluation function, UCT scored 57.3% against DOαβ and RM 53.7%. The
  gaps between the sampling methods shrank.
- In general: "Using a good evaluation function instead of random simulations
  helps all sampling algorithms, but the amount of improvement is different for
  individual algorithms in different domains."

**Battlesnake practice uses the same heuristic.**

- [Schier19] and [Mahlau24] measure *area control* with a flood fill from all
  heads. A cell reachable by two snakes at the same time goes to the longer one,
  following the head-to-head rule.
- [Mahlau24] also deletes tails as the fill advances, and adds a health term.
- [Mahlau24]'s DUCT baselines use this heuristic at the leaves, with no
  rollouts. Their strength is set only by the iteration budget.

## 7. Head-to-head results

**Tron, 4 boards, 100k simulations per move [Lanctot13Tron, Tables 2–3].**
Overall win rate against all other variants:

| Variant | Overall win rate |
|---|---|
| DUCB1T(max) | 62.3 ± 0.6% |
| DUCB1T(mix) | 54.8% |
| UCB1T (sequential) | 54.3% |
| RM | 53.1% |
| UCT (sequential) | 51.4% |
| DUCT(max) | 49.1% |
| DUCT(mix) | 39.5% |
| Exp3 | 35.5% |

Head to head, summed over boards:

- DUCB1T(max) vs DUCT(max): 61%.
- DUCT(max) vs DUCT(mix): 58%.
- DUCB1T(max) vs DUCB1T(mix): 56%.
- RM vs DUCT(max): 55%.
- Sequential UCT vs DUCT(max): 51%.

Other findings:

- Deterministic strategies generally beat stochastic ones. Unlike in
  Goofspiel, Exp3 did not beat DUCT(max), "possibly because mistakes caused by
  uncertainty in the final move selection are easy to recognize and exploit in
  Tron".
- On a small 6×6 board the gaps shrank and RM came first (52.2%). This is
  "possibly because the stochastic strategies are finding more situations where
  mixing is important".

**Tron, random playouts, 2.5 s per move [Tak14, Table II].**

- SUCT vs DUCT: 53.5 ± 2.4%.
- DUCT vs Exp3: 59.8%.
- DUCT vs RM: 78.0%.
- SUCT vs RM: 80.3%.
- DUCB1T vs DUCT: 52.9 ± 2.4%. In the other games DUCB1T and DUCT performed
  "almost equally".

Across all nine games: DUCT 68.3%, SUCT 63.4%, Exp3 38.8%, RM 29.5%. RM
dominates only where mixing matters: it beats DUCT 95.0% in Goofspiel, and also
wins in Oshi-Zumo. The results are intransitive: in Goofspiel, DUCT beats SUCT,
SUCT beats RM, and RM beats DUCT.

**Tron, empty 13×13 board, 1 s per move [Bosansky16, Table 4].**

- With random rollouts, UCT(0.6) beats OOS 70.6%, Exp3 64.8% and RM 57.0%, and
  scores 46.2% against DOαβ.
- With the evaluation function, UCT(2) scores 57.3% against DOαβ but loses to
  RM(0.2) (46.7%) and OOS (47.0%). It roughly ties Exp3 (49.7%).
- RM with the evaluation function beats UCT 53.3%, Exp3 54.2% and DOαβ 53.7%.
- The random player scores 1–3% against every search.

**Goofspiel(13) [Lanctot14Goof, Table 1].** Here mixing is essential:

- DUCT(max) lost to every other algorithm.
- RM and OOS were best: RM vs DUCT(max) 63.3%, RM vs DUCT(mix) 53.2%.

**Battlesnake.** No published head-to-head comparison of SM-MCTS variants uses
the official duel rules, as far as this survey found.

- [Mahlau24] trained AlphaZero with different search variants at 2,000 search
  iterations, against heuristic DUCT baselines (Appendix C).
- In their Tron mode, RM and SM-OOS trained best. In the stochastic 2-player
  mode, the logit-equilibrium backup was best by a wide margin. In the 4-player
  mode it was best, closely followed by DUCT.

## 8. Theory: convergence, exploitability, and when mixing matters

**DUCT does not converge to a Nash equilibrium.** [Shafiei09] gives the
counterexample of Biased Rock-Paper-Scissors.

- The unique NE is (0.0625, 0.625, 0.3125).
- UCT (with `C = 100` on payoffs in [0, 100]) settles into a "balanced"
  cycle whose visit frequencies are (⅓, ⅓, ⅓). That is not an equilibrium, and
  an NE player (CFR in their experiments) can exploit it.
- [Lanctot14Goof] confirms this in Goofspiel(4): DUCT's exploitability "starts
  to increase after 20000 iterations". Exp3, RM and OOS+ converge there.
- Random tie-breaking brings DUCT much closer to equilibrium, but gives no
  guarantee [Bosansky16].

**Regret-minimizing selection converges, under conditions.**

- [Lisy13] proves the following. Suppose the selection method is ε-Hannan
  consistent, i.e. its average regret is eventually at most ε, and it tries
  every joint action infinitely often. Then SM-MCTS converges to an approximate
  subgame-perfect NE.
- The proof assumes nodes back up the *mean* value ("RMM"). Empirically,
  backing up the sample converges slightly faster [Lisy13 §5.1].
- [Kovarik19] refines this. With the standard sample backup, Hannan consistency
  alone is *not* enough: they build a consistent selection rule under which
  SM-MCTS converges far from equilibrium.
- Adding averaging (SM-MCTS-A) restores the guarantee, but converges more
  slowly. An "unbiased payoff observations" property is also sufficient without
  averaging, and Exp3 and RM empirically have it.
- The guaranteed distance from equilibrium is `C·ε`. For game depth `D`, `C`
  is at most of order `D²·2^D`, and at least `2D` in the worst case
  [Kovarik19 Table 1].
- With a constant exploration rate γ, some games admit no ε-NE with `ε < γD`
  [Bosansky16 Thm. 4.5].

**Remove exploration from the output.**

- Mixing uniform exploration into the strategy and then averaging leaves `γ` of
  noise in the average strategy σ̄.
- [Kovarik19 §6] removes it: `μ̄ = (σ̄ − γ·uniform)/(1 − γ)`. In a matrix game this
  halves the exploitability bound from 2γ to γ (Prop. 6.2).
- Empirically the gain is "large", and the authors recommend that "the
  exploration should always be removed".

**When mixing matters.** Optimal Tron play is mostly pure:

- In [Bosansky16]'s count of 5×6 Tron states, mixed equilibria appear only in
  mid-game contests. At depth 8, 106 states need mixing against 54,304 that do
  not, and none do from depth 11 on.
- This is their explanation for UCT doing well in Tron: it "is able to quickly
  disregard other actions, if a single action is optimal."
- Mixing is essential in Goofspiel and Oshi-Zumo, where RM and OOS win.
- Battlesnake duels resemble Tron, with one extra mixing hot-spot: head-to-head
  stand-offs. When a head-to-head is fatal to both or to the shorter snake,
  whether to commit or retreat is a matching-pennies-like choice (our reading,
  not a result from the literature).

## 9. Battlesnake and Tron bots in practice

**Top hand-written bots search sequentially, with area heuristics.**

- a1k0n's Google AI Challenge 2010 winner used minimax with alpha-beta,
  depth-limited by time. Its evaluation grew from a Voronoi territory count to
  articulation points and "chamber trees" [a1k0n10]. The post-mortem does not
  say how simultaneous moves were handled.
- [Schier19] (2nd place, Battlesnake Victoria, intermediate division) works as
  follows:
  - It turns off food spawning inside the search, which makes the search
    deterministic.
  - It applies moves one snake at a time ("delayed move execution"), so the
    tree is sequential.
  - It runs maxⁿ (each snake maximizes its own score) for more than 2 snakes,
    and alpha-beta for 2. Snakes too far away to interact are masked out.
  - Its heuristic combines length advantage, flood-fill area control (contested
    cells go to the longer snake), and health relative to the distance to food.
    The weights were tuned by a genetic algorithm.
- [coreyja22] describes the same sequential tree with paranoid minimax and
  maxⁿ, plus flood-fill area control.

**MCTS in Battlesnake.**

- [Mahlau24] publishes a Battlesnake implementation. Its baseline agents use
  SM-MCTS with DUCT (`c = 2`, no policy prior) and the area-control/health
  heuristic at the leaves, with no rollouts. Agent strength is varied only
  through the iteration budget, exactly the benchmark wanted here.
- [Archinuk23] uses a single argmax over all 81 joint actions of 4 snakes,
  maximizing a combined score `Q + U`:
  - `Q` comes from per-snake reward summaries.
  - `U` is a PUCT bonus whose prior is the product of each snake's network
    probabilities.
  - Each snake's move is then sampled from its own visit distribution.
  - This differs from DUCT, where each player maximizes its own bandit
    separately.
- The open-source *nicanelo-snake* engine [nicanelo] runs PUCT for its own
  snake only. It samples rival moves from a softmax opponent model, which is a
  modeled opponent rather than a game-theoretic one.

**JAX tooling.** [mctx] is DeepMind's batched JAX MCTS library for
AlphaZero/MuZero. It has no simultaneous moves. Its fixed-capacity tree layout
is a useful template:

- Arrays such as `node_visits[B, N]`, `parents[B, N]` and
  `children_index[B, N, num_actions]`.
- One new node per simulation, and parent pointers for the backup.

## 10. Practical pitfalls

- **Deterministic tie-breaking in DUCT.** Use random tie-breaking [Bosansky16].
  Unvisited actions should get a large finite priority plus noise, not `+inf`.
  Otherwise the ties between unvisited actions are broken in index order.
- **Reward scale against `C` and γ.** Normalize to `[0, 1]` first (§5). Exp3
  degrades with wide payoff ranges [Bosansky16].
- **Exp3 overflow.** Use the max-subtracted or ratio form [Lisy13,
  Bosansky16].
- **Wrong perspective on updates.** Each player's statistics must use *its
  own* utility (`u2 = 1 − u1`). In RM, player 2's regret uses
  `reward(a1, b2)` with player 1's action fixed, and the sign must be flipped
  [Tak14, Bosansky16 Eq. 16].
- **Exploration left in the output.** Remove it (§8). For RM, average the
  pure regret-matching strategy σ, not the γ-mixed sampling distribution
  [Lanctot14Goof, Kovarik19].
- **Overfitting by self-play tuning.**
  - Results are intransitive, and the best parameters depend on the opponent.
  - [Tak14] advises "systematic testing against several different variants".
  - Gaps between parameter settings are larger against weaker opponents and
    shrink with more search time [Bosansky16 §6.5.6–6.5.7].
- **Expensive heuristics cost iterations.** In Tron, cut-off heuristics lost on
  a board where they rarely applied [DenTeuling12]. Keep the leaf evaluation
  cheap, or measure the trade-off.
- **Sequential trees and collisions.** A serialized tree mishandles
  contested-square and head-to-head moves unless patched [DenTeuling12]. It also
  plays defensively [Tak14].
- **Reusing subtrees** between turns is safe in simultaneous-move games
  [Bosansky16 §5.2]. It is an optional optimization.

## Recommendations for slinky

1. **Default: DUCT on the joint-action (stacked matrix) tree.** Use
   per-player UCB1, values in `[0, 1]` (`u1 = 1 − u0`), `C = 1.4`, random
   tie-breaking within 0.01, and the most-visited final move ("max").
   - Rationale: DUCT is the most robust variant across nine games [Tak14] and
     strong in Tron [Lanctot13Tron, Bosansky16]. It is the simplest, and it
     extends to N players by giving each player its own bandit.
   - Expose a `ucb1_tuned` flag. It costs one extra array, has no tuned
     parameter, and was the best Tron variant in [Lanctot13Tron] and
     [Perick12].
2. **Optional alternative: SM-MCTS with regret matching.** Use `γ = 0.2`, the
   sample backup, and a final move sampled from the average strategy
   (`argmax` as a purified option).
   - It is the sound choice: it empirically converges to a Nash equilibrium
     [Lisy13, Kovarik19], and with an evaluation function it was one of the two
     strongest sampling methods in Tron, with OOS [Bosansky16].
   - It is a cheap fixed-size addition: a 4×4 joint table per node.
   - Expect it to lose to DUCT if leaves are evaluated with random rollouts
     [Tak14, Bosansky16].
3. **Leaf evaluation: a heuristic, not rollouts.** Use a time-aware
   area-control flood fill (§6), mapped as `v0 = ½ + ½·tanh(Δ/τ)` with `τ = 5`
   [Bosansky16]. Terminal states get exact values: win 1, loss 0,
   mutual elimination ½.
   - Make the number of random-legal rollout steps a parameter (default 0) for
     ablations.
4. **Transitions inside the tree.** Use the real rules (`env.step` on an
   `obs=None` env), sampling the food spawn once when a node is expanded and
   storing the child state. This keeps every head-to-head and body collision
   exact.
5. **Benchmark** iteration budgets 1, 16, 32, 64, … 1024, against
   `random_legal`, the DQN, and the same agent at half the budget. Also run the
   ablations: max/mix, UCB1-Tuned, RM, and rollout leaves. The concrete spec,
   with formulas, data layout and expected trends, is below.

### Implementation spec (duel; written to generalize to N players)

**Data layout.** One tree per game, `vmap`ped over games by `play_match`.

- **Capacity.** `M = num_simulations + 1` nodes. The root is node 0, and each
  simulation adds at most one node.
- **Per node:**
  - `state` (the `State` pytree stacked to `[M, ...]`);
  - `parent[M]` and `parent_action[M, P]`;
  - `children[M, A**P]` (int32, −1 means unexpanded; the joint index is
    `a0 + 4·a1`);
  - `legal[M, P, A]`;
  - `terminal[M]`;
  - `value[M, P]` (the raw leaf or terminal value).
- **DUCT statistics:** `N[M, P, A]` (int32), `W[M, P, A]` (float32), and
  optionally `W2[M, P, A]` for UCB1-Tuned. The node's visit count is
  `n_s = Σ_a N[s, 0, a]`.
- **RM statistics:** `Nj[M, A, A]`, `Xj[M, A, A]` (sums of `v0`),
  `R[M, P, A]`, `S[M, P, A]`.
- **Loops.** `lax.fori_loop` over simulations. Selection is a `lax.while_loop`
  that descends while the node is non-terminal and the chosen child exists.
  Backup is a `lax.while_loop` up the parent pointers.
- **Memory.** About 1 KB per node, so roughly 1 MB per game at 1024
  simulations. Keep `batch_size` around 32 at 2048 simulations or more.

**Expansion.** When the selected joint action `(a0, a1)` has no child, create
it:

```
child = env.step(fold_in(key, node_id), parent.state, joint_action)   # env = BattlesnakeEnv(config, obs=None)
terminal = child.done
value = terminal_value(child) if terminal else leaf_eval(child)       # float32[P] in [0, 1]
legal = env.action_mask(child)
```

- **Food spawns.** The spawn is sampled once, at expansion, and the child is
  stored, so the tree stays consistent.
- **Deterministic alternative.** Skip the spawn by calling `rules.rules_step`,
  incrementing the turn and recomputing `done`. This removes chance from the
  tree, as [Schier19] does, but leaves the search blind to new food.
- **`terminal_value` (duel).** If exactly one snake is alive, it gets 1 and
  the other 0. If none are, both get ½.
- **Optional pruning.** Collapse a doomed player's row to a single action. A
  doomed player is one with `certainly_starving` set, or no move that is safe
  under the mask. Note that `action_mask` also returns an all-True row when all
  four moves are genuinely safe, e.g. a stacked snake on turn 0. For N > 2, a
  dead player always gets a single action.

**DUCT selection.** Done independently for each player `p` at node `s`:

```
q     = W / N
bonus = C · sqrt(log(n_s) / N)                                            # UCB1, C = 1.4
      | sqrt(log(n_s)/N · min(0.25, W2/N − q² + sqrt(2·log(n_s)/N)))      # UCB1-Tuned option
score = where(~legal, −inf, where(N == 0, BIG + U(0,1), q + bonus + 0.01·U(0,1)))
a_p   = argmax(score)
```

Then look up `children[s, a0 + 4·a1]`. Descend if the child exists; otherwise
expand it.

**DUCT backup.** Use the leaf or terminal value vector `v` (with `v1 = 1 − v0`
in the duel). At each ancestor `s`, reached by joint action `a`, and for each
player `p`:

```
N[s, p, a_p] += 1;   W[s, p, a_p] += v_p;   W2[s, p, a_p] += v_p²
```

Back up the sample `v`, not the mean [Lisy13, Kovarik19].

**DUCT final move.** Done for each player `p`:

- **Default:** `argmax_a N[0, p, a]` over the legal moves, breaking ties by
  `W/N`.
- **`final="mix"`:** sample in proportion to `N[0, p, :]`.

One search yields a move for every seat, so the policy returns `int32[P]`.

**RM selection.** At node `s`:

- **Initial fill.** While some legal joint cell has `Nj == 0` (at most 16,
  usually 9 or fewer), pick one of them uniformly.
- **Otherwise:**

  ```
  σ_p = R⁺[s, p] / Σ R⁺[s, p]   (uniform over legal if Σ R⁺ = 0)
  a_p ~ (1 − γ)·σ_p + γ·uniform(legal_p),   γ = 0.2
  ```

**RM backup.** At node `s`, with joint action `(a0, a1)` and value `v0`:

```
σ_p from R[s] (unchanged since this simulation's selection);   S[s, p] += σ_p
Xj[a0, a1] += v0;   Nj[a0, a1] += 1
Q0[b] = v0 if b == a0 else Xj[b, a1] / Nj[b, a1];          R[s, 0, b] += legal0[b] · (Q0[b] − v0)
Q1[b] = 1 − v0 if b == a1 else 1 − Xj[a0, b] / Nj[a0, b];  R[s, 1, b] += legal1[b] · (Q1[b] − (1 − v0))
```

Guard `Nj == 0` by using `v0` (or `1 − v0` for player 1). After the initial
fill, every legal cell has been visited.

**RM final move.** `π_p = S[0, p] / Σ S[0, p]`. This needs no exploration
removal, because `S` accumulates σ rather than the γ-mixed distribution. By
default, sample from `π_p`; `final="argmax"` gives the purified move.

**Leaf evaluation.** Use a time-aware simultaneous flood fill (area control),
with a fixed number of steps `D = H + W` in a `fori_loop`. Two facts make it
work:

- A body cell with countdown `k` is free from move `k` onward, assuming no
  eating.
- Food reached on move `d` saves a snake with health `h` if `d ≤ h`, because
  feeding happens before the starvation elimination.

```
F_i = one_hot(head_i);  claimed = F_0 | F_1;  A_i = 0;  fd_i = ∞
for d in 1..D:
    free  = on_board & ~any_i(alive_i & body_i > d) & ~claimed
    n_i   = dilate4(F_i) & free                          # pad-and-shift, as in maps.spawn_mask
    both  = n_0 & n_1
    F_i   = (n_i & ~both) | (both & len_i > len_j)        # contested cells go to the longer snake; ties to nobody
    claimed |= n_0 | n_1
    A_i  += popcount(F_i);   fd_i = d if fd_i == ∞ and any(F_i & food) else fd_i
A_i = 0 where fd_i > health_i                             # starving
Δ   = (A_0 − A_1) + λ_len · (len_0 − len_1)               # λ_len = 1 (untuned)
v0  = ½ + ½·tanh(Δ / τ),  τ = 5 cells  [Bosansky16];   v1 = 1 − v0
```

`tanh` never reaches ±1, so the exact terminal values (0, ½, 1) always dominate
heuristic ones. As an ablation, `rollout_steps = r` plays `r` `random_legal`
steps before evaluating, and uses the terminal value if the game ends on the
way.

**Terminal and eliminated players.**

- Terminal nodes are never expanded. When selection reaches one, its exact
  value is backed up and the visit counted.
- In the duel, every non-terminal node has both snakes alive, since the game
  ends when one or fewer remain.
- A finished root (`play_match` keeps calling the policy for games that are
  already over) must not produce NaNs: guard every division.

**Budgets and benchmark.**

- **Throughput first.** Measure simulations per second on about 8 games.
- **Budgets.** `num_simulations` in {1, 16, 32, 64, 128, 256, 512, 1024}, plus
  2048 if throughput allows.
- **Opponents.**
  - `random_legal`;
  - the DQN, played greedily;
  - the same agent at half the budget (a ladder);
  - a 1-ply greedy area-control snake (the leaf heuristic used directly), if
    one is built.
- **Match settings.**
  - Set `max_turns` (e.g. 500), and report `score ± CI` and `mean_turns`.
  - Use 256–1000 games at small budgets and 64–128 at 512 or more. Keep every
    command under 10 minutes.
  - Build each policy once (`functools.lru_cache`), so `play_match` compiles
    once.
- **Ablations** at a middle budget (128–256):
  - `C` in {0.5, 1.0, 1.4, 2.0}, and UCB1-Tuned;
  - `final` set to max or mix;
  - tie-break noise turned off;
  - RM with `γ` in {0.1, 0.2, 0.3};
  - `rollout_steps` in {0, 10}.

**Expected trends (sanity checks).**

1. **`MCTS(1)` ≈ `random_legal`** (score ≈ 0.5). One simulation visits one
   random legal root action, and that becomes the move.
2. **Monotone against fixed opponents.** Against `random_legal` the score
   should not fall as the budget grows (within the CI), and should be near 1
   after a few dozen simulations. This is a hypothesis: the area heuristic alone
   should already beat a random mover. Against the DQN, record the budget where
   the score crosses 0.5. There is no prior for it.
3. **Ladder.**
   - `MCTS(2k)` vs `MCTS(k)` should score above 0.5 at every `k`, with
     shrinking margins.
   - Equal-budget self-play should score ≈ 0.5. This checks seat symmetry.
   - Mean game length in self-play should grow with the budget.
4. **Variant ordering seen in Tron.** At equal budget:
   - Heuristic leaves should beat random-rollout leaves clearly [Bosansky16,
     DenTeuling12].
   - "max" should be at least as good as "mix" (58% in [Lanctot13Tron]).
   - UCB1-Tuned should be at least as good as UCB1, by a small margin (52.9–61%
     in [Tak14, Lanctot13Tron]).
   - Random tie-breaking should be at least as good as deterministic.
   - RM vs DUCT with heuristic leaves should land around 45–55% (RM 53.3% in
     [Bosansky16]). DUCT should beat RM heavily with random rollouts (78% in
     [Tak14]).
   - A big RM loss *with heuristic leaves* points to a sign or perspective bug
     in the regret update.
5. **Tactics.** The agent should never play a masked move when an unmasked one
   exists. It should avoid head-to-heads with an equal or longer snake when an
   alternative of similar area exists. Test both on hand-built positions.

## Further reading

- [Becker25] backs up finite-depth regret-matching search with a learned value
  function. It is relevant if the heuristic leaf is later replaced by the DQN or
  another learned value.

## References

- **[Lanctot13Tron]** M. Lanctot, C. Wittlinger, M. H. M. Winands, N. G. P. Den
  Teuling. *Monte Carlo Tree Search for Simultaneous Move Games: A Case Study in
  the Game of Tron.* Proc. 25th Benelux Conference on Artificial Intelligence
  (BNAIC 2013), pp. 104–111.
  <https://dke.maastrichtuniversity.nl/m.winands/documents/sm-tron-bnaic2013.pdf>
- **[Tak14]** M. J. W. Tak, M. Lanctot, M. H. M. Winands. *Monte Carlo Tree
  Search Variants for Simultaneous Move Games.* IEEE Conference on Computational
  Intelligence and Games (CIG 2014), pp. 232–239. doi:10.1109/CIG.2014.6932889.
  <https://mlanctot.info/files/papers/cig14-smmctsggp.pdf>
- **[Lisy13]** V. Lisý, V. Kovařík, M. Lanctot, B. Bošanský. *Convergence of
  Monte Carlo Tree Search in Simultaneous Move Games.* NeurIPS 2013.
  arXiv:1310.8613. <https://arxiv.org/abs/1310.8613>
- **[Lanctot14Goof]** M. Lanctot, V. Lisý, M. H. M. Winands. *Monte Carlo Tree
  Search in Simultaneous Move Games with Applications to Goofspiel.* Computer
  Games Workshop at IJCAI 2013 (CGW 2013), CCIS vol. 408, Springer 2014,
  pp. 28–43. doi:10.1007/978-3-319-05428-5_3.
  <https://dke.maastrichtuniversity.nl/m.winands/documents/wcg13-smmcts.pdf>
- **[Bosansky16]** B. Bošanský, V. Lisý, M. Lanctot, J. Čermák, M. H. M.
  Winands. *Algorithms for computing strategies in two-player simultaneous move
  games.* Artificial Intelligence 237 (2016), pp. 1–40.
  doi:10.1016/j.artint.2016.03.005. Author PDF:
  <https://dke.maastrichtuniversity.nl/m.winands/documents/sm-journal.pdf>
- **[Kovarik19]** V. Kovařík, V. Lisý. *Analysis of Hannan consistent selection
  for Monte Carlo tree search in simultaneous move games.* Machine Learning 109
  (2020), pp. 1–50 (online 2019). doi:10.1007/s10994-019-05832-z.
  arXiv:1804.09045. <https://arxiv.org/abs/1804.09045>
- **[Shafiei09]** M. Shafiei, N. R. Sturtevant, J. Schaeffer. *Comparing UCT
  versus CFR in Simultaneous Games.* IJCAI 2009 Workshop on General Game Playing
  (GIGA'09), pp. 75–82. <https://webdocs.cs.ualberta.ca/~nathanst/papers/uctcfr.pdf>
- **[DenTeuling12]** N. G. P. Den Teuling, M. H. M. Winands. *Monte-Carlo Tree
  Search for the Simultaneous Move Game Tron.* Computer Games Workshop at ECAI
  2012, pp. 126–141.
  <https://dke.maastrichtuniversity.nl/m.winands/documents/Tronpaper.pdf>
- **[Perick12]** P. Perick, D. L. St-Pierre, F. Maes, D. Ernst. *Comparison of
  different selection strategies in Monte-Carlo Tree Search for the game of
  Tron.* IEEE CIG 2012, pp. 242–249. doi:10.1109/CIG.2012.6374162. Only the
  abstract was read; the PDF host
  (<https://people.montefiore.uliege.be/dlstpierre/publications/tron2012.pdf>)
  was unreachable.
- **[Teytaud11]** O. Teytaud, S. Flory. *Upper Confidence Trees with Short Term
  Partial Information.* Applications of Evolutionary Computation (EvoGames
  2011), LNCS 6624, pp. 153–162. doi:10.1007/978-3-642-20525-5_16. Not opened;
  cited as described by [Lanctot14Goof], [Bosansky16] and [Kovarik19].
- **[Mahlau24]** Y. Mahlau, F. Schubert, B. Rosenhahn. *Mastering Zero-Shot
  Interactions in Cooperative and Competitive Simultaneous Games.* ICML 2024.
  arXiv:2402.03136. <https://arxiv.org/abs/2402.03136>
- **[Schier19]** M. B. Schier, N. Wüstenbecker. *Adversarial N-player Search
  using Locality for the Game of Battlesnake.* SKILL 2019, Lecture Notes in
  Informatics S-15, pp. 109–120. <https://dl.gi.de/handle/20.500.12116/29001>
- **[Archinuk23]** F. Archinuk, D. Bell, L. McKee-Reid, E. Showers, N.
  Woloshyn. *Monte Carlo Tree Search and Reinforcement Learning for a Four
  Player, Simultaneous Move Game.* University of Victoria report (PDF dated
  March 2023).
  <https://uvicai.ca/assets/images/MCTS-and-RL-for-a-Four-Player-Simultaneous-Move-Game.pdf>
- **[a1k0n10]** A. Sloane (a1k0n). *Google AI Challenge post-mortem*, 2010-03-04.
  <https://www.a1k0n.net/2010/03/04/google-ai-postmortem.html>
- **[coreyja22]** coreyja. *Minimax in Battlesnake*, blog post, 2022-03-05.
  <https://coreyja.com/posts/BattlesnakeMinimax/Minimax%20in%20Battlesnake/>
- **[nicanelo]** *nicanelo-snake*, a TypeScript Battlesnake engine (GitHub).
  <https://github.com/ItsSam11/nicanelo-snake>
- **[mctx]** DeepMind. *Mctx: MCTS-in-JAX* (GitHub); tree layout in
  `mctx/_src/tree.py`. <https://github.com/google-deepmind/mctx>
- **[Becker25]** T. Becker, Z. Sunberg. *Simultaneous AlphaZero: Extending Tree
  Search to Markov Games.* arXiv:2512.12486. <https://arxiv.org/abs/2512.12486>
