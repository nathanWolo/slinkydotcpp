# Why the multi-mode generalist gains nothing overall in constrictor

The multi-mode PPO generalist (`baselines/ppo_modes.py`, results in
[`baselines/README.md`](../../baselines/README.md#multi-mode-ppo-ppo_modespy))
beats its single-mode specialists head to head in standard (0.81), the duel
(0.62–0.71 against three specialist seeds) and royale (0.70–0.71, greedy and
sampled). Its constrictor head-to-head first read 0.406, which looked like a
loss. This note is the investigation of that gap.

How to read this file:

- **[benchmark]** marks a number reproducible from the committed results:
  `benchmarks/results/modes.jsonl` (`python benchmarks/modes.py --table`) or a
  run's committed `metrics.jsonl`.
- **[side analysis]** marks a result from exploratory scripts written for this
  investigation, run with the final networks, the training snapshots and the
  trainer's own functions. Two independent reviews re-ran the key scripts and
  checked the numbers. Several claims were corrected or weakened in that
  review, and only the checked versions are given here.
- **The scripts and their printed outputs** are kept as a record in
  [`multimode_constrictor_analysis/`](multimode_constrictor_analysis/) (see
  its README). They are unmaintained and not linted, and many need the
  training snapshots (`params-<steps>.npz`) or intermediate arrays, which are
  not committed.
- Everything rests on one training seed per run, except the duel specialist
  (three seeds).

## 1. The 0.406 was 12 games

- **Constrictor games between two greedy `ppo_modes` networks are
  deterministic given the start.** In constrictor, food is inert and the
  `ppo_modes` networks see it blanked out, so nothing random reaches their
  moves. The standard map has two start layouts for four snakes: the corners
  and the edge midpoints. Holding two of the four seats, the generalist can
  sit in 6 ways per layout, which makes 12 distinct games. [side analysis]
- **The benchmark's 1,024 games are those 12 repeated.** The generalist
  wins 4, draws 2 and loses 6, a share of 5/12 = 0.417. The benchmark printed
  [0.376, 0.436], which only covers which of the 12 came up (their exact
  mean is inside it). As evidence about strength the 12 games are the whole
  sample, and a bootstrap over them gives [0.17, 0.67]. It loses both games
  where its two snakes start on opposite edge midpoints. [side analysis]
- **Every other cell is varied games.** The heuristic breaks exact ties at
  random, `random_legal` is random, and `ppo` (the duel checkpoint) sees the
  randomly spawned food. Counted directly, constrictor G,H,H,H gave 251
  distinct games in 256 and G,P,P,P 201. The head-to-heads in standard,
  royale and the duel gave 128 distinct games in 128, because food matters
  there. [side analysis]
- **The benchmark now plays a sampled head-to-head as well.** Both networks
  sample their moves (`Gs,Ss,Gs,Ss`). `--table` marks a constrictor line-up
  of greedy networks only as *deterministic*.

## 2. Without the repeats: level when sampled, slightly behind close to greedy

| measure | generalist's share | source |
|---|---|---|
| sampled play, G,S,G,S, 1,024 games (essentially all distinct) | 0.494 [0.463, 0.524] | [benchmark] |
| sampled play, 1,024 games on independent keys | 0.495 [0.466, 0.524] | [side analysis] |
| sampled play, 2,048 games (the first 1,024 are the benchmark's) | 0.488 [0.468, 0.508] | [side analysis] |
| 3 sampled opening moves, then greedy (631 distinct of 1,024) | 0.490 [0.461, 0.518] | [side analysis] |
| greedy with a random legal move 2% / 5% / 10% of the time (2,048 games each) | 0.417 [0.396, 0.437] / 0.438 [0.418, 0.458] / 0.461 [0.440, 0.481] | [side analysis] |
| against 3 heuristics, G − S per seat (paired, 1,024 games) | −0.012 ± 0.032 | [benchmark] |
| against 3 `ppo`, G − S per seat (paired, 1,024 games) | +0.056 ± 0.039 | [benchmark] |
| against 1 heuristic and 2 `random_legal`, G − S per seat (paired, 1,024 games) | −0.046 ± 0.038 | [benchmark] |

- **No overall gain.** The generalist is level with its specialist under
  sampled play, with 3 sampled opening moves and against three heuristics.
  It is slightly ahead against `ppo` and slightly behind with one heuristic
  and two `random_legal`.
- **A small deficit close to greedy play.** With occasional uniform random
  moves it is 0.04–0.08 behind. The three eps runs reuse the same random
  draws, so they are not independent confirmations.
  - With a single sampled opening move, then greedy play, it scores 0.446
    [0.417, 0.474] per game. Those are only 189 distinct games, and over the
    distinct games it is 0.492 [0.421, 0.561]. [side analysis]
- **Where that deficit sits** (2% random moves, 0.083 behind per game).
  - Its snakes are alive at the end of the contest less often: 23.2% of
    seats against 32.7%.
  - Its two snakes collide head-on with each other more often: 6.2% of
    seats against 4.5% for the specialist's pair, and those games are almost
    always lost.
  - Grouped by the game's first head-on: games whose first head-on is
    generalist-vs-specialist account for −0.064 per game, and games whose
    first is between the generalist's own snakes −0.030. Games whose first
    is between the specialist's snakes give +0.022 back.
  - A generalist-vs-specialist head-on kills one of each, so the loss in
    those games comes mostly from the play that follows. The surviving
    generalist snake is also slightly worse placed at the collision (−0.80
    [−1.55, −0.03] Voronoi cells). [side analysis]
- **Royale does not depend on the protocol.** The generalist's lead holds
  greedy (0.705), sampled (0.698) and at every matched training snapshot:
  0.545, 0.646, 0.678, 0.706 and 0.701 against the specialist at the same
  step (512 games each). [benchmark; side analysis]

So the question becomes: why does the generalist gain a lot in the food modes
and nothing overall in constrictor?

## 3. What the evidence supports

No ablation separates the explanations below; each is labelled by how strong
its evidence is.

**Specialist headroom (consistent with every measurement, untested
causally).**

- **The constrictor specialist's curve is flat.**
  - Constrictor games last about 31 turns, against 150–270 in the other
    modes at the end of training. Per env step, its specialist completes 3–7
    times as many games as the others over the run (4–8 times at the end).
  - Its score against the heuristic is flat from 3M env steps (0.230, 0.230,
    0.234, 256 games each). [benchmark]
  - A flat curve fits a specialist near its ceiling but does not show it. The
    standard specialist's curve is flat too, at the floor, and the generalist
    still beats it 0.81.
- **The royale specialist is still climbing** (0.146 at 3M, 0.203 at 5.24M).
  The generalist's royale snapshot at 2M is already level with the final
  royale specialist: 0.480 [0.437, 0.524].
  - The standard specialist's score against the heuristic stays at the floor
    (0.008 to 0.025), which cannot show whether it still learns. It scores
    0.316 against three `ppo`, above parity. [benchmark; side analysis]
- **The account (untested).** If pooled experience helps where a specialist is
  short of data, this is the pattern it would produce.

**Shared skills (partly supported for what the critic learns; untested for
the policy).**

- **Royale shares the length levers of standard and the duel; constrictor
  lacks them.** Royale is standard plus a visible hazard zone. Food makes a
  snake longer there, and the longer snake survives a head-on collision.
  - It is the longer snake in 75% of its head-on collisions with the
    specialist (sampled play).
  - Among living snakes, the generalist's are about 0.25–0.3 cells longer
    25–75 moves in (sampled play; in greedy play only at move 25). The lead
    is gone by move 100 and reverses at 150–200, and pooled over the game it
    is not significant (+0.12 [−0.03, 0.26]).
  - In constrictor every snake has the same length, food does nothing, and
    head-on collisions always kill both snakes. Other skills, such as
    avoiding traps, matter in both modes. [side analysis]
- **The value-loss gradients agree early on.** The value-loss gradients of
  standard, the duel and royale point the same way at 1–3M env steps (mean
  de-noised cosine 0.66–0.87 over the three pairs). [side analysis]
  - Constrictor's is clearly less aligned with theirs at 1M and 3M (0.04 and
    0.10). At 2M the difference is borderline (0.46 against 0.80).
  - By 4M and 5.24M the food modes no longer agree with each other either
    (0.06 and 0.19, with wide intervals).
  - At 5.24M constrictor's points against standard's: raw cosine −0.31
    [−0.43, −0.19]. The de-noised figure is unstable there (−0.83, interval
    [−2.03, −0.46]).
  - The food modes also agree at the constrictor specialist's parameters.
    That does not show a property of the tasks, though: there the food-mode
    rollouts are off the specialist's distribution, and a shared value
    miscalibration could by itself align them.
- **Caveat on the gradients.** 83–98% of the value gradient's squared norm
  sits in the 257-parameter value head, at the generalist's snapshots.
  - On the shared convolutional and dense layers alone, constrictor's value
    gradient lines up with the food modes' at 2M and 3M (0.69, 0.63). It is
    not clearly aligned at 1M (−0.04) or 4M (0.16), whose intervals are wide.
  - At 5.24M it is slightly negative on the same statistic: −0.18
    [−0.61, 0.15], raw cosine −0.11 [−0.19, −0.03]. Only standard's pair is
    clearly negative (−0.57 [−0.79, −0.32]).
  - The value term is only 1/20–1/50 of those layers' gradient. The
    policy-gradient term dominates them and was pure noise at the sample
    sizes used.
  - So there is no direct evidence either way about what the shared layers
    learn for constrictor. [side analysis]
- **Zero-shot transfer is no better into royale than into constrictor.** The
  standard specialist scores 0.098 in royale and 0.108 in constrictor
  against three heuristics (256 games). "Royale is close to standard" holds
  at most for the value function, not for the policy as a whole.
  [benchmark]

**Candidate interference effects (differences seen; cost not shown).**

- **Moving-tail habit.** The countdown channels give tail-moves until a body
  cell is free, which is true in the food modes but not in constrictor,
  where tails never move. Changing only those channels flips the
  generalist's greedy move about 5 times as often as the specialist's.
  - When every body is made to look far from clearing, the flip rate is
    0.110 against 0.022 (decisions with at least two legal moves, from
    sampled G,S,G,S games).
  - Reweighting the specialist's decisions to the generalist's confidence
    gives 0.026, about 4 times.
  - Whether the generalist reads the channels as tails that will move is only
    partly settled. Some probes move its trap rate in the tail-moving
    direction, where the specialist's barely moves:
    - making every body look about to clear raises its trap rate by 0.042
      (the specialist's by 0.003);
    - shifting the countdowns 5 turns later lowers it by 0.009 (the
      specialist's by 0.004).
  - Other probes do not. Making every body look far from clearing leaves its
    trap rate unchanged (+0.001), and a 30-turn shift raises it (+0.029).
    [side analysis]
- **Moves that only have room if tails move.** Some decisions have a move
  that is roomy only if tails move. Where that room margin is 5 cells, the
  generalist picks it 13% of the time, against 7% for the specialist (2,048
  sampled games).
  - The ratio is about 1.9× at margins of 5 and 10 cells and about 1.35× at
    1 and 3.
  - In fully sampled play its snakes die in such traps 0.013 more often per
    seat per game (0.028 against 0.016), while total deaths are equal
    (0.782 against 0.780).
  - It was not measured in the perturbed greedy play where the deficit
    lies. [side analysis]
- **A small value-loss weight.** The value loss is scaled per mode by
  1/variance of the GAE returns, which gives constrictor a weight of about
  0.3 (0.28–0.34 after the first tenth of training) against 0.9–1.6 for the
  other modes: 8% of the total.
  - The raw Monte Carlo returns vary equally for both policies (0.342 and
    0.345).
  - The generalist's GAE returns vary a little less (0.177 against 0.192),
    because its critic is flatter (variance of V 0.031 against 0.066).
  - Lower variance raises the weight slightly, so the flatter critic and the
    low weight partly offset each other rather than reinforce.
  - The generalist's constrictor value error (MSE) is 6–9% above the
    specialist's in the training logs, and 1.5–3% on fresh rollouts.
  - Its explained variance is lower (0.22 against 0.35). With a similar MSE,
    its flatter critic removes less of the return variance: 0.041 against
    0.058 on fresh rollouts. [benchmark; side analysis]
- **A mid-training dip.** The generalist's constrictor score against three
  heuristics fell from 1M to 3M env steps by −0.074 ± 0.043, then recovered.
  The specialist was ahead only at 2–3M (−0.109 ± 0.045 at 3M, 512 games).
  [side analysis]

**Ruled out.**

- **A wrong mode input.** Told a wrong mode, the generalist changes 2–6% of
  its greedy moves. Its strength drops by 0.02–0.04 in constrictor (told
  duel: −0.040 [−0.080, −0.001], 256 paired games). The benchmark always tells
  it the right mode, so this cannot cause a gap. This tests only the mode
  input; habits carried over through the board, like the moving-tail one
  above, are a separate question. [benchmark; side analysis]
- **A broken constrictor critic.** Its value error is close to the
  specialist's (above).

**Not established.**

- **Rock-paper-scissors between snapshots.** Constrictor snapshot
  head-to-heads swing from 0.13 to 0.72, and a round robin looked cyclic.
  With 12 distinct games per pair, the transitivity test gives p ≈ 0.075.
  At that power, intransitivity is neither established nor ruled out.
  [side analysis]

**Not tested.**

- **Extra gradient steps.** Specialists trained with the generalist's 16
  minibatches are no better head to head in standard, royale or the duel, so
  the extra steps do not explain the gains there. They tend to weaken a
  specialist head to head, though (duel −0.106 ± 0.049, royale
  −0.044 ± 0.030), and the constrictor control was stopped. So a cost of the
  extra steps in constrictor is possible. [benchmark]

## 4. Not finished

Two experiments were cut short. A container restart interrupted them; they
were resumed, then stopped for good when no more training was wanted. This
is recorded in the runs' own logs, which are not committed:

- **Fine-tuning test.** The generalist and the constrictor specialist each
  get 1M more constrictor-only env steps, then play each other. This would
  show whether the generalist's features are as good as the specialist's.
  The last saved checkpoint is at 0.26M. The live logs, since overwritten,
  showed about 0.45M before the restart, which can no longer be checked.
- **Constrictor control.** A 16-minibatch constrictor specialist, stopped at
  2.9M of 5.24M env steps.

Other ways to settle the open part (each costs more training than finishing
the two runs above):

- a generalist trained without the per-mode value weighting;
- a three-mode generalist without constrictor, compared in the food modes;
- a constrictor specialist given four times the env steps, to test the
  headroom account.
