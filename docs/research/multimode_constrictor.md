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
- **The scripts and their small outputs** are kept as a record in
  [`multimode_constrictor_analysis/`](multimode_constrictor_analysis/). They
  are unmaintained and not linted, their paths are those of the original
  session, and many need the training snapshots (`params-<steps>.npz`),
  which are not committed.
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

## 2. Measured properly, it is level

| measure | generalist's share | source |
|---|---|---|
| sampled play, G,S,G,S, 1,024 games (essentially all distinct) | 0.494 [0.463, 0.524] | [benchmark] |
| sampled play, two more runs (1,024 and 2,048 games) | 0.495 [0.466, 0.524] and 0.488 [0.468, 0.508] | [side analysis] |
| 3 sampled opening moves, then greedy (631 distinct of 1,024) | 0.490 [0.461, 0.518] | [side analysis] |
| greedy with a random legal move 2% / 5% / 10% of the time (2,048 games each) | 0.417 [0.396, 0.437] / 0.438 [0.418, 0.458] / 0.461 [0.440, 0.481] | [side analysis] |
| against 3 heuristics, G − S per seat (paired, 1,024 games) | −0.012 ± 0.032 | [benchmark] |
| against 3 `ppo`, G − S per seat (paired, 1,024 games) | +0.056 ± 0.039 | [benchmark] |
| against 1 heuristic and 2 `random_legal`, G − S per seat (paired, 1,024 games) | −0.046 ± 0.038 | [benchmark] |

- **No overall gain.** The generalist is level with its specialist under
  sampled play and against three heuristics. It is slightly ahead against
  `ppo` and slightly behind with one heuristic and two `random_legal`.
- **Small near-greedy deficit.** With occasional random moves it is 0.04–0.08
  behind. The three runs reuse the same random draws, so they are not
  independent confirmations.
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
  - Constrictor games last about 32 turns, against 150–270 in the other
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
    (0.016 to 0.025), which cannot show whether it still learns. It scores
    0.316 against three `ppo`, above parity. [benchmark; side analysis]
- **The account (untested).** If pooled experience helps where a specialist is
  short of data, this is the pattern it would produce.

**Shared skills (supported for what the critic learns; untested for the
policy).**

- **Royale shares its levers with standard and the duel; constrictor has
  none of them.** Royale is standard plus a visible hazard zone.
  - In sampled royale play the generalist's snakes are about 0.25–0.3 cells
    longer 25–75 moves in; in greedy play only at move 25.
  - It is the longer snake in 75% of its head-on collisions with the
    specialist, and the longer snake survives a head-on collision.
  - In constrictor every snake has the same length, food does nothing, and
    head-on collisions always kill both snakes. [side analysis]
- **The value-loss gradients agree.** Those of standard, the duel and royale
  point the same way: de-noised cosine 0.66–0.87 at 1–3M env steps. That
  holds at the constrictor specialist's parameters too, so it is a property
  of the tasks.
  - Constrictor's is much less aligned with theirs over all parameters
    (0.04–0.46 at 1–4M).
  - At 5.24M it points against standard's (−0.83 [−2.03, −0.46]). Its mean
    against the three food modes there is −0.33 [−1.10, 0.06]. [side analysis]
- **Caveat on the gradients.** 83–99% of the value gradient's squared norm
  sits in the 257-parameter value head. On the shared convolutional and dense layers,
  constrictor's value gradient lines up with the food modes' (0.69 at 2M,
  0.63 at 3M). The policy-gradient term, which dominates those layers' gradient,
  was pure noise at the sample sizes used. So there is no direct evidence
  that the shared layers learn less for constrictor. [side analysis]
- **Zero-shot transfer is no better into royale than into constrictor.** The
  standard specialist scores 0.098 in royale and 0.108 in constrictor
  against three heuristics (256 games). "Royale is close to standard" holds
  for the value function, not for the policy as a whole. [benchmark]

**Small interference costs (real, small).**

- **Moving-tail habit.** The countdown channels give tail-moves until a body
  cell is free, which is true in the food modes but not in constrictor,
  where tails never move. Changing only those channels flips the
  generalist's greedy move about 5 times as often as the specialist's: 0.101
  against 0.019 when every body is made to look far from clearing. This
  holds after matching the two policies' confidence. [side analysis]
- **Moves that only have room if tails move.** In decisions where the room
  available differs depending on whether tails move, the generalist picks
  the move that is only roomy if they do 13% of the time, against 7% for the
  specialist. The gap is 2× at margins of 5 and 10 cells and about 1.35× at
  1 and 3. The cost is about 0.013 extra trap deaths per generalist seat per
  game, under 0.01 of share. [side analysis]
- **Where the near-greedy deficit comes from.** With 2% random moves the
  generalist is 0.083 behind per game.
  - Its seats run out of safe moves more often: 49.4% of them against 41.7%.
  - It collides head-on with its own teammate more often: 6.2% of seats
    against 4.5%, and those games are almost always lost.
  - Split by the game's first head-on, generalist-vs-specialist head-ons cost
    0.064 per game and the generalist's own two snakes colliding 0.030.
    Specialist-vs-specialist head-ons win 0.022 back. [side analysis]
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
  - Its explained variance is lower (0.22 against 0.35), mostly because its
    returns vary less. [benchmark; side analysis]
- **A mid-training dip.** The generalist's constrictor score against three
  heuristics fell from 1M to 3M env steps by −0.074 ± 0.043, then recovered.
  The specialist was ahead only at 2–3M (−0.109 ± 0.045 at 3M, 512 games).
  [side analysis]

**Ruled out.**

- **Mode confusion.** Told a wrong mode, the generalist changes 2–6% of its
  greedy moves. Its strength drops by 0.02–0.04 in constrictor (told duel:
  −0.040 [−0.080, −0.001], 256 paired games). The benchmark always tells it
  the right mode, so this cannot cause a gap. [benchmark; side analysis]
- **A broken constrictor critic.** Its value error is close to the
  specialist's (above).
- **Rock-paper-scissors between snapshots.** Constrictor snapshot
  head-to-heads swing from 0.13 to 0.72, and a round robin looked cyclic.
  With 12 distinct games per pair, the transitivity test gives p ≈ 0.075, so
  intransitivity is not established. [side analysis]

**Not tested.**

- **Extra gradient steps.** Specialists trained with the generalist's 16
  minibatches are no better head to head in standard, royale or the duel, so
  the extra steps do not explain the gains there. They tend to weaken a
  specialist head to head, though (duel −0.106 ± 0.049, royale
  −0.044 ± 0.030), and the constrictor control was stopped. So a cost of the
  extra steps in constrictor is possible. [benchmark]

## 4. Not finished

Two experiments were started and stopped before the end, because no more
training was wanted:

- **Fine-tuning test.** The generalist and the constrictor specialist each
  get 1M more constrictor-only env steps, then play each other. This would
  show whether the generalist's features are as good as the specialist's.
  The runs reached about 0.45M env steps before a container restart stopped
  them (seen in their live logs, since overwritten). The last saved
  checkpoint is at 0.26M.
- **Constrictor control.** A 16-minibatch constrictor specialist, stopped at
  2.9M of 5.24M env steps.

Other ways to settle the open part (each costs more training than finishing
the two runs above):

- a generalist trained without the per-mode value weighting;
- a three-mode generalist without constrictor, compared in the food modes;
- a constrictor specialist given four times the env steps, to test the
  headroom account.
