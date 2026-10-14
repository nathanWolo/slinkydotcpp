# Constrictor investigation: analysis scripts

These are the exploratory scripts and small outputs behind the **[side analysis]**
results of [`../multimode_constrictor.md`](../multimode_constrictor.md). They are
kept as a record, not as maintained code:

- **Not linted or tested**, and excluded from ruff.
- **Paths are the original session's.** `scratch/multimode/...` was a scratch
  directory, and `runs/...` the training runs.
- **Many need the training snapshots** (`runs/<run>/params-<steps>.npz`), which are
  not committed. Only the final networks are, in `baselines/checkpoints/`.
- **Large outputs were left out**: arrays (`.npz`, `.npy`), logs and any text
  file over 200 kB.
- **The note is the reference.** Where a script's output and the note differ,
  the note holds the checked number.

| directory | what it measures |
|---|---|
| `diversity/` | distinct games per benchmark cell (the 12-game finding) |
| `wrong_mode_moves.py` | how often the generalist's greedy move changes when told a wrong mode |
| `snapshots/` | matched training-snapshot head-to-heads and the constrictor round robin |
| `behaviour/` | simulated games under greedy, sampled and eps-greedy play: lengths, space, trap deaths, determinism |
| `tailprobe/` | counterfactual probes of the tail-countdown channels and room-if-tails-move decisions |
| `gradients/` | per-mode value- and policy-gradient alignment at training snapshots |
| `diagnostics/` | per-mode training-curve windows and gaps from `metrics.jsonl` |
| `synthesis/` | the eps-greedy split and the round robin's effective sample size |
| `verify-method/`, `verify-statistics/` | the reviewers' re-runs: Monte Carlo return variance, value error, bootstraps, paired heuristic cells, snapshot checks |
