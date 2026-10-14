# Constrictor investigation: analysis scripts

These are the exploratory scripts and small outputs behind the **[side analysis]**
results of [`../multimode_constrictor.md`](../multimode_constrictor.md). They are
kept as a record, not as maintained code:

- **Not linted or tested**, and excluded from ruff.
- **Paths.** In the scripts, paths into the original scratch directory were
  rewritten to this directory. Others are absolute (`/home/user/slinkydotcpp/...`)
  or relative to the repository root, and `runs/...` are the training runs.
  The commands recorded at the top of the re-run outputs keep the original
  `scratch/multimode/analysis/...` paths: that is where the scripts and their
  data were when they ran.
- **Many cannot be re-run from the repository.** They read the training
  snapshots (`runs/<run>/params-<steps>.npz`) or intermediate files (`.npz`,
  `cells.jsonl`), which are not committed. Only the final networks are, in
  `baselines/checkpoints/`.
- **Large outputs were left out**: arrays (`.npz`, `.npy`), logs and any text
  file over 200 kB.
- **Printed outputs.** Where a script printed its results, they are next to it
  (`*.txt`, `*.json`; `diagnostics/out_*.txt`; `*_out.txt` for the outputs re-run
  for this archive).
- **`investigation_log.json`** is the raw output of the investigation: five
  investigations, a synthesis, and two reviews that re-ran key scripts.
  Several claims in the investigations and the synthesis were corrected or
  weakened by the reviews and by a later fact-check.
- **The note is the reference.** Where a script's output, the log and the note
  differ, the note holds the checked number.

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
