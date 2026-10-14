#!/bin/bash
cd /home/user/slinkydotcpp
D=docs/research/multimode_constrictor_analysis/snapshots
OUT=$D/cells.jsonl
SNAPS="1048576 2097152 3145728 4194304 5242880"
G=runs/ppo-modes-seed0
run() { # mode seats A B games
  taskset -c 3 .venv/bin/python benchmarks/modes.py --game-mode "$1" --seats "$2" --agent "A=$3" --agent "B=$4" --games "$5" --out $OUT 2>&1 | grep -E "done|Error|Traceback"
}
for MODE in constrictor royale; do
  S=runs/ppo-modes-$MODE-seed0
  if [ $MODE = constrictor ]; then NG=1024; else NG=512; fi
  for k in $SNAPS; do   # (a)
    run $MODE A,B,A,B "ppo_modes:$G:snapshot=$k" "ppo_modes:$S" $NG
  done
  for k in $SNAPS; do   # (c)
    run $MODE A,B,A,B "ppo_modes:$G:snapshot=$k" "ppo_modes:$S:snapshot=$k" $NG
  done
  for k in 1048576 2097152 3145728 4194304; do   # (b)
    run $MODE A,B,A,B "ppo_modes:$S:snapshot=$k" "ppo_modes:$S" $NG
  done
  if [ $MODE = constrictor ]; then
    for k in $SNAPS; do   # (d)
      run $MODE A,H,H,H "ppo_modes:$G:snapshot=$k" heuristic 512
      run $MODE A,H,H,H "ppo_modes:$S:snapshot=$k" heuristic 512
    done
  fi
done
echo ALLDONE
