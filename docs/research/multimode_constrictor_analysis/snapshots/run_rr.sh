#!/bin/bash
# constrictor round robin over {G@k, S@k}, 512 games per new pair (existing 1024-game cells are reused)
cd /home/user/slinkydotcpp
D=docs/research/multimode_constrictor_analysis/snapshots
OUT=$D/cells.jsonl
until grep -q ALLDONE $D/run.log; do sleep 10; done
SNAPS="1048576 2097152 3145728 4194304 5242880"
G=runs/ppo-modes-seed0
S=runs/ppo-modes-constrictor-seed0
gname() { echo "ppo_modes:$G:snapshot=$1"; }
sname() { if [ "$1" = 5242880 ]; then echo "ppo_modes:$S"; else echo "ppo_modes:$S:snapshot=$1"; fi; }
run() {
  taskset -c 3 .venv/bin/python benchmarks/modes.py --game-mode constrictor --seats A,B,A,B --agent "A=$1" --agent "B=$2" --games 512 --out $OUT 2>&1 | grep -E "done|Error|Traceback|finished"
}
# G-S
for i in $SNAPS; do for j in $SNAPS; do run "$(gname $i)" "$(sname $j)"; done; done
# G-G (i<j)
for i in $SNAPS; do for j in $SNAPS; do if [ $i -lt $j ]; then run "$(gname $i)" "$(gname $j)"; fi; done; done
# S-S (i<j)
for i in $SNAPS; do for j in $SNAPS; do if [ $i -lt $j ]; then run "$(sname $i)" "$(sname $j)"; fi; done; done
echo RRDONE
