#!/bin/bash
cd /home/user/slinkydotcpp
for s in 2097152 3145728 4194304 5242880; do
  taskset -c 0 .venv/bin/python docs/research/multimode_constrictor_analysis/gradients/grads.py runs/ppo-modes-seed0 params-$s.npz docs/research/multimode_constrictor_analysis/gradients/G-$s.npz > docs/research/multimode_constrictor_analysis/gradients/G-$s.log 2>&1
done
taskset -c 0 .venv/bin/python docs/research/multimode_constrictor_analysis/gradients/grads.py runs/ppo-modes-constrictor-seed0 params-5242880.npz docs/research/multimode_constrictor_analysis/gradients/Sconstrictor-5242880.npz > docs/research/multimode_constrictor_analysis/gradients/Sconstrictor-5242880.log 2>&1
taskset -c 0 .venv/bin/python docs/research/multimode_constrictor_analysis/gradients/grads.py runs/ppo-modes-standard-seed0 params-5242880.npz docs/research/multimode_constrictor_analysis/gradients/Sstandard-5242880.npz > docs/research/multimode_constrictor_analysis/gradients/Sstandard-5242880.log 2>&1
echo ALLDONE > docs/research/multimode_constrictor_analysis/gradients/run_all.done
