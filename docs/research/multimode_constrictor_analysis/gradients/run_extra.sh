#!/bin/bash
cd /home/user/slinkydotcpp
while [ ! -f docs/research/multimode_constrictor_analysis/gradients/run_all.done ]; do sleep 5; done
taskset -c 0 .venv/bin/python docs/research/multimode_constrictor_analysis/gradients/grads.py runs/ppo-modes-seed0 params-5242880.npz docs/research/multimode_constrictor_analysis/gradients/G-5242880-s1.npz --seed 1 > docs/research/multimode_constrictor_analysis/gradients/G-5242880-s1.log 2>&1
taskset -c 0 .venv/bin/python docs/research/multimode_constrictor_analysis/gradients/grads.py runs/ppo-modes-seed0 params-1048576.npz docs/research/multimode_constrictor_analysis/gradients/G-1048576-s1.npz --seed 1 --rollouts 4 > docs/research/multimode_constrictor_analysis/gradients/G-1048576-s1.log 2>&1
echo DONE > docs/research/multimode_constrictor_analysis/gradients/run_extra.done
