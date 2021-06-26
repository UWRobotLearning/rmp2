# RMP2


Code for R:SS 2021 paper *RMP2: A Structured Composable Policy Class for Robot Learning*. [[Paper](https://arxiv.org/abs/2103.05922)] 

### Installation
```
git clone https://github.com/UWRobotLearning/rmp2.git
cd rmp2
conda env create -f environment.yml
. startup.sh
```

### Hand-designed RMP2 for Robot Control
To run a goal reaching task for a Franka robot:
```
python examples/rmp2/rmp2_franka.py
```

To run a goal reaching task for a 3-link robot:
```
python examples/rmp2/rmp2_3link.py
```

### Training RMP2 Policies with RL
**Note:** The instruction below is for the 3-link robot. To run experiments with the franka robot, simply replace `3link` by `franka`.

To train an NN policy from scratch (without RMP2):
```
python run_3link_nn.py
```

To train an NN residual policy:
```
python run_3link_nn.py --env 3link_residual
```

To train an RMP residual policy:
```
python run_3link_residual_rmp.py
```

To restore training of a policy:
```
python restore_training.py --ckpt-path ~/ray_results/[EXPERIMENT_NAME]/[RUN_NAME]/
```

To visualize the trained policy:
```
python examples/rl/run_policy_rollouts.py --ckpt-path ~/ray_results/[EXPERIMENT_NAME]/[RUN_NAME]/
```

### Citation
If you use this source code, please cite the below article,

```
@inproceedings{Li-RSS-21,
    author = "Li, Anqi and Cheng, Ching-An and Rana, M Asif and Xie, Man and Van Wyk, Karl and Ratliff, Nathan and Boots, Byron",
    booktitle = "Robotics: Science and Systems ({R:SS})",
    title = "{{RMP}2: A Structured Composable Policy Class for Robot Learning}",
    year = "2021"
}
```
