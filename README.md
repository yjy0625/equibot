# EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning

Jingyun Yang*, Zi-ang Cao*, Congyue Deng, Rika Antonova, Shuran Song, Jeannette Bohg

<a href='https://equi-bot.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2407.01479'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/FFrl_TEXrUw)

![Overview figure](https://equi-bot.github.io/images/teaser.jpg)

This repository includes:

* Implementation of the EquiBot method and a Diffusion Policy baseline that takes point clouds as input.
* A set of three simulated mobile manipulation environments: Cloth Folding, Object Covering, and Box Closing.
* Data generation, training, and evaluation scripts that accompany the above algorithms and environments.

## Getting Started

### Installation

This codebase is tested with the following setup: Ubuntu 20.04, an RTX 4090 GPU, CUDA 11.8.

```
conda create -n lfd python=3.10 -y
conda activate lfd

conda install -y fvcore iopath ffmpeg -c iopath -c fvcore
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install -e .
```

Then, in the last two lines of [this config file](equibot/policies/configs/base.yaml), enter the wandb entity and project names for logging purposes.

### Demonstration Generation

The following code generates demonstrations for simulated mobile environments. To change number of generated demos, change `--num_demos 50` to a different number.

```
python -m equibot.envs.sim_mobile.generate_demos --data_out_dir ../data/fold \
    --num_demos 50 --cam_dist 2 --cam_pitches -75 --task_name fold

python -m equibot.envs.sim_mobile.generate_demos --data_out_dir ../data/cover \
    --num_demos 50 --cam_dist 2 --cam_pitches -75 --task_name cover

python -m equibot.envs.sim_mobile.generate_demos --data_out_dir ../data/close \
    --num_demos 50 --cam_dist 1.5 --cam_pitches -45 --task_name close
```

### Training

The following code runs training for our method and the Diffusion Policy baseline. Fill the dataset path with the `data_out_dir` argument in the previous section. Make sure the dataset path ends with `pcs`. To run this code for the `cover` and `close` environments, substitute occurances of `fold` with `cover` or `close`.

```
# diffusion policy baseline (takes point clouds as input)
python -m equibot.policies.train --config-name fold_mobile_dp \
    prefix=sim_mobile_fold_7dof_dp \
    data.dataset.path=[data out dir in the last section]/pcs

# our method (equibot)
python -m equibot.policies.train --config-name fold_mobile_equibot \
    prefix=sim_mobile_fold_7dof_equibot \
    data.dataset.path=[data out dir in the last section]/pcs
```

### Evaluation

The commands below evaluates the trained EquiBot policy on the four different setups mentioned in the paper: `Original`, `R+Su`, `R+Sn`, and `R+Sn+P`. To run these evaluations for the DP baseline, simply replace all occurances of `equibot` to`dp`. For the log directory, fill `[log_dir]` with the absolute path to log directory. By default, this directory is `./log`.

```
# Original setup
python -m equibot.policies.eval --config-name fold_mobile_equibot \
    prefix="eval_original_sim_mobile_fold_equibot_s1" \
    mode=eval \
    training.ckpt="[log_dir]/train/sim_mobile_fold_7dof_equibot_s1/ckpt01999.pth" \
    env.args.max_episode_length=50 env.vectorize=true

# R+Su setup
python -m equibot.policies.eval --config-name fold_mobile_equibot \
    prefix="eval_rsu_sim_mobile_fold_7dof_equibot_s1" \
    mode=eval \
    training.ckpt="[log_dir]/train/sim_mobile_fold_7dof_equibot_s1/ckpt01999.pth" \
    env.args.scale_high=2 \
    env.args.uniform_scaling=true \
    env.args.randomize_rotation=true \
    env.args.randomize_scale=true env.vectorize=true

# R+Sn setup
python -m equibot.policies.eval --config-name fold_mobile_equibot \
    prefix="eval_rsn_sim_mobile_fold_7dof_equibot_s1" \
    mode=eval \
    training.ckpt="[log_dir]/train/sim_mobile_fold_7dof_equibot_s1/ckpt01999.pth" \
    env.args.scale_high=2 \
    env.args.scale_aspect_limit=1.33 \
    env.args.randomize_rotation=true \
    env.args.randomize_scale=true \
    env.vectorize=true

# R+Sn+P setup
python -m equibot.policies.eval --config-name fold_mobile_equibot \
    prefix="eval_rsnp_sim_mobile_fold_7dof_equibot_s1" \
    mode=eval \
    training.ckpt="[log_dir]/train/sim_mobile_fold_7dof_equibot_s1/ckpt01999.pth" \
    env.args.scale_high=2 \
    env.args.scale_aspect_limit=1.33 \
    env.args.randomize_rotation=true \
    env.args.randomize_scale=true \
    +env.args.randomize_position=true \
    +env.args.rand_pos_scale=0.5 \
    env.vectorize=true
```

## License

This codebase is licensed under the terms of the MIT License.
