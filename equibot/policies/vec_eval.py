import os
import sys
import torch
import hydra
import omegaconf
import wandb
import numpy as np
import getpass as gt
from tqdm import tqdm

from equibot.policies.eval import organize_obs
from equibot.policies.utils.media import combine_videos


def run_eval(
    env,
    agent,
    vis=False,
    num_episodes=1,
    log_dir=None,
    reduce_horizon_dim=True,
    verbose=False,
    use_wandb=False,
    ckpt_name=None,
):
    if hasattr(agent, "obs_horizon") and hasattr(agent, "ac_horizon"):
        obs_horizon = agent.obs_horizon
        ac_horizon = agent.ac_horizon
    else:
        obs_horizon = 1
        ac_horizon = 1

    images = []
    obs_history = []
    num_envs = len(env.remotes)
    for i in range(num_envs):
        images.append([])
    state = env.reset()

    env_module_name = env.get_attr("__module__")[0]

    pred_horizon = agent.pred_horizon if hasattr(agent, "pred_horizon") else 1
    rgb_render = render = env.env_method("render")
    obs = organize_obs(render, rgb_render, state)
    for i in range(obs_horizon):
        obs_history.append(obs)
    for i in range(num_envs):
        images[i].append(rgb_render[i]["images"][0][..., :3])

    sample_pc = render[0]["pc"]
    mean_num_points_in_pc = np.mean([len(render[k]["pc"]) for k in range(len(render))])

    done = [False] * num_envs
    if log_dir is not None:
        history = []
        for i in range(num_envs):
            history.append(dict(action=[], eef_pos=[]))
    t = 0
    pbar = tqdm(
        list(range(env.get("args").max_episode_length // ac_horizon)),
        leave=False,
        desc="Vec Eval",
    )
    while not np.all(done):
        # make obs for agent
        if obs_horizon == 1 and reduce_horizon_dim:
            agent_obs = obs
        else:
            agent_obs = dict()
            for k in obs.keys():
                if k == "pc":
                    # point clouds can have different number of points
                    # so do not stack them
                    agent_obs[k] = [o[k] for o in obs_history[-obs_horizon:]]
                else:
                    agent_obs[k] = np.stack([o[k] for o in obs_history[-obs_horizon:]])

        # predict actions
        ac = agent.act(
            agent_obs
        )  # (num_envs, agent.pred_horizon, num_eef * dof)

        if log_dir is not None:
            for i in range(num_envs):
                history[i]["action"].append(ac[i])
                history[i]["eef_pos"].append(obs["state"][i])

        # take actions
        for ac_ix in range(ac_horizon):
            agent_ac = ac[:, ac_ix] if len(ac.shape) > 2 else ac
            env.step_async(agent_ac, dummy_reward=True)
            state, _, done, _ = env.step_wait()
            rgb_render = render = env.env_method("render")
            obs = organize_obs(render, rgb_render, state)
            obs_history.append(obs)
            if len(obs) > obs_horizon:
                obs_history = obs_history[-obs_horizon:]
            for i in range(num_envs):
                images[i].append(rgb_render[i]["images"][0][..., :3])
        t += 1
        pbar.update(1)
    pbar.close()
    rews = np.array(env.env_method("compute_reward"))
    print(f"Episode rewards: {rews.round(3)}.")

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        for ep_ix in range(num_envs):
            np.savez(
                os.path.join(
                    log_dir, f"eval_{ckpt_name}_ep{ep_ix:02d}_rew{rews[ep_ix]:.3f}.npz"
                ),
                action=np.array(history[ep_ix]["action"]),
                eef_pos=np.array(history[ep_ix]["eef_pos"]),
            )

    images = np.array(images)
    metrics = dict(rew=np.mean(rews))
    if vis:
        vis_frames = images  # N, T, H, W, C
        vis_rews = np.zeros_like(vis_frames[:, :, :20])
        for i in range(num_envs):
            num_pixels = int(vis_frames.shape[-2] * rews[i])
            vis_rews[i, :, 2:, :num_pixels] = 255
        vis_frames = np.concatenate([vis_frames, vis_rews], axis=2)
        if use_wandb:
            metrics["vis_pc"] = wandb.Object3D(sample_pc)
        metrics["rew_values"] = rews
        metrics["vis_rollout"] = vis_frames
        metrics["mean_pc_size"] = mean_num_points_in_pc
    return metrics
