import os
import sys
import time
import torch
import hydra
import omegaconf
import wandb
import numpy as np
import getpass as gt
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf

from equibot.policies.utils.media import combine_videos, save_video
from equibot.policies.utils.misc import get_env_class, get_dataset, get_agent
from equibot.envs.subproc_vec_env import SubprocVecEnv


def organize_obs(render, rgb_render, state):
    if type(render) is list:
        obs = dict(
            pc=[r["pc"] for r in render],
            rgb=np.array([r["images"][0][..., :3] for r in rgb_render]),
            # depth=np.array([r["depths"][0] for r in render]),
            state=np.array(state),
        )
        for k in ["eef_pos", "eef_rot"]:
            if k in render[0]:
                obs[k] = [r[k] for r in render]
        return obs
    elif type(render) is dict:
        obs = organize_obs([render], [rgb_render], [state])
        return {k: v[0] for k, v in obs.items()}


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
    if vis:
        vis_frames = []

    if hasattr(agent, "obs_horizon") and hasattr(agent, "ac_horizon"):
        obs_horizon = agent.obs_horizon
        ac_horizon = agent.ac_horizon
        pred_horizon = agent.pred_horizon
    else:
        obs_horizon = 1
        ac_horizon = 1
        pred_horizon = 1

    images, rews = [], []
    for ep_ix in range(num_episodes):
        images.append([])
        obs_history = []
        state = env.reset()

        rgb_render = render = env.render()
        obs = organize_obs(render, rgb_render, state)
        for i in range(obs_horizon):
            obs_history.append(obs)
        images[-1].append(rgb_render["images"][0][..., :3])

        if ep_ix == 0:
            sample_pc = render["pc"]

        done = False
        global_features = None
        prev_reward = None
        if log_dir is not None:
            history = dict(action=[], eef_pos=[])
        while not done:
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
                        agent_obs[k] = np.stack(
                            [o[k] for o in obs_history[-obs_horizon:]]
                        )

            # predict actions
            st = time.time()
            ac, ac_dict = agent.act(agent_obs, return_dict=True)
            print(f"Inference time: {time.time() - st:.3f}s")
            if ac_dict is not None:
                if (
                    "expected_eef_pos" in ac_dict
                    and ac_dict["expected_eef_pos"] is not None
                ):
                    env.visualize_anchor(ac_dict["expected_eef_pos"][:, :3])
                if "expected_pc" in ac_dict and ac_dict["expected_pc"] is not None:
                    if hasattr(env, "visualize_pc"):
                        env.visualize_pc(ac_dict["expected_pc"])
            else:
                print(f"Warning: ac dict is none!")
            if log_dir is not None:
                history["action"].append(ac)
                history["eef_pos"].append(obs["state"])

            # take actions
            for ac_ix in range(ac_horizon):
                if len(obs["pc"]) == 0 or len(obs["pc"][0]) == 0:
                    ac_dict = None
                    break
                agent_ac = ac[ac_ix] if len(ac.shape) > 1 else ac
                state, rew, done, info = env.step(agent_ac, dummy_reward=True)
                if hasattr(env, "visualize_eef_frame"):
                    env.visualize_eef_frame(state)
                rgb_render = render = env.render()
                obs = organize_obs(render, rgb_render, state)
                obs_history.append(obs)
                if len(obs) > obs_horizon:
                    obs_history = obs_history[-obs_horizon:]
                images[-1].append(rgb_render["images"][0][..., :3])
                if vis:
                    vis_frames.append(rgb_render["images"][0][..., :3])
                curr_reward = env.compute_reward()
                if prev_reward is None or curr_reward > prev_reward:
                    prev_reward = curr_reward
                if (
                    ac_dict is None
                    or done
                    or (
                        hasattr(env, "_failed_safety_check")
                        and env._failed_safety_check
                    )
                ):
                    break
            if (
                ac_dict is None
                or done
                or (hasattr(env, "_failed_safety_check") and env._failed_safety_check)
            ):
                break
        if ac_dict is not None:
            rew = env.compute_reward()
        else:
            rew = prev_reward
        rews.append(rew)
        print(f"Episode {ep_ix + 1} reward: {rew:.4f}.")

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            np.savez(
                os.path.join(log_dir, f"eval_ep{ep_ix:02d}_rew{rew:.3f}.npz"),
                action=np.array(history["action"]),
                eef_pos=np.array(history["eef_pos"]),
            )
    max_num_images = np.max([len(images[i]) for i in range(len(images))])
    for i in range(len(images)):
        if len(images[i]) < max_num_images:
            images[i] = images[i] + [images[i][-1]] * (max_num_images - len(images[i]))
    images = np.array(images)
    rews = np.array(rews)

    pos_idxs, neg_idxs = np.where(rews >= 0.5)[0], np.where(rews < 0.5)[0]
    metrics = dict(rew=np.mean(rews))
    fps = 30 if "sim_mobile" in env.__module__ else 4
    if use_wandb:
        if len(pos_idxs) > 0:
            metrics["video_pos"] = wandb.Video(
                combine_videos(images[pos_idxs][:6], num_cols=5).transpose(0, 3, 1, 2),
                fps=30,
            )
        if len(neg_idxs) > 0:
            metrics["video_neg"] = wandb.Video(
                combine_videos(images[neg_idxs][:6], num_cols=5).transpose(0, 3, 1, 2),
                fps=30,
            )
        if vis:
            metrics["vis_rollout"] = images
            metrics["vis_pc"] = wandb.Object3D(sample_pc)
    else:
        metrics["vis_rollout"] = images
    return metrics


@hydra.main(config_path="configs", config_name="fold_synthetic")
def main(cfg):
    assert cfg.mode == "eval"
    device = torch.device(cfg.device)
    if cfg.use_wandb:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            tags=["eval"],
            name=cfg.prefix,
            settings=wandb.Settings(code_dir="."),
            config=wandb_config,
        )
    np.random.seed(cfg.seed)

    if cfg.env.vectorize:
        env_fns = []
        env_class = get_env_class(cfg.env.env_class)
        env_args = dict(OmegaConf.to_container(cfg.env.args, resolve=True))

        def create_env(env_args, i):
            env_args["seed"] = cfg.seed * 100 + i
            return env_class(OmegaConf.create(env_args))

        env = SubprocVecEnv(
            [
                lambda seed=i: create_env(env_args, seed)
                for i in range(cfg.training.num_eval_episodes)
            ]
        )
        from equibot.policies.vec_eval import run_eval as run_vec_eval

        eval_fn = run_vec_eval
    else:
        env = get_env_class(cfg.env.env_class)(cfg.env.args)
        eval_fn = run_eval

    agent = get_agent(cfg.agent.agent_name)(cfg)
    agent.train(False)

    if os.path.isdir(cfg.training.ckpt):
        ckpt_dir = cfg.training.ckpt
        ckpt_paths = list(glob(os.path.join(ckpt_dir, "ckpt*.pth")))
        assert len(ckpt_paths) >= cfg.eval.num_ckpts_to_eval
        ckpt_paths = list(sorted(ckpt_paths))[-cfg.eval.num_ckpts_to_eval :]
        assert f"{cfg.eval.last_ckpt}" in ckpt_paths[-1]
    else:
        ckpt_paths = [cfg.training.ckpt]

    rew_list = []

    for ckpt_path in ckpt_paths:
        ckpt_name = ckpt_path.split("/")[-1].split(".")[0]
        agent.load_snapshot(ckpt_path)

        log_dir = os.getcwd()

        eval_metrics = eval_fn(
            env,
            agent,
            vis=True,
            num_episodes=cfg.training.num_eval_episodes,
            log_dir=log_dir,
            reduce_horizon_dim=cfg.data.dataset.reduce_horizon_dim,
            verbose=True,
            ckpt_name=ckpt_name,
        )
        mean_rew = eval_metrics["rew"]
        print(f"Evaluation results: mean rew = {mean_rew}")
        rew_list.append(mean_rew)
        if cfg.use_wandb:
            wandb.log(
                {"eval/" + k: v for k, v in eval_metrics.items() if k != "vis_rollout"}
            )
        else:
            save_filename = os.path.join(
                os.getcwd(), f"vis_{ckpt_name}_rew{mean_rew:.3f}.mp4"
            )
        if "vis_rollout" in eval_metrics:
            if len(eval_metrics["vis_rollout"].shape) == 4:
                save_video(eval_metrics["vis_rollout"], save_filename, fps=30)
            else:
                assert len(eval_metrics["vis_rollout"][0].shape) == 4
                for eval_idx, eval_video in enumerate(eval_metrics["vis_rollout"]):
                    episode_rew = eval_metrics["rew_values"][eval_idx]
                    save_filename = os.path.join(
                        os.getcwd(),
                        f"vis_{ckpt_name}_ep{eval_idx}_rew{episode_rew:.3f}.mp4",
                    )
                    save_video(eval_video, save_filename)
        del eval_metrics
    np.savez(os.path.join(os.getcwd(), "info.npz"), rews=np.array(rew_list))


if __name__ == "__main__":
    main()
