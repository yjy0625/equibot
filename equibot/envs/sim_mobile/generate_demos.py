"""
Demonstration generation for simulated mobile robot envs.

@yjy0625, @contactrika

"""

import os
import sys
import cv2
import logging
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

from equibot.envs.sim_mobile.utils.anchors import create_trajectory
from equibot.envs.sim_mobile.utils.multi_camera import MultiCamera, get_camera_info
from equibot.envs.sim_mobile.utils.project import unproject_depth
from equibot.envs.sim_mobile.utils.init_utils import rotate_around_z
from equibot.policies.utils.media import add_caption_to_img, save_video, save_image
from equibot.policies.utils.misc import get_env_class


np.set_printoptions(precision=2, linewidth=150, threshold=10000, suppress=True)


def plan_actions_from_sketch(
    init_anchor_pos,
    sketch,
    initial_gripper_pose,
    sim_frequency,
    num_sec_per_unit=2.0,
):
    """
    Given a sketch for where manipulation anchors should move to, plan a
    sequence of actions.

    Args:
        init_anchor_pos: initial anchor position
        sketch: a list of tuple; each tuple contains 2 items, that describe
                the desired x and y coordinates of the anchor
        sim_frequency: simulation frequency
        num_sec_per_unit: how much time to take for traversing distance with
                          length equal to 1 unit

    Returns:
        actions: a list of actions to be executed in the environment
    """
    # create buffer for current anchor attach positions
    curr_anchor_pos = init_anchor_pos.copy()[:, :3]

    # init buffer for generated actions and desired positions
    num_anchors = len(sketch[0])
    ac_dim = 7 * num_anchors
    actions = np.zeros([0, ac_dim])

    prev_grip_ac = initial_gripper_pose

    # loop through each item in the sketch and generate actions
    # each action speficies the desired position and velocity change
    # for each anchor being manipulated
    for i, anchor_targets in enumerate(sketch):
        anchor_actions = []
        for j, anchor_target in enumerate(anchor_targets):
            grip_ac, target_x, target_y, target_z = anchor_target
            # compute waypoints and number of steps needed between them
            num_steps = int(num_sec_per_unit * sim_frequency)
            waypoints = [
                curr_anchor_pos[j],
                np.array([target_x, target_y, target_z]),
            ]
            waypoint_dists = [
                np.linalg.norm(waypoints[i + 1] - waypoints[i])
                for i in range(len(waypoints) - 1)
            ]
            waypoint_steps = [
                max(int(waypoint_dist * num_steps), 1)
                for waypoint_dist in waypoint_dists
            ] + [0]

            # plan actions
            anchor_j_actions = create_trajectory(
                waypoints, waypoint_steps, sim_frequency
            )
            anchor_j_actions = np.concatenate(
                [
                    np.ones_like(anchor_j_actions[:, [0]]) * prev_grip_ac[j],
                    anchor_j_actions,
                ],
                axis=1,
            )
            anchor_actions.append(anchor_j_actions)

        # fill no-op actions for shorter sequences
        # and append actions for all anchors
        max_len = np.max([len(acs) for acs in anchor_actions])
        for j in range(len(anchor_actions)):
            curr_len = len(anchor_actions[j])
            if curr_len < max_len:
                ac_dim = len(anchor_actions[j][0])
                noop = np.concatenate([anchor_targets[j], np.zeros(3)])
                anchor_actions[j] = np.concatenate(
                    [anchor_actions[j], np.array([noop] * (max_len - curr_len))]
                )
            anchor_actions[j][-1][0] = anchor_targets[j][0]

        actions = np.concatenate([actions, np.concatenate(anchor_actions, axis=1)])

        # update current anchor and attach positions
        prev_grip_ac = np.array([x[0] for x in anchor_targets])
        curr_anchor_pos = [np.array([x[1], x[2], x[3]]) for x in anchor_targets]

    return actions


def rotate_sketch(sketch, ang):
    sketch = np.array(sketch)  # (T, E, 4)
    gripper_ac, eef_pos = sketch[..., [0]], sketch[..., 1:]  # TE1, TE3
    eef_pos = rotate_around_z(eef_pos.reshape(-1, 3), ang).reshape(eef_pos.shape)
    sketch = np.concatenate([gripper_ac, eef_pos], axis=-1)
    sketch = [[tuple(ss) for ss in s] for s in sketch]
    return sketch


def run_demo(args, counter=0):
    # setup directories used for saving info
    os.makedirs(os.path.join(args.data_out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.data_out_dir, "pcs"), exist_ok=True)
    prefix = args.data_out_dir.split("/")[-1]
    episode_name = f"{prefix}_ep{counter:06d}"
    saved_files = []

    # seeding
    np.random.seed(args.seed)

    seed_env = args.seed_env if args.seed_env is not None else args.seed
    seed_cam = args.seed_cam if args.seed_cam is not None else args.seed
    rng_env = np.random.RandomState(seed_env)
    rng_cam = np.random.RandomState(seed_cam)

    # modify env args according to env
    if args.task_name == "fold":
        args.deform_elastic_stiffness = 150.0
        args.deform_friction_coeff = 1.0
    elif args.task_name == "cover":
        args.deform_elastic_stiffness = 300.0
        args.deform_friction_coeff = 10.0

    # create simulation env
    env = get_env_class(args.task_name)(args, rng_env)

    # get initial positions of anchors
    obs = env.reset()

    # plan actions
    wait_steps = 0
    init_gripper_pose_val = 0
    if args.task_name == "fold":
        init_gripper_pose_val = 1
        xy_scale = env._soft_object_scale
        xx, yy = xy_scale[0], xy_scale[1]
        zz = 0.2 * yy
        sketch = [
            [(1, 0.2 * xx, 0.0 * yy, zz), (1, -0.2 * xx, 0.0 * yy, zz)],
            [(0, 0.2 * xx, -0.2 * yy, 0.02), (0, -0.2 * xx, -0.2 * yy, 0.02)],
        ]
    elif args.task_name == "cover":
        init_gripper_pose_val = 1
        xy_scale = env._soft_object_scale
        xx, yy = xy_scale[0], xy_scale[1]
        rigid_z_scale = env._rigid_object_scale[-1]
        sketch = [
            [
                (1, -0.2 * xx, -0.1 * yy, 0.25 * rigid_z_scale),
                (1, 0.2 * xx, -0.1 * yy, 0.25 * rigid_z_scale),
            ],
            [
                (1, -0.2 * xx, 0.15 * yy, 0.25 * rigid_z_scale),
                (1, 0.2 * xx, 0.15 * yy, 0.25 * rigid_z_scale),
            ],
            [(0, -0.2 * xx, 0.15 * yy, 0.01), (0, 0.2 * xx, 0.15 * yy, 0.01)],
        ]
    elif args.task_name == "close":
        L, W, H = env._box_size
        T = env._box_thickness
        sketch = [
            [(0, -L * 0.4, W * 0.0, H + L), (0, L * 0.4, W * 0.0, H + L)],
            [(0, -L * 0.8, W * 0.7, H + L / 2), (0, L * 0.8, W * 0.7, H + L / 2)],
            [(0, -L * 0.8, W * 1.8, H * 0.8), (0, L * 0.8, W * 1.8, H * 0.8)],
            [(0, -L * 0.4, W * 1.8, H * 0.8), (0, L * 0.4, W * 1.8, H * 0.8)],
            [
                (0, -L * 0.4, W * 0.3, H + W * 1.5),
                (0, L * 0.4, W * 0.3, H + W * 1.5),
            ],
        ]
    else:
        raise ValueError(f"Task name {args.task_name} not found.")
    sketch = rotate_sketch(sketch, env._object_rotation[-1])

    # create buffers for episode info
    imgs = []

    # execute the planned actions
    t = 0
    record_t = 0
    cam_dist = args.cam_dist
    sim_unstable = False
    num_skip_steps = 0
    for step_idx, step in enumerate(tqdm(sketch)):
        if sim_unstable:
            break
        if step_idx == 0:
            initial_gripper_pose = (
                np.array([x[0] for x in sketch[0]]) * init_gripper_pose_val
            )
        else:
            initial_gripper_pose = np.array([x[0] for x in sketch[step_idx - 1]])
        sketch_step = [step]
        sim_freq = env.freq
        num_sec_per_unit = 20.0 / args.speed_multiplier
        actions = plan_actions_from_sketch(
            obs,
            sketch_step,
            initial_gripper_pose,
            sim_freq,
            num_sec_per_unit=num_sec_per_unit,
        )
        print(f"Length of actions: {len(actions)}")

        for step_t, action in enumerate(actions):
            if sim_unstable:
                break

            action = action.reshape(-1, 7).copy()
            grip_actions = action[:, [0]]
            expected_eef = action[:, 1:4]
            eef_actions = (action[:, 1:4] - obs[:, :3]) * env.freq
            eef_actions = np.clip(eef_actions, -1.0, 1.0)
            action = np.concatenate([grip_actions, eef_actions], axis=-1).flatten()
            if args.dof == 7:
                action = np.concatenate(
                    [action.reshape(-1, 4), np.zeros((len(action) // 4, 3))], axis=-1
                )

            should_record = (
                t % args.cam_rec_interval == 0
                if args.cam_rec_interval > 0
                else (step_t == len(actions) - 1 or (step_idx == 0 and step_t == 0))
            )
            if should_record:
                yaws, pitches = [], []

                def sample_vals(s):
                    if len(s) == 2:
                        return (
                            rng_cam.rand(args.cam_num_views) * (s[1] - s[0]) + s[0]
                        )
                    elif len(s) == 1:
                        return s
                    else:
                        raise ValueError("Length of {s} should be 1 or 2.")

                for y in sample_vals(args.cam_yaws):
                    for p in sample_vals(args.cam_pitches):
                        yaws.append(y)
                        pitches.append(p)
                num_views = len(yaws)
                cam_info = {
                    "yaws": yaws,
                    "pitches": pitches,
                    "dist": cam_dist * np.max(env._soft_object_scale),
                    "views": list(np.arange(num_views)),
                    "fov": 30,
                    "width": 240,
                    "height": 240,
                }
                H, W = cam_info["height"], cam_info["width"]
                render_dict = env.render(
                    cam_info=cam_info,
                    return_depth=True,
                    return_seg=True,
                    hide_eef=True,
                )
                view_images = render_dict["images"]
                view_depths = render_dict["depths"]
                view_segs = render_dict["segs"]

                # get mesh data of the object of interest
                soft_obj_ids = (
                    env.soft_obj_ids
                    if hasattr(env, "soft_obj_ids")
                    else env.soft_ids
                )
                rigid_obj_ids = (
                    env.rigid_obj_ids
                    if hasattr(env, "rigid_obj_ids")
                    else env.rigid_ids
                )
                cam_target = (
                    env._cam_target
                    if hasattr(env, "_cam_target")
                    else env.camera_config["target"]
                )
                obj_ids = soft_obj_ids[:1] + rigid_obj_ids
                mesh_xyzs_list = [
                    env.sim.getMeshData(obj_id)[1] for obj_id in soft_obj_ids[:1]
                ]
                mesh_xyzs_list += [
                    env._get_rigid_body_mesh(obj_id) for obj_id in rigid_obj_ids
                ]
                mesh_idxs = np.concatenate(
                    [np.full((len(x),), i) for i, x in enumerate(mesh_xyzs_list)]
                )
                mesh_xyzs = np.concatenate(mesh_xyzs_list)
                cam_vals = MultiCamera.get_cam_vals(
                    [0] * num_views,
                    yaws,
                    pitches,
                    cam_info["dist"],
                    cam_target,
                    cam_info["fov"],
                    float(W / H),
                )

                for img_ix, img in enumerate(view_images):
                    img_name = (
                        f"{prefix}_ep{counter:06d}_view{img_ix}_t{record_t:02d}"
                    )

                    # process images
                    img = img[..., :3]

                    # compute segmentation
                    segs = view_segs[img_ix]
                    seg = np.isin(segs, obj_ids).astype(np.uint8) * 255
                    object_pixels = np.array(np.where(seg == 255)).T
                    seg = np.repeat(seg[:, :, None], 3, axis=-1)

                    # save projected mesh vertex pixel locations
                    # reference: https://stackoverflow.com/questions/60430958
                    view_mat, p_proj_mat = cam_vals[img_ix][:2]
                    view_mat = np.array(view_mat).reshape((4, 4), order="C")
                    p_proj_mat = np.array(p_proj_mat).reshape((4, 4), order="C")
                    cx = (1 - p_proj_mat[0, 2]) * W / 2
                    cy = (p_proj_mat[1, 2] + 1) * H / 2
                    proj_mat = np.array(
                        [
                            [-p_proj_mat[0, 0] * W / 2, 0, cx],
                            [0, p_proj_mat[1, 1] * H / 2, cy],
                            [0, 0, 1],
                        ]
                    )

                    # unproject segmentation mask to get partial PC
                    view_depth = view_depths[img_ix].T
                    extrinsics = view_mat.copy().T
                    extrinsics[[1, 2]] *= -1
                    extrinsics = np.linalg.inv(extrinsics)
                    intrinsics = proj_mat.copy()
                    intrinsics[0, 0] *= -1
                    partial_pc = unproject_depth(
                        [view_depth],
                        [intrinsics],
                        [extrinsics],
                        filter_pixels=[object_pixels],
                        clip_radius=10.0,
                    )
                    save_dir = os.path.join(args.data_out_dir, "pcs")
                    save_path = os.path.join(save_dir, f"{img_name}.npz")

                    if hasattr(env, "workspace_boundary"):
                        workspace_boundary = env.workspace_boundary
                    else:
                        workspace_boundary = (
                            np.array([-10, -10]),
                            np.array([10, 10]),
                        )
                    pc_out_of_bounds = np.any(
                        partial_pc[:, :2] < workspace_boundary[0][None]
                    ) or np.any(partial_pc[:, :2] > workspace_boundary[1][None])
                    if len(partial_pc) == 0:
                        sim_unstable = True
                        print(
                            f"Warning: simulation unstable; cutting episode short."
                        )
                        break
                    if (
                        pc_out_of_bounds
                        or np.min(partial_pc[:, 2]) < -0.05
                        or np.max(partial_pc[:, 2] > 1.0)
                    ):
                        # simulation unstable; do not save data
                        if not args.task_name.endswith("closing-v1"):
                            sim_unstable = True
                            print(
                                f"Warning: simulation unstable; cutting episode short."
                            )
                            break

                    # subsample pc if too large
                    num_points = 4096
                    if len(partial_pc) >= num_points:
                        sampled_indices = np.random.choice(
                            len(partial_pc), size=4096, replace=False
                        )
                        partial_pc = partial_pc[sampled_indices]

                    np.savez(
                        save_path,
                        pc=partial_pc,
                        rgb=img,
                        action=action,
                        eef_pos=obs,
                    )
                    saved_files.append(save_path)

                cam_info = get_camera_info(args)
                aux_cam_info = get_camera_info(args, aux=True)
                cam_info["dist"] *= np.max(env._soft_object_scale)
                aux_cam_info["dist"] *= np.max(env._soft_object_scale)
                img = env.render(cam_info=cam_info)["images"][0][..., :3]
                aux_img = env.render(cam_info=aux_cam_info)["images"][0][..., :3]
                imgs.append(np.concatenate([img, aux_img], axis=1))

                record_t += 1

            obs, _, _, _ = env.step(action, dummy_reward=True)

            t += 1

    final_rew = env.compute_reward()
    print(f"Episode reward: {final_rew}")

    if final_rew >= args.data_rew_threshold:
        # write video to file
        video_path = os.path.join(args.data_out_dir, "images", episode_name + ".mp4")
        save_video(np.array(imgs), video_path, fps=10)
        print(f"Saved video to {video_path}.")
        return 1
    else:
        for f in saved_files:
            os.remove(f)
        return 0


def get_args(parent=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Demo Generation", add_help=False)

    # Main/demo args.
    parser.add_argument(
        "--task_name", type=str, default="fold", help="Name of the task"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--seed_env", type=int, default=666, help="Random seed for environment"
    )
    parser.add_argument(
        "--seed_cam", type=int, default=66666, help="Random seed for camera"
    )
    # Simulation args. Note: turn up frequency when deform stiffness is high.
    parser.add_argument(
        "--sim_frequency",
        type=int,
        default=500,
        help="Number of simulation steps per second",
    )  # 250-1K
    parser.add_argument("--dof", type=int, default=7, help="Action dim")
    parser.add_argument("--num_eef", type=int, default=2, help="Number of end-effectors")
    parser.add_argument("--max_episode_length", type=int, default=50, help="Max episode length")
    parser.add_argument("--ac_noise", type=float, default=0.02, help="Action noise")
    # deform/SoftBody obj args.
    parser.add_argument("--randomize_rotation", action="store_true")
    parser.add_argument("--randomize_scale", action="store_true")
    parser.add_argument("--uniform_scaling", action="store_true")
    parser.add_argument(
        "--deform_bending_stiffness",
        type=float,
        default=0.01,
        help="deform spring elastic stiffness (k)",
    )  # 1.0-300.0
    parser.add_argument(
        "--deform_damping_stiffness",
        type=float,
        default=1.0,
        help="deform spring damping stiffness (c)",
    )
    parser.add_argument(
        "--deform_elastic_stiffness",
        type=float,
        default=300.0,
        help="deform spring elastic stiffness (k)",
    )  # 1.0-300.0
    parser.add_argument(
        "--deform_friction_coeff",
        type=float,
        default=10.0,
        help="deform friction coefficient",
    )
    # Camera args.
    parser.add_argument(
        "--cam_resolution", type=int, default=240, help="Point cloud resolution"
    )
    parser.add_argument(
        "--cam_rec_interval",
        type=int,
        default=1,
        help="How many steps to skip between each cam shot",
    )
    parser.add_argument(
        "--cam_num_views", type=int, default=1, help="Number of views to sample."
    )
    parser.add_argument("--vis", action="store_true")
    # Data generation.
    parser.add_argument("--num_demos", type=int, default=1)
    parser.add_argument("--data_out_dir", type=str, default=None)
    parser.add_argument("--data_rew_threshold", type=float, default=0.9)
    parser.add_argument("--cam_pitches", type=int, nargs="*", default=[-90])
    parser.add_argument("--cam_yaws", type=int, nargs="*", default=[0])
    parser.add_argument("--cam_fov", type=int, default=45)
    parser.add_argument("--cam_dist", type=float, default=1.0)
    parser.add_argument("--speed_multiplier", type=float, default=1.0)

    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    # read args
    args, _ = get_args()
    seed = args.seed
    seed_env = args.seed_env
    seed_cam = args.seed_cam

    pattern_ix = 0
    for i in tqdm(range(args.num_demos), desc="Demos"):
        while True:
            args.seed = (seed * 99999 + pattern_ix) % 100001
            args.seed_env = (seed_env * 99999 + pattern_ix) % 100001
            args.seed_cam = (seed_cam * 99999 + pattern_ix) % 100001
            success = run_demo(args, pattern_ix)
            pattern_ix += 1
            if success:
                break


if __name__ == "__main__":
    main()
