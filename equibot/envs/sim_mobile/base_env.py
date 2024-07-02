import os
import gym
import time
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bclient
import pybullet_data
from scipy.spatial.transform import Rotation

from equibot.envs.sim_mobile.utils.transformations import quat_multiply, axisangle2quat, quat2mat
from equibot.envs.sim_mobile.utils.frame_converter import wrap_angle
from equibot.envs.sim_mobile.utils.convert_coords import (
    global_to_local_pose,
    local_to_global_pose,
)
from equibot.envs.sim_mobile.utils.bullet_robot import BulletRobot
from equibot.envs.sim_mobile.utils.info import SIM_ROBOT_INFO, DEFAULT_ANCHOR_CONFIG
from equibot.envs.sim_mobile.utils.render import add_debug_region
from equibot.envs.sim_mobile.utils.plan_control_traj import plan_control_traj
from equibot.envs.sim_mobile.utils.anchors import get_closest
from equibot.envs.sim_mobile.utils.init_utils import (
    load_rigid_object,
    load_soft_object,
    rotate_around_z,
)
from equibot.envs.sim_mobile.utils.multi_camera import MultiCamera
from equibot.envs.sim_mobile.utils.project import unproject_depth


class BaseEnv(object):
    ARM_MOUNTING_HEIGHT = 0.335
    BASE_VEL_THRESH = 0.05
    SIM_FREQ = 360
    SIM_GRAVITY = -9.8
    SIM_ARM_QPOS_ERR_THRESH = 0.001  # sim movement error threshold for arms

    def __init__(self, args, rng=None, rng_act=None):
        self.args = args
        self.num_eef = args.num_eef
        self.dof = args.dof
        self.max_episode_length = args.max_episode_length
        self.rng = rng if rng is not None else np.random.RandomState(args.seed)
        self.rng_act = rng_act if rng_act is not None else rng
        self.base_diff_thresh = (0.015, 0.015, 0.02)
        self.debug = False
        self._last_action_time = None

        self._init_robots()
        self._init_rendering()

    # Init a dummy observation_space and action_space for vectorized env
    @property
    def action_space(self):
        return gym.spaces.Box(  # dummy action space
            low=-1.0, high=1.0, shape=(self.args.num_eef, 7), dtype=np.float32
        )

    @property
    def observation_space(self):
        return gym.spaces.Box(  # dummy observation space
            low=-1.0, high=1.0, shape=(self.args.num_eef, 13), dtype=np.float32
        )

    def _init_robots(self):
        args = self.args

        self._rigid_object_scale = np.array([0.0, 0.0, 0.0])
        self._soft_object_scale = np.array([0.0, 0.0])
        self._object_rotation = np.array([0.0, 0.0, 0.0])
        self.scene_offset = np.array([0.0, 0.0])

        self.constraint_ids = [None] * len(self.robot_config)
        self.randomize_scale = args.randomize_scale
        self.randomize_rotation = args.randomize_rotation
        self.randomize_position = (
            args.randomize_position if hasattr(args, "randomize_position") else False
        )
        if self.randomize_position:
            self.rand_pos_scale = args.rand_pos_scale
        self.uniform_scaling = args.uniform_scaling
        self.ac_noise = args.ac_noise
        self.vis = args.vis
        self.freq = args.freq if hasattr(args, "freq") else 5
        self.flip_agents = False
        self.xy_action_scale = 1.0
        self.z_action_scale = 1.0
        self._init_base_rot = [item["rest_base_pose"][2] for item in self.robot_config]
        self._init_arm_pos = np.array(
            [item["rest_arm_pos"] for item in self.robot_config]
        )
        self.base_diff_thresh = (0.001, 0.001, 0.01)
        self._done = None
        self.arm_only = False

    def _reset_robots(self):
        robots = []
        self._init_base_rot = []
        for robot_config_item in self.robot_config:
            robot_info = SIM_ROBOT_INFO[robot_config_item["sim_robot_name"]]
            robot_path = os.path.join(self.data_path, "robots", robot_info["file_name"])
            base_pos = [*robot_config_item["rest_base_pose"][0:2], 0]
            base_rot = robot_config_item["rest_base_pose"][2]
            base_quat = pybullet.getQuaternionFromEuler([0, 0, base_rot])
            global_target_ee_pos, _, global_target_ee_quat = local_to_global_pose(
                robot_config_item["rest_arm_pos"],
                robot_config_item["rest_arm_rot"],
                base_xy=base_pos[0:2],
                base_rot=base_rot,
                height_offset=BaseEnv.ARM_MOUNTING_HEIGHT,
                debug=self.debug,
            )
            robot = BulletRobot(
                self.sim,
                robot_path,
                control_mode="position",
                ee_joint_name=robot_info["ee_joint_name"],
                ee_link_name=robot_info["ee_link_name"],
                base_pos=base_pos,
                base_quat=base_quat,
                global_scaling=1,
                use_fixed_base=False,
                use_fixed_arm=False,
                rest_arm_pos=global_target_ee_pos,
                rest_arm_quat=global_target_ee_quat,
                debug=False,
            )
            if self.debug:  # print local-global coordinate conversion
                global_ee_pos, global_ee_quat, *_ = robot.get_ee_pos_quat_vel()
                global_ee_ori_deg = (
                    np.array(pybullet.getEulerFromQuaternion(global_ee_quat))
                    / np.pi
                    * 180
                )
                print(
                    "Loaded sim robot with EE global_pos",
                    global_ee_pos,
                    "global_ee_ori deg",
                    global_ee_ori_deg,
                )
                local_ee_pos, local_ee_ori, _ = global_to_local_pose(
                    global_ee_pos,
                    global_ee_quat,
                    base_pos[0:2],
                    base_pos[2],
                    self.ARM_MOUNTING_HEIGHT,
                )
                print(
                    "Relative/local EE pos",
                    local_ee_pos,
                    "local_ee_ori deg",
                    local_ee_ori / np.pi * 180,
                )
            robots.append(robot)
            self._init_base_rot.append(robot_config_item["rest_base_pose"][-1])
        return robots

    def _init_rendering(self):
        self.camera_config = self.default_camera_config

    def init_vis_anchors(self):
        anchor_ids = []
        for anchor_info in self.anchor_config:
            radius, rgba = anchor_info["radius"], anchor_info["rgba"]
            mass, pos = 0.0, anchor_info["pos"]
            anchorVisualShape = self.sim.createVisualShape(
                pybullet.GEOM_SPHERE, radius=radius, rgbaColor=rgba
            )
            anchor_id = self.sim.createMultiBody(
                baseMass=mass,
                basePosition=pos,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=anchorVisualShape,
                useMaximalCoordinates=True,
            )
            anchor_ids.append(anchor_id)
        return anchor_ids

    def init_eef_frame_anchors(self):
        anchor_ids = []

        for ax in range(len(self.robots) * 3):
            anchor_info = {
                "radius": 0.01,
                "rgba": [0.0, 0.0, 0.0, 1.0],
                "pos": [10, 0, -10],
            }
            anchor_info["rgba"][ax % 3] = 1.0
            for i in range(10):
                radius, rgba = anchor_info["radius"], anchor_info["rgba"]
                mass, pos = 0.0, anchor_info["pos"]
                anchorVisualShape = self.sim.createVisualShape(
                    pybullet.GEOM_SPHERE, radius=radius, rgbaColor=rgba
                )
                anchor_id = self.sim.createMultiBody(
                    baseMass=mass,
                    basePosition=pos,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=anchorVisualShape,
                    useMaximalCoordinates=True,
                )
                anchor_ids.append(anchor_id)
        return anchor_ids

    def init_pc_anchors(self):
        anchor_ids = []
        anchor_info = {
            "radius": 0.01,
            "rgba": [0.918, 0.263, 0.208, 1.0],
            "pos": [10, 0, -10],
        }
        for i in range(512):
            radius, rgba = anchor_info["radius"], anchor_info["rgba"]
            mass, pos = 0.0, anchor_info["pos"]
            anchorVisualShape = self.sim.createVisualShape(
                pybullet.GEOM_SPHERE, radius=radius, rgbaColor=rgba
            )
            anchor_id = self.sim.createMultiBody(
                baseMass=mass,
                basePosition=pos,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=anchorVisualShape,
                useMaximalCoordinates=True,
            )
            anchor_ids.append(anchor_id)
        return anchor_ids

    def visualize_anchor(self, positions):
        for i, pos in enumerate(positions):
            self.sim.resetBasePositionAndOrientation(
                self.anchor_ids[i], pos, [0, 0, 0, 1]
            )

    def visualize_eef_frame(self, states):
        states = states.reshape(len(self.robots), -1)
        for i, state in enumerate(states):
            eef_pos = state[:3]
            eef_dir_x = state[3:6]
            eef_dir_z = state[6:9]
            eef_dir_y = np.cross(eef_dir_z, eef_dir_x)
            for j in range(10):
                self.sim.resetBasePositionAndOrientation(
                    self.eef_frame_anchor_ids[30 * i + j],
                    eef_pos + eef_dir_x * 0.01 * (j + 1),
                    [0, 0, 0, 1],
                )
                self.sim.resetBasePositionAndOrientation(
                    self.eef_frame_anchor_ids[30 * i + 10 + j],
                    eef_pos + eef_dir_y * 0.01 * (j + 1),
                    [0, 0, 0, 1],
                )
                self.sim.resetBasePositionAndOrientation(
                    self.eef_frame_anchor_ids[30 * i + 20 + j],
                    eef_pos + eef_dir_z * 0.01 * (j + 1),
                    [0, 0, 0, 1],
                )

    def visualize_pc(self, pc):
        if not hasattr(self, "pc_anchor_ids"):
            return
        idxs = np.random.randint(0, len(pc), 512)
        sampled_pc = pc[idxs]
        for i, pos in enumerate(sampled_pc):
            self.sim.resetBasePositionAndOrientation(
                self.pc_anchor_ids[i], pos, [0, 0, 0, 1]
            )

    @property
    def data_path(self):
        return os.path.join(os.path.dirname(__file__), "assets")

    @property
    def default_camera_config(self):
        cfg = {
            "pitch": -40,  # ATTENTION: -90 causes blank pybullet gui
            "roll": 0,
            "yaw": 0,
            "distance": 5,
            "fov": 30,
            "target": [0, 0, 0.2],
            "aspect_ratio": 1.0,
        }
        return cfg

    @property
    def base_tgt_lows(self):
        return (0.0, -0.8)

    @property
    def base_tgt_highs(self):
        return (1.7, 1.2)

    @property
    def obs_xy_lows(self):
        return (-0.2, -1.2)

    @property
    def obs_xy_highs(self):
        return (2.05, 1.65)

    @property
    def robot_config(self):
        raise NotImplementedError()

    @property
    def anchor_config(self):
        return []

    @property
    def rigid_objects(self):
        return []

    @property
    def soft_objects(self):
        return []

    def compute_reward(self):
        raise NotImplementedError()

    @property
    def base_steps(self):
        return BaseEnv.SIM_FREQ // self.freq

    @property
    def arm_steps(self):
        return BaseEnv.SIM_FREQ // self.freq

    @property
    def name(self):
        return "base"

    def reset(self, dummy_obs=False):
        start = time.time()
        if self.debug:
            print(f"[base env] start of reset")
        self._t = 0
        self._internal_t = 0
        self._frames = []
        self._episode_reward = 0.0
        self._ac_noise_multiplier = self.rng.rand()
        self._randomize_object_scales()
        self._reset_sim()
        self._init_rendering()
        obs = self._get_obs()
        self._done = False

        if self.debug:
            print(f"[base env] end of reset")
            print(f"[reset] {time.time() - start:.3f}s")
        return obs

    def step(self, action, dummy_reward=False, dummy_obs=False):
        st = time.time()

        state_dict = self._compute_global_ee_poses()
        if state_dict is None:
            return None, 0.0, True, {}

        action = action.reshape(-1, self.dof)

        # add action noise if required
        if self.dof >= 4:
            start_ix = 1
        else:
            start_ix = 0
        for i in range(len(action)):
            action[i, start_ix : start_ix + 3] += (
                self.rng.randn(3)
                * (self.ac_noise * self._ac_noise_multiplier)
                # * np.linalg.norm(action[i, start_ix:start_ix + 3])
            )

        # compute target offsets
        target_pos_vels = action.reshape(-1, self.dof)[:, 1:4]
        if self.flip_agents:
            target_pos_vels = target_pos_vels[::-1]

        target_offsets = target_pos_vels / self.freq
        st = time.time()

        # compute target base pose
        if not self.arm_only:
            target_base_poses = state_dict["base_poses"].copy()
            target_base_poses[:, -1] = self._init_base_rot
            target_base_poses[:, :2] += target_offsets[:, :2]
        else:
            target_base_poses = self._init_base_pose
        st = time.time()

        # compute target arm velocity
        if not self.arm_only:
            target_arm_pos_vels = target_pos_vels.copy()
            target_arm_pos_vels[:, :2] = 0
        else:
            target_arm_pos_vels = target_pos_vels.copy()
            for i in range(len(self.robots)):
                target_arm_pos_vels[i] = rotate_around_z(
                    target_arm_pos_vels[i], np.pi - state_dict["base_poses"][i, -1]
                )
        if self.dof == 4:
            target_arm_ori_vels = None
        else:
            target_arm_ori_vels = action.reshape(-1, self.dof)[:, -3:]
            for i in range(len(self.robots)):
                target_arm_ori_vels[i] = rotate_around_z(
                    target_arm_ori_vels[i], np.pi - state_dict["base_poses"][i, -1]
                )
            target_arm_ori_vels *= 180 / np.pi
        st = time.time()

        # take gripper action
        grip_actions = action.reshape(-1, self.dof)[:, 0].flatten()
        self._move_grippers(grip_actions)
        st = time.time()

        # move arms and bases in one step
        self._move_bases_and_arms(
            target_base_poses,
            target_arm_pos_vels,
            target_arm_ori_vels,
            duration=1.0 / self.freq,
            wait=False,
        )
        st = time.time()

        # sleep
        if self._last_action_time is not None:
            elapsed_time = time.time() - self._last_action_time
            target_time = 1.0 / self.freq
            if elapsed_time <= target_time:
                time.sleep(max(0, target_time - elapsed_time))
        self._last_action_time = time.time()
        st = time.time()

        # get observation
        state = self._get_obs(dummy_obs=dummy_obs)
        st = time.time()

        # get return info
        self._t += 1
        st = time.time()
        rew = 0.0 if dummy_reward else self.compute_reward()
        done = self._t >= self.max_episode_length
        st = time.time()

        return state, rew, done, {}

    def _sample_scale(self, low, high, size, aspect_limit=None):
        while True:
            scale = self.rng.rand(size) * (high - low) + low
            if aspect_limit is None or scale.max() / scale.min() < aspect_limit:
                break
        return scale

    def _randomize_object_scales(self):
        # initialize object randomizations
        if self.randomize_scale:
            scale_low, scale_high = self.args.scale_low, self.args.scale_high
            aspect_limit = self.args.scale_aspect_limit
            sampled_scale = self._sample_scale(scale_low, scale_high, 5, aspect_limit)
            self._rigid_object_scale = sampled_scale[:3]
            self._soft_object_scale = sampled_scale[3:]
            if self.uniform_scaling:
                self._rigid_object_scale[:] = self._soft_object_scale[0]
                self._soft_object_scale[:] = self._soft_object_scale[0]
        else:
            self._rigid_object_scale = np.array([1.0, 1.0, 1.0])
            self._soft_object_scale = np.array([1.0, 1.0])
        if self.randomize_rotation:
            self._object_rotation = [0.0, 0.0, self.rng.rand() * np.pi * 2]
        else:
            self._object_rotation = [0.0, 0.0, 0.0]
        if self.randomize_position:
            self.scene_offset = self.rng.randn(2) * self.rand_pos_scale
        else:
            self.scene_offset = np.zeros((2,))

    def _reset_sim(self):
        if not hasattr(self, "sim"):
            sim = self.sim = bclient.BulletClient(connection_mode=pybullet.DIRECT)
        else:
            sim = self.sim
        args = self.args

        # Reset simulator to clear out deformables (no other way to do this).
        sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
        sim.setGravity(0, 0, BaseEnv.SIM_GRAVITY)
        sim.setTimeStep(1.0 / BaseEnv.SIM_FREQ)

        sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        floor_id = self.sim.loadURDF("plane.urdf")

        sim.setAdditionalSearchPath(self.data_path)
        texture_id = sim.loadTexture("textures/wood.jpg")
        sim.changeVisualShape(
            floor_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id
        )

        # add rigid objects to the scene
        self.rigid_ids = []
        if args.vis:
            print(f"Will load {len(self.rigid_objects):d} objects...")
        self._rigid_graspable = []
        self._rigid_objects = self.rigid_objects
        for idx, rigid_obj_info in enumerate(self._rigid_objects):
            self._rigid_graspable.append(
                "graspable" in rigid_obj_info and rigid_obj_info["graspable"]
            )
            full_obj_path = os.path.join(self.data_path, rigid_obj_info["path"])
            kwargs = {}
            if "mass" in rigid_obj_info:
                kwargs["mass"] = rigid_obj_info["mass"]
            rigid_id = load_rigid_object(
                sim,
                full_obj_path,
                rigid_obj_info["scale"],
                rigid_obj_info["pos"],
                rigid_obj_info["orn"],
                **kwargs,
            )
            self.rigid_ids.append(rigid_id)

        # add soft objects to the scene
        self.soft_ids = []
        self._soft_objects = self.soft_objects
        for soft_obj_info in self._soft_objects:
            full_obj_path = os.path.join(self.data_path, soft_obj_info["path"])
            print(f"soft object path: {full_obj_path}")
            soft_id = load_soft_object(
                sim,
                full_obj_path,
                soft_obj_info["scale"],
                soft_obj_info["pos"],
                soft_obj_info["orn"],
                args.deform_bending_stiffness,
                args.deform_damping_stiffness,
                args.deform_elastic_stiffness,
                args.deform_friction_coeff,
                mass=soft_obj_info["mass"],
                collision_margin=soft_obj_info["collision_margin"],
                fuzz_stiffness=False,
                use_self_collision=False,
            )
            self.soft_ids.append(soft_id)

        sim.stepSimulation()

        self.robots = self._reset_robots()

        self.constraint_ids = [None] * len(self.robots)
        self.anchor_ids = self.init_vis_anchors()
        self.eef_frame_anchor_ids = self.init_eef_frame_anchors()
        if hasattr(self.args, "visualize_pc") and self.args.visualize_pc:
            self.pc_anchor_ids = self.init_pc_anchors()
        add_debug_region(self.sim, self.base_tgt_lows, self.base_tgt_highs)

        if self.vis:
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    def _get_base_poses(self):
        return np.array([r.get_base_pose() for r in self.robots])

    def _get_base_vels(self):
        res_vels = []
        for robot in self.robots:
            _, vel, _ = robot.get_base_pose_vel()
            res_vels.append(vel)
        return np.vstack(res_vels)

    def _get_local_ee_poses(self, base_poses):
        local_ee_poss, local_ee_oris = [], []
        for i, robot in enumerate(self.robots):
            global_ee_pos, global_ee_quat, *_ = robot.get_ee_pos_quat_vel()
            local_ee_pos, local_ee_ori, _ = global_to_local_pose(
                global_ee_pos,
                global_ee_quat,
                base_poses[i, 0:2],
                base_poses[i, 2],
                height_offset=BaseEnv.ARM_MOUNTING_HEIGHT,
                debug=False,
            )
            local_ee_poss.append(local_ee_pos)
            local_ee_oris.append(local_ee_ori)
        return np.array(local_ee_poss), np.array(local_ee_oris)

    def _move_grippers(self, grip_actions):
        for i, robot in enumerate(self.robots):
            if grip_actions[i] < 0.5:
                if self.debug:
                    print(f"[open gripper {i}]")
                robot.open_gripper()
                self._detach_sim_anchor(i)
            else:
                if self.debug:
                    print(f"[close gripper {i}]")
                robot.close_gripper()
                if i == 0:
                    ee_poss = self._get_obs()
                    self.grasp_ee_dist = np.linalg.norm(ee_poss[1] - ee_poss[0])
                self._attach_sim_anchor(i)

    def _move_bases_and_arms(
        self,
        target_base_poses,
        target_arm_pos_vels,
        target_arm_ori_vels=None,
        duration=1.0,
        wait=False,
    ):
        start0 = time.time()

        # NOTE: we assume that arm's position velocity only has height component
        assert np.all(target_arm_pos_vels[:, :2] == 0)
        base_poses = self._get_base_poses()
        local_ee_poss, local_ee_oris = self._get_local_ee_poses(base_poses)
        local_ee_poss[:, :2] = self._init_arm_pos[:, :2]
        target_ee_poss = local_ee_poss + target_arm_pos_vels / self.freq

        # clip arm heights so arm does not go into the ground plane
        target_ee_poss[:, -1] = np.maximum(
            target_ee_poss[:, -1], -self.ARM_MOUNTING_HEIGHT + 0.01
        )

        # compute target ee orientations
        if target_arm_ori_vels is None:
            target_ee_oris = local_ee_oris
        else:
            target_ee_oris = []
            for robot, base_pose, target_arm_ori_vel, local_ee_ori in zip(
                self.robots, base_poses, target_arm_ori_vels, local_ee_oris
            ):
                global_ee_pos, global_ee_quat, *_ = robot.get_ee_pos_quat_vel()
                target_global_ee_quat = quat_multiply(
                    axisangle2quat(target_arm_ori_vel), global_ee_quat
                )
                _, target_ee_ori, _ = global_to_local_pose(
                    global_ee_pos,
                    target_global_ee_quat,
                    base_pose[0:2],
                    base_pose[2],
                    self.ARM_MOUNTING_HEIGHT,
                )
                target_ee_ori = np.array(
                    [
                        wrap_angle(tgt, src)
                        for src, tgt in zip(local_ee_ori, target_ee_ori)
                    ]
                )
                target_ee_oris.append(target_ee_ori)
            target_ee_oris = np.array(target_ee_oris)

        curr_base_poses = self._get_base_poses()
        diff_thresh = np.array(self.base_diff_thresh)
        base_poses = self._get_base_poses()
        curr_base_vels = self._get_base_vels()

        target_base_trajs = []
        target_qposes = []

        for i, robot in enumerate(self.robots):
            # compute base targets
            target_ang = wrap_angle(target_base_poses[i][2], curr_base_poses[i][2])
            target_base_ori = np.array([0, 0, target_ang])
            curr_base_ori = np.array([0, 0, curr_base_poses[i][2]])
            pose_diff = np.abs(target_base_poses[i] - curr_base_poses[i])
            num_steps = int(self.base_steps)
            target_pos = np.concatenate([target_base_poses[i][:2], [0]])
            current_pos = np.concatenate([curr_base_poses[i][:2], [0]])
            target_base_xyz_traj, target_base_ori_traj = plan_control_traj(
                target_pos,
                target_base_ori,
                num_steps=num_steps,
                freq=BaseEnv.SIM_FREQ,
                curr_pos=current_pos,
                curr_quat=curr_base_ori,
                ori_type="euler",
            )
            target_base_xy_traj = target_base_xyz_traj[:, :2]
            target_rot_traj = target_base_ori_traj[:, [2]]
            target_base_traj = np.concatenate(
                [target_base_xy_traj, target_rot_traj], axis=1
            )
            target_base_trajs.append(target_base_traj)

            # compute arm targets
            if self.debug:
                print("[move arms] EE pos local -> global")
            global_target_ee_pos, _, global_target_ee_quat = local_to_global_pose(
                target_ee_poss[i],
                target_ee_oris[i],
                base_xy=base_poses[i][0:2],
                base_rot=base_poses[i][2],
                height_offset=BaseEnv.ARM_MOUNTING_HEIGHT,
                debug=self.debug,
            )
            target_qpos = robot.ee_pos_to_qpos(
                ee_pos=np.array(global_target_ee_pos),
                ee_quat=np.array(global_target_ee_quat),
            )
            curr_qpos = robot.get_qpos()

            num_steps = int(self.arm_steps)
            traj_progress = np.ones([num_steps])
            traj_progress[:num_steps] = np.linspace(0, 1, num_steps + 1)[1:]
            qpos_traj = (target_qpos - curr_qpos)[None] * traj_progress[
                :, None
            ] + curr_qpos
            target_qposes.append(qpos_traj)

        assert len(target_qposes[0]) == self.arm_steps  # check

        sim_step = 0
        completed_bases = set()
        completed_arms = set()
        move_start_time = time.time()
        while (
            len(completed_bases) < len(self.robots)
            or len(completed_arms) < len(self.robots)
        ) and sim_step < max(self.base_steps, self.arm_steps):
            start = time.time()

            for i, robot in enumerate(self.robots):
                # move arms
                curr_qpos = robot.get_qpos()
                if i in completed_arms:
                    robot.move_to_qpos(curr_qpos, mode=pybullet.POSITION_CONTROL)
                else:
                    st = sim_step if sim_step < len(target_qposes[i]) else -1
                    robot.move_to_qpos(
                        target_qposes[i][st], mode=pybullet.POSITION_CONTROL
                    )
                    arm_diff = np.abs(curr_qpos - target_qposes[i][-1])
                    if np.all(arm_diff < BaseEnv.SIM_ARM_QPOS_ERR_THRESH) and (
                        st == len(target_qposes[i]) - 1 or st == -1
                    ):
                        if self.debug:
                            print(
                                f"[move bases and arms] arm {i} done (step {sim_step})"
                            )
                        completed_arms.add(i)

                # move bases
                if i in completed_bases:
                    robot.move_base(curr_base_poses[i])
                else:
                    curr_base_pose = curr_base_poses[i]
                    target_pose = np.array(target_base_poses[i])
                    target_pose[2] = wrap_angle(target_pose[2], curr_base_pose[2])
                    target_diff = target_pose - curr_base_pose
                    pose_diff_ok = np.all(np.abs(target_diff) <= diff_thresh)
                    _, lin_vel, ang_vel = robot.get_base_pose_vel()
                    curr_vel = np.max(
                        [np.linalg.norm(lin_vel), np.linalg.norm(ang_vel)]
                    )
                    vel_is_small = np.all(np.abs(curr_vel) <= BaseEnv.BASE_VEL_THRESH)
                    st = sim_step if sim_step < len(target_base_trajs[i]) else -1
                    robot.move_base(target_base_trajs[i][st])
                    if (
                        pose_diff_ok
                        and vel_is_small
                        and (st == len(target_base_trajs[i]) or st == -1)
                    ):  # move done
                        completed_bases.add(i)
                        if self.debug:
                            print(
                                f"[move bases and arms] base {i} done (step {sim_step})"
                            )

            if len(completed_bases) < len(self.robots) or len(completed_arms) < len(
                self.robots
            ):
                self._step_simulation()
            sim_step += 1
            curr_base_poses = self._get_base_poses()  # update base poses

        if self.debug:
            global_ee_poss = []
            for robot in self.robots:
                global_ee_pos, global_ee_quat, *_ = robot.get_ee_pos_quat_vel()
                global_ee_poss.append(global_ee_pos)
            global_ee_poss = np.array(global_ee_poss)
            local_ee_poss, local_ee_oris = self._get_local_ee_poses(base_poses)

        return True

    def _step_simulation(self, record=True):
        self.sim.stepSimulation()
        if self.vis and record:
            record_interval = BaseEnv.SIM_FREQ // 30
            if self._internal_t % record_interval == 0:
                # render frames so resulting sequence is 10fps
                self._frames.append(self._render())
        self._internal_t += 1

    def _get_rigid_body_mesh(self, obj_id, link_index=None):
        # get mesh of rigid body, assuming it hasn't rotated from its canonical
        # pose
        mesh = self.sim.getMeshData(obj_id)[1]
        obj_pos, obj_ori = self.sim.getBasePositionAndOrientation(obj_id)
        rotation = Rotation.from_quat(obj_ori)
        rotation_matrix = rotation.as_matrix()
        mesh = np.dot(mesh, rotation_matrix.T)
        mesh += np.array(obj_pos)[None]
        return mesh

    def _get_closest(self, pos, obj_ids):
        """Get the closest point from a position among several meshes."""
        min_dist, selected_obj_id, selected_vertex_id = np.inf, None, None
        selected_vertex_pos, selected_link_idx = None, -1
        kwargs = {}
        if hasattr(pybullet, "MESH_DATA_SIMULATION_MESH"):
            kwargs["flags"] = pybullet.MESH_DATA_SIMULATION_MESH
        for obj_id in obj_ids:
            if obj_id in self.soft_ids:
                # soft object
                _, mesh_vertices = self.sim.getMeshData(obj_id, **kwargs)
                link_idxs = None
            elif self.sim.getNumJoints(obj_id) == 0:
                # free rigid object
                mesh_vertices = self._get_rigid_body_mesh(obj_id)
                link_idxs = None
            else:
                # compound rigid object
                num_joints = self.sim.getNumJoints(obj_id)
                mesh_vertices = []
                link_idxs = []
                for i in range(num_joints):
                    mesh_vertices.append(self._get_rigid_body_mesh(obj_id, i))
                    link_idxs.append(np.array([i] * len(mesh_vertices[-1])))
                mesh_vertices = np.concatenate(mesh_vertices)
                link_idxs = np.concatenate(link_idxs)
            vertex_id = get_closest(pos, mesh_vertices)[0]
            vertex_pos = np.array(mesh_vertices[vertex_id])
            vertex_dist = np.linalg.norm(pos - vertex_pos)
            if vertex_dist < min_dist:
                selected_obj_id = obj_id
                selected_vertex_id = vertex_id
                selected_vertex_pos = mesh_vertices[vertex_id]
                min_dist = vertex_dist
                if link_idxs is not None:
                    selected_link_idx = link_idxs[vertex_id]
        if min_dist > 0.15:
            return None, None, None, None
        return (
            selected_obj_id,
            selected_vertex_id,
            selected_vertex_pos,
            selected_link_idx,
        )

    def _attach_sim_anchor(self, robot_i):
        robot_id = self.robots[robot_i].info.robot_id
        link_id = self.robots[robot_i].info.ee_link_id
        ee_pos = self.robots[robot_i].get_ee_pos()

        if self.constraint_ids[robot_i] is not None:
            return

        graspable_ids = self.soft_ids + [
            x for i, x in enumerate(self.rigid_ids) if self._rigid_graspable[i]
        ]
        obj_id, vertex_id, vertex_pos, link_idx = self._get_closest(
            ee_pos, graspable_ids
        )
        if obj_id is None:
            return
        if obj_id in self.soft_ids:
            constraint_id = self.sim.createSoftBodyAnchor(
                obj_id, vertex_id, robot_id, link_id, ee_pos
            )
            self.constraint_ids[robot_i] = constraint_id
        else:
            # use plain physics for rigid object grasping
            pass

    def _detach_sim_anchor(self, robot_i):
        if self.constraint_ids[robot_i] is not None:
            self.sim.removeConstraint(self.constraint_ids[robot_i])
            self.constraint_ids[robot_i] = None

    def _compute_global_ee_poses(self):
        base_poses = self._get_base_poses()
        local_ee_poss, local_ee_oris = self._get_local_ee_poses(base_poses)
        global_ee_poss, global_ee_oris = [], []
        for i, robot in enumerate(self.robots):
            base_xy = base_poses[i, 0:2]
            global_ee_pos, global_ee_ori, _ = local_to_global_pose(
                local_ee_poss[i],
                local_ee_oris[i],
                base_xy=base_xy,
                base_rot=base_poses[i][2],
                height_offset=BaseEnv.ARM_MOUNTING_HEIGHT,
            )
            global_ee_poss.append(global_ee_pos)
            global_ee_oris.append(global_ee_ori)
        return dict(
            base_poses=base_poses,
            local_ee_poss=local_ee_poss,
            local_ee_oris=local_ee_oris,
            global_ee_poss=np.array(global_ee_poss),
            global_ee_oris=np.array(global_ee_oris),
        )

    def _get_obs(self, dummy_obs=False):
        global_ee_poses = self._compute_global_ee_poses()
        global_ee_poss = global_ee_poses["global_ee_poss"]
        if self.dof == 4:
            return global_ee_poss
        else:
            eef_pos = global_ee_poss
            global_ee_oris = global_ee_poses["global_ee_oris"]
            eef_rot = np.array(
                [
                    quat2mat(pybullet.getQuaternionFromEuler(ori.tolist()))
                    for ori in global_ee_oris
                ]
            )
            dir1 = eef_rot[:, :, 0]
            dir2 = eef_rot[:, :, 2]
            for i in range(len(dir1)):
                assert np.allclose(
                    np.cross(dir2[i], dir1[i]), eef_rot[:, :, 1][i], atol=1e-4
                )
            gravity_dir = np.zeros_like(eef_pos) + np.array([0.0, 0.0, -1.0])
            gripper_pose = np.array(
                [(self.constraint_ids[i] is not None) for i in range(len(self.robots))]
            )[:, None]
            return np.concatenate(
                [eef_pos, dir1, dir2, gravity_dir, gripper_pose], axis=-1
            )

    def _render(self):
        return self.render()["images"][0][..., :3]

    def render(
        self,
        return_seg=False,
        return_depth=True,
        return_pc=True,
        cam_info=None,
        hide_eef=False,
    ):
        cam_target = self.camera_config["target"]
        default_cam_info = {
            "yaws": [self.camera_config["yaw"]],
            "rolls": [self.camera_config["roll"]],
            "pitches": [self.camera_config["pitch"]],
            "dist": self.camera_config["distance"],
            "views": [0],
            "fov": self.camera_config["fov"],
            "width": self.args.cam_resolution,
            "height": self.args.cam_resolution,
            "target": cam_target,
        }
        if cam_info is not None:
            default_cam_info.update(cam_info)
        cam_info = default_cam_info

        rendered_img = MultiCamera.render(
            self.sim,
            self.rigid_ids,
            cam_rolls=cam_info["rolls"],
            cam_yaws=cam_info["yaws"],
            cam_pitches=cam_info["pitches"],
            cam_dist=cam_info["dist"],
            views=cam_info["views"],
            fov=cam_info["fov"],
            cam_target=cam_target,
            width=cam_info["width"],
            height=cam_info["height"],
            return_seg=(return_seg or return_pc),
            return_depth=return_depth,
            debug=self.debug,
        )
        rendered_img["images"] = [
            np.array(x).reshape(self.args.cam_resolution, self.args.cam_resolution, 4)
            for x in rendered_img["images"]
        ]
        if (return_seg or return_pc) and len(self.rigid_ids) > 0:
            rendered_img["segs"] = np.mod(rendered_img["segs"], (1 << 24))
        if return_pc:
            assert cam_info is not None
            assert return_depth
            pc, seg_img = self._compute_pc(
                cam_info,
                rendered_img["depths"],
                rendered_img["segs"],
                rendered_img["images"],
                return_seg_img=True,
            )
            rendered_img["pc"] = pc
            self._prev_pc_count = len(pc)
        else:
            rendered_img["pc"] = None

        return rendered_img

    def _compute_pc(
        self, cam_info, view_depths, view_segs, view_images, return_seg_img=False
    ):
        # construct camera info
        num_views = len(cam_info["yaws"])
        H, W = cam_info["height"], cam_info["width"]

        # get camera information
        cam_vals = MultiCamera.get_cam_vals(
            [0] * num_views,
            cam_info["yaws"],
            cam_info["pitches"],
            cam_info["dist"],
            cam_info["target"],
            cam_info["fov"],
            float(W / H),
        )
        img_ix = 0

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

        # get object pixels from segmentation
        obj_ids = self.soft_ids[:1] + self.rigid_ids
        segs = view_segs[img_ix]
        seg = np.isin(segs, obj_ids).astype(np.uint8) * 255

        object_pixels = np.array(np.where(seg == 255)).T

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
        if return_seg_img:
            return partial_pc, seg
        else:
            return partial_pc
