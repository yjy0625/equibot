import os
import re
import pybullet as p
import numpy as np
from tempfile import NamedTemporaryFile

from equibot.envs.sim_mobile.base_env import BaseEnv
from equibot.envs.sim_mobile.utils.init_utils import rotate_around_z


def evaluate_and_replace_expressions(input_file, output_file, local_vars):
    with open(input_file, "r") as file:
        content = file.read()

    def eval_expression(match):
        expression = match.group(1)
        try:
            locals().update(local_vars)
            return str(eval(expression))
        except Exception as e:
            return f"ERROR: {e}"

    # find all "${...}" expressions and replace with evaluated results
    processed_content = re.sub(r"\${(.*?)}", eval_expression, content)
    with open(output_file, "w") as file:
        file.write(processed_content)


class ClosingEnv(BaseEnv):
    BASE_INIT_ROT = 0
    OTHER_BASE_INIT_ROT = np.pi

    @property
    def robot_config(self):
        L, W, H = self._box_size
        left_pos = np.array([-L / 2 - L * 0.35, W * 0.1, 0.005])
        right_pos = np.array([L / 2 + L * 0.35, W * 0.1, 0.005])
        init_base_pos = np.stack([left_pos, right_pos])
        init_base_pos[0, 0] -= 0.75
        init_base_pos[1, 0] += 0.75
        init_base_pos[0, 0] -= L / 3
        init_base_pos[1, 0] += L / 3
        init_base_pos = rotate_around_z(init_base_pos, self._object_rotation[-1])
        init_base_pos[:, :2] += self.scene_offset[None]
        init_base_rot = [self._object_rotation[-1] + np.pi, self._object_rotation[-1]]
        rest_arm_pos = np.array([0.75, 0.0, H * 1.0 - self.ARM_MOUNTING_HEIGHT])
        rest_arm_rot = np.array([np.pi * 0.5, np.pi, np.pi / 2])
        robots = [
            {
                "sim_robot_name": "kinova",
                "rest_base_pose": np.array(
                    [init_base_pos[0, 0], init_base_pos[0, 1], init_base_rot[0]]
                ),
                "rest_arm_pos": rest_arm_pos,
                "rest_arm_rot": rest_arm_rot,
            },
            {
                "sim_robot_name": "kinova",
                "rest_base_pose": np.array(
                    [init_base_pos[1, 0], init_base_pos[1, 1], init_base_rot[1]]
                ),
                "rest_arm_pos": rest_arm_pos,
                "rest_arm_rot": rest_arm_rot,
            },
        ]
        return robots

    @property
    def default_camera_config(self):
        cfg = super().default_camera_config
        cfg["pitch"] = -45
        cfg["yaw"] = 45
        cfg["distance"] = 1.5 * np.max(self._box_size / np.array([0.2, 0.2, 0.1]))
        cfg["target"] = [self.scene_offset[0], self.scene_offset[1], 0.1]
        return cfg

    @property
    def rigid_objects(self):
        return []

    @property
    def anchor_config(self):
        return []

    @property
    def soft_objects(self):
        return []

    @property
    def name(self):
        return "close"

    def visualize_anchor(self, pos):
        return super().visualize_anchor(pos + np.array([[0, 0, 0.09]]))

    def visualize_pc(self, pos):
        return super().visualize_pc(pos + np.array([[0, 0, 0.09]]))

    def _init_robots(self):
        self._box_size = np.array([0.0, 0.0, 0.0])
        super()._init_robots()

    def _reset_sim(self):
        # constants
        box_size = np.array([0.145, 0.12, 0.115]) * self._rigid_object_scale
        self._box_size = box_size
        self._box_thickness = 0.005

        super()._reset_sim()

        # add box to sim
        self.rigid_ids.append(self._init_box())
        self._rigid_graspable.append(True)

    def _init_box(self):
        dir_path = os.path.dirname(__file__)
        template_path = os.path.join(dir_path, "assets/closing/template.urdf")
        local_vars = dict(
            L=self._box_size[0],
            W=self._box_size[1],
            H=self._box_size[2],
            T=self._box_thickness,
        )
        with NamedTemporaryFile(mode="w", suffix=".urdf") as f:
            evaluate_and_replace_expressions(template_path, f.name, local_vars)
            box_id = p.loadURDF(
                f.name,
                np.zeros([3]),
                useFixedBase=True,
                flags=p.URDF_MAINTAIN_LINK_ORDER
                | p.URDF_USE_SELF_COLLISION
                | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            )

        p.changeVisualShape(box_id, -1, rgbaColor=[0.678, 0.573, 0.439, 1.0])

        init_pos = np.array([0, 0, self._box_size[2] / 2])
        init_pos[:2] += self.scene_offset
        rotation_quaternion = p.getQuaternionFromEuler(
            [0, 0, self._object_rotation[-1]]
        )
        p.resetBasePositionAndOrientation(box_id, init_pos, rotation_quaternion)

        for joint_index in range(8):
            p.changeDynamics(
                box_id,
                joint_index,
                lateralFriction=0.0,
                spinningFriction=0.0,
                rollingFriction=0.0,
                linearDamping=0.0,
                angularDamping=0.0,
            )
            p.setJointMotorControl2(
                box_id, joint_index, controlMode=p.VELOCITY_CONTROL, force=0
            )

        return box_id

    def _get_obs(self, dummy_obs=False):
        obs = super()._get_obs(dummy_obs=dummy_obs)
        return obs

    def compute_reward(self):
        joint_states = [p.getJointState(self.rigid_ids[0], ix)[0] for ix in [5, 6, 7]]
        return np.clip(
            1.0 - np.abs(np.array(joint_states) - 3.14).mean() / 3.14, 0.0, 1.0
        )

    def _get_rigid_body_mesh(self, obj_id, link_index=None):
        if obj_id in self.rigid_ids:
            mesh_vertices = []
            num_links = p.getNumJoints(obj_id)
            link_idxs = list(range(num_links)) if link_index is None else [link_index]
            for link_idx in link_idxs:
                col_data = p.getCollisionShapeData(obj_id, link_idx)
                size = col_data[0][3]
                local_pos = col_data[0][5]
                link_state = p.getLinkState(obj_id, link_idx)
                pos, ori = link_state[0], link_state[1]

                get_num_pts = lambda s: (
                    20 if s > self._box_thickness * 4 else 2
                )  # max(2, np.round(s / 0.01).astype(int))
                for dx in np.linspace(
                    -0.5 * size[0], 0.5 * size[0], get_num_pts(size[0])
                ):
                    for dy in np.linspace(
                        -0.5 * size[1], 0.5 * size[1], get_num_pts(size[1])
                    ):
                        for dz in np.linspace(
                            -0.5 * size[2], 0.5 * size[2], get_num_pts(size[2])
                        ):
                            dxyz = tuple((np.array([dx, dy, dz]) + local_pos).tolist())
                            vertex = p.multiplyTransforms(pos, ori, dxyz, [0, 0, 0, 1])[
                                0
                            ]
                            mesh_vertices.append(vertex)
            verts = np.array(mesh_vertices)
            verts[:, :2] += self.scene_offset
            return verts
        else:
            return super()._get_rigid_body_mesh(obj_id)
