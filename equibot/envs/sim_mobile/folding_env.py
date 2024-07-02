import os
import numpy as np

from equibot.envs.sim_mobile.base_env import BaseEnv
from equibot.envs.sim_mobile.utils.init_utils import scale_mesh, rotate_around_z


class FoldingEnv(BaseEnv):
    BASE_INIT_ROT = 0
    OTHER_BASE_INIT_ROT = np.pi
    INITIAL_HEIGHT = 0.02

    @property
    def robot_config(self):
        init_base_pos = np.array([[0.21, 0.2, 0.01], [-0.21, 0.2, 0.01]])
        init_base_pos[:, :2] *= self._soft_object_scale[None]
        init_base_pos[0, 0] += 0.7 / np.sqrt(2)
        init_base_pos[0, 1] += 0.7 / np.sqrt(2)
        init_base_pos[1, 0] -= 0.7 / np.sqrt(2)
        init_base_pos[1, 1] += 0.7 / np.sqrt(2)
        init_base_pos = rotate_around_z(init_base_pos, self._object_rotation[-1])
        init_base_pos[:, :2] += self.scene_offset[None]
        init_base_rot = [
            self._object_rotation[-1] + np.pi / 4,
            self._object_rotation[-1] + np.pi / 4 * 3,
        ]
        rest_arm_pos = np.array(
            [0.7, 0.0, self.INITIAL_HEIGHT - self.ARM_MOUNTING_HEIGHT]
        )
        rest_arm_rot = np.array([[np.pi * 0.6, 0.0, np.pi / 2]] * 2)
        robots = [
            {
                "sim_robot_name": "kinova",
                "rest_base_pose": np.array(
                    [init_base_pos[0, 0], init_base_pos[0, 1], init_base_rot[0]]
                ),
                "rest_arm_pos": rest_arm_pos,
                "rest_arm_rot": rest_arm_rot[0],
            },
            {
                "sim_robot_name": "kinova",
                "rest_base_pose": np.array(
                    [init_base_pos[1, 0], init_base_pos[1, 1], init_base_rot[1]]
                ),
                "rest_arm_pos": rest_arm_pos,
                "rest_arm_rot": rest_arm_rot[1],
            },
        ]
        return robots

    @property
    def rigid_objects(self):
        return []

    @property
    def default_camera_config(self):
        cfg = super().default_camera_config
        cfg["pitch"] = -75
        cfg["distance"] = np.max(self._soft_object_scale) * 2
        cfg["target"] = [self.scene_offset[0], self.scene_offset[1], 0.1]
        return cfg

    @property
    def anchor_config(self):
        return []

    @property
    def soft_objects(self):
        xy_scale = self._soft_object_scale
        print(f"Randomizing object scale to {xy_scale}!")
        obj_path = os.path.join(
            self.data_path,
            "folding/towel_small.obj"
        )
        obj_path = scale_mesh(obj_path, np.array([xy_scale[0], xy_scale[1], 1.0]))
        obj_path = "/".join(obj_path.split("/")[-2:])
        return [
            {
                "path": obj_path,
                "scale": 2.0,
                "pos": [self.scene_offset[0], self.scene_offset[1], 0.001],
                "orn": self._object_rotation,
                "mass": 0.5,
                "collision_margin": 0.005,
            }
        ]

    @property
    def name(self):
        return "fold"

    def _randomize_object_scales(self):
        super()._randomize_object_scales()
        self._soft_object_scale *= np.array([0.6875, 0.6875])

    def _reset_sim(self):
        super()._reset_sim()

        vertices = self.sim.getMeshData(self.soft_ids[0])[1]
        self._corner_idxs = self._get_corner_vertex_indices(self.soft_ids[0])
        mesh_xyzs = np.array(self.sim.getMeshData(self.soft_ids[0])[1])
        self._init_corner_positions = mesh_xyzs[self._corner_idxs]

        texture_id = self.sim.loadTexture("textures/comforter.png")
        self.sim.changeVisualShape(
            self.soft_ids[0], -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id
        )

    def compute_reward(self):
        obj_id = self.soft_ids[0]
        mesh_xyzs = np.array(self.sim.getMeshData(obj_id)[1])
        corner_positions = mesh_xyzs[self._corner_idxs]
        dist = 0.0
        dist += np.linalg.norm(corner_positions[1] - self._init_corner_positions[0]) / 2
        dist += np.linalg.norm(corner_positions[3] - self._init_corner_positions[2]) / 2
        min_dist = self.INITIAL_HEIGHT
        max_dist = self._soft_object_scale[1] * 0.4
        dist = 1.0 - max(0, dist - min_dist) / (max_dist - min_dist)
        return max(0.0, min(1.0, dist))

    def _get_corner_vertex_indices(self, obj_id):
        mesh_xyzs = np.array(self.sim.getMeshData(obj_id)[1])
        mesh_xyzs = rotate_around_z(mesh_xyzs, -self._object_rotation[-1])
        max_xy = np.max(mesh_xyzs, axis=0)[:2]
        min_xy = np.min(mesh_xyzs, axis=0)[:2]

        minx = np.where(mesh_xyzs[:, 0] <= min_xy[0] + 0.001)[0]
        maxx = np.where(mesh_xyzs[:, 0] >= max_xy[0] - 0.001)[0]
        miny = np.where(mesh_xyzs[:, 1] <= min_xy[1] + 0.001)[0]
        maxy = np.where(mesh_xyzs[:, 1] >= max_xy[1] - 0.001)[0]

        find_overlapping_index = lambda a, b: list(set(a).intersection(set(b)))[0]
        corners = [
            find_overlapping_index(minx, miny),
            find_overlapping_index(minx, maxy),
            find_overlapping_index(maxx, miny),
            find_overlapping_index(maxx, maxy),
        ]
        return corners
