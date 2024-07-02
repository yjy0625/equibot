import os
import pybullet
import numpy as np
import trimesh

from equibot.envs.sim_mobile.base_env import BaseEnv
from equibot.envs.sim_mobile.utils.init_utils import scale_mesh, rotate_around_z


class CoveringEnv(BaseEnv):
    @property
    def name(self):
        return "covering"

    @property
    def robot_config(self):
        init_base_pos = np.array([[-0.2, -0.1, 0.005], [+0.2, -0.1, 0.005]])
        init_base_pos[:, :2] *= self._soft_object_scale[None]
        init_base_pos[0, 0] -= 0.7 / np.sqrt(2)
        init_base_pos[1, 0] += 0.7 / np.sqrt(2)
        init_base_pos[0, 1] -= 0.7 / np.sqrt(2)
        init_base_pos[1, 1] -= 0.7 / np.sqrt(2)
        init_base_pos = rotate_around_z(init_base_pos, self._object_rotation[-1])
        init_base_pos[:, :2] += self.scene_offset[None]
        init_base_rot = [
            self._object_rotation[-1] + np.pi * 1.25,
            self._object_rotation[-1] - np.pi * 0.25,
        ]
        rest_arm_pos = np.array([0.7, 0.0, 0.01 - self.ARM_MOUNTING_HEIGHT])
        rest_arm_rot = np.array([np.pi * 0.6, 0.0, np.pi / 2])
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

    def _randomize_object_scales(self):
        # override object scale variable so default size is equal to source size
        super()._randomize_object_scales()
        self._rigid_object_scale *= np.array([1.0, 0.7, 0.5]) * 1
        self._soft_object_scale *= np.array([0.6875, 0.6875]) * 1

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
    def rigid_objects(self):
        ang = self._object_rotation[-1]
        self._box_size = 0.05
        print(f"Randomizing rigid object scale to {self._rigid_object_scale}!")
        obj_path = os.path.join(self.data_path, "covering/box.obj")
        obj_path = scale_mesh(obj_path, self._rigid_object_scale)
        obj_path = "/".join(obj_path.split("/")[-2:])
        return [
            {
                "path": obj_path,
                "scale": 1.0,
                "pos": [
                    self.scene_offset[0],
                    self.scene_offset[1],
                    self._box_size * self._rigid_object_scale[-1],
                ],
                "orn": self._object_rotation,
            }
        ]

    @property
    def soft_objects(self):
        scale = 2.0
        mass = 0.2
        collision_margin = 0.001
        xy_scale = self._soft_object_scale
        print(f"Randomizing object scale to {xy_scale}!")
        obj_path = os.path.join(self.data_path, "folding/towel.obj")
        obj_path = scale_mesh(obj_path, np.array([xy_scale[0], xy_scale[1], 1.0]))
        obj_path = "/".join(obj_path.split("/")[-2:])
        ang = self._object_rotation[-1]
        init_y = (
            -0.3 * self._soft_object_scale[1] * np.cos(ang) + self.scene_offset[1]
        )
        return [
            {
                "path": obj_path,
                "scale": scale,
                "pos": [
                    0.3 * self._soft_object_scale[1] * np.sin(ang)
                    + self.scene_offset[0],
                    init_y,
                    0.001,
                ],
                "orn": self._object_rotation,
                "mass": mass,
                "collision_margin": collision_margin,
            }
        ]

    def compute_reward(self):
        obj_id = self.soft_ids[0]
        cloth_mesh_xyzs = np.array(self.sim.getMeshData(obj_id)[1])
        cloth_vol = trimesh.convex.convex_hull(cloth_mesh_xyzs)
        rigid_mesh_xyzs = self._get_rigid_body_mesh(self.rigid_ids[0])
        rigid_vol = trimesh.convex.convex_hull(rigid_mesh_xyzs)
        rigid_volume = rigid_vol.volume
        intersect_volume = rigid_vol.intersection(cloth_vol).volume
        return intersect_volume / rigid_volume

    def _reset_sim(self):
        self.args.deform_elastic_stiffness = 100.0
        self.args.deform_friction_coeff = 1.0
        super()._reset_sim()

        # make floor friction higher
        pybullet.changeDynamics(self.rigid_ids[0], -1, lateralFriction=0.3)

        # preload box mesh vertices
        # this is required because pybullet by default doesn't save all vertices
        # of the object
        obj_path = os.path.join(self.data_path, "covering/box.obj")
        mesh = trimesh.load(obj_path).vertices * self._rigid_object_scale
        self._box_vertices = mesh

        texture_id = self.sim.loadTexture("textures/comforter.png")
        self.sim.changeVisualShape(
            self.soft_ids[0], -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id
        )

    def _get_rigid_body_mesh(self, obj_id):
        assert obj_id == self.rigid_ids[0]
        mesh = self._box_vertices.copy()
        mesh = rotate_around_z(mesh, self._object_rotation[-1])
        mesh += np.array(self._rigid_objects[0]["pos"])[None]
        return mesh
