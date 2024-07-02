"""
Utilities to convert between local coordinates that are relative to the
base and global world coordinates.

@contactrika

"""

import numpy as np
import pybullet


def wrap_angle(x):  # wrap to range [-pi, pi]
    return np.mod(np.array(x) + np.pi, np.pi * 2) - np.pi


def flip_rot(rot):
    flipped_rot = rot + np.pi
    if flipped_rot > np.pi:
        flipped_rot = rot - np.pi
    return flipped_rot


def global_to_local_pose(
    global_pos, global_quat, base_xy, base_rot, height_offset, debug=False
):
    # Convert the global pose (in world coordinates) to local coordinates
    # that are relative to the base.
    # github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/
    # pybullet_envs/minitaur/robots/utilities/kinematics_utils.py
    # link_position_in_base_frame()
    arm_ori = [0, 0, np.pi]  # x for base is -x for arm in reality
    base_arm_pos, base_arm_quat = pybullet.multiplyTransforms(
        [*base_xy, height_offset],
        pybullet.getQuaternionFromEuler([0, 0, base_rot]),
        [0, 0, 0],
        pybullet.getQuaternionFromEuler(arm_ori),
    )
    inv_trans, inv_rot = pybullet.invertTransform(base_arm_pos, base_arm_quat)
    local_pos, local_quat = pybullet.multiplyTransforms(
        inv_trans, inv_rot, global_pos, global_quat
    )
    local_ori = wrap_angle(pybullet.getEulerFromQuaternion(local_quat))
    if debug:
        print(
            "global_to_local_pose: base_xy",
            base_xy,
            "base_rot deg",
            base_rot / np.pi * 180,
            "\nglobal_pos",
            global_pos,
            "global_ori deg",
            np.array(pybullet.getEulerFromQuaternion(global_quat)) / np.pi * 180,
            "\nlocal_pos",
            local_pos,
            "local_ori deg",
            np.array(local_ori) / np.pi * 180,
        )
    return local_pos, local_ori, local_quat


def local_to_global_pose(
    local_pos, local_ori, base_xy, base_rot, height_offset, debug=False
):
    # Converts the given local pos and ori to global coordinates.
    arm_ori = [0, 0, np.pi]  # x for base is -x for arm in reality
    base_arm_pos, base_arm_quat = pybullet.multiplyTransforms(
        [*base_xy, height_offset],
        pybullet.getQuaternionFromEuler([0, 0, base_rot]),
        [0, 0, 0],
        pybullet.getQuaternionFromEuler(arm_ori),
    )
    local_quat = pybullet.getQuaternionFromEuler(local_ori)
    global_pos, global_quat = pybullet.multiplyTransforms(
        base_arm_pos, base_arm_quat, local_pos, local_quat
    )
    global_ori = wrap_angle(pybullet.getEulerFromQuaternion(global_quat))
    if debug:
        print(
            "local_to_global_pose: base_xy",
            base_xy,
            "base_rot deg",
            base_rot / np.pi * 180,
            "\nlocal_pos",
            local_pos,
            "local_ori deg",
            np.array(pybullet.getEulerFromQuaternion(local_quat)) / np.pi * 180,
            "\nglobal_pos",
            global_pos,
            "global_ori deg",
            np.array(global_ori) / np.pi * 180,
        )
    return global_pos, global_ori, global_quat
