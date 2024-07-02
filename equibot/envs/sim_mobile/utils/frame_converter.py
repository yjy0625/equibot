"""
Utilities to convert coordinate frames for odometry from cameras.

@yjy0625, @jimmyyhwu.

"""

import numpy as np


def wrap_angle(x, ref):
    return ref + np.mod(x - ref + np.pi, np.pi * 2) - np.pi


class CoordFrameConverter(object):
    # Author: Jimmy Wu. Date: September 2022.
    def __init__(self, pose_in_map=np.zeros(3), pose_in_odom=np.zeros(3)):
        self.origin = None
        self.basis = None
        self.update(pose_in_map, pose_in_odom)

    def update(self, pose_in_map, pose_in_odom):
        self.basis = pose_in_map[2] - pose_in_odom[2]
        dx = pose_in_odom[0] * np.cos(self.basis) - pose_in_odom[1] * np.sin(self.basis)
        dy = pose_in_odom[0] * np.sin(self.basis) + pose_in_odom[1] * np.cos(self.basis)
        self.origin = (pose_in_map[0] - dx, pose_in_map[1] - dy)

    def convert_position(self, position):
        x, y = position
        x = x - self.origin[0]
        y = y - self.origin[1]
        xp = x * np.cos(-self.basis) - y * np.sin(-self.basis)
        yp = x * np.sin(-self.basis) + y * np.cos(-self.basis)
        return (xp, yp)

    def convert_heading(self, th):
        return th - self.basis

    def convert_pose(self, pose):
        x, y, th = pose
        return (*self.convert_position((x, y)), self.convert_heading(th))
