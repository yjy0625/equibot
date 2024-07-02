#
# MultiCamera class that processes depth and RGB camera input from PyBullet.
# The code follows basic recommendations from PyBullet forums.
# Note that it uses assumptions of the camera setup, which work in the
# current pybullet versions, but ultimately might change in the future.
# Using pybullet versions from 2.6.4 to 2.8.1 should work fine.
#
# @contactrika
#
import os
import sys
import math
import time

import numpy as np

np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)
import pybullet


def assert_close(ars, ars0):
    for ar, ar0 in zip(ars, ars0):
        assert np.linalg.norm(np.array(ar) - np.array(ar0)) < 1e-6


def get_camera_info(args, aux=False):
    if not aux:
        yaws, pitches = [], []
        for y in args.cam_yaws:
            for p in args.cam_pitches:
                yaws.append(y)
                pitches.append(p)
        num_views = len(yaws)
    else:
        yaws = [90]
        pitches = [-20]
        num_views = 1
    cam_info = {
        "yaws": yaws,
        "pitches": pitches,
        "dist": args.cam_dist,
        "views": list(np.arange(num_views)),
        "fov": args.cam_fov,
        "width": args.cam_resolution,
        "height": args.cam_resolution,
    }
    return cam_info


class MultiCamera:
    # In non-GUI mode we will render without X11 context but *with* GPU accel.
    # examples/pybullet/examples/testrender_egl.py
    # Note: use alpha=1 (in rgba), otherwise depth readings are not good
    # Using defaults from PyBullet.
    # See examples/pybullet/examples/pointCloudFromCameraImage.py
    PYBULLET_FAR_PLANE = 10000
    PYBULLET_NEAR_VAL = 0.01
    PYBULLET_FAR_VAL = 1000.0

    @staticmethod
    def init(viz):
        if viz:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1
            )

    @staticmethod
    def get_cam_vals(
        cam_rolls, cam_yaws, cam_pitches, cam_dist, cam_target, fov, aspect_ratio=1.0
    ):
        # load static variables
        near_val = MultiCamera.PYBULLET_NEAR_VAL
        far_val = MultiCamera.PYBULLET_FAR_VAL
        far_plane = MultiCamera.PYBULLET_FAR_PLANE

        # compute cam vals
        cam_vals = []
        for cam_roll, cam_yaw, cam_pitch in zip(cam_rolls, cam_yaws, cam_pitches):
            view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
                cam_target, cam_dist, cam_yaw, cam_pitch, cam_roll, upAxisIndex=2
            )
            np_view_matrix = np.array(view_matrix).reshape(4, 4)
            proj_matrix = pybullet.computeProjectionMatrixFOV(
                fov, aspect_ratio, near_val, far_val
            )
            forward_vec = -np_view_matrix[:3, 2]
            horizon = np_view_matrix[:3, 0] * far_plane * 2 * aspect_ratio
            vertical = np_view_matrix[:3, 1] * far_plane * 2
            cam_vals.append(
                [
                    view_matrix,
                    proj_matrix,
                    forward_vec,
                    horizon,
                    vertical,
                    cam_dist,
                    cam_target,
                ]
            )

        return cam_vals

    def soft_ptcloud(sim, softbody_ids, debug=False):
        """Add SoftBody vertex positions to the point cloud."""
        deform_ptcloud = []
        deform_tracking_ids = []
        if softbody_ids is not None:
            for i in range(len(softbody_ids)):
                num_verts, verts = sim.getMeshData(softbody_ids[i])
                for v in verts:
                    deform_ptcloud.append(np.array(v))
                    deform_tracking_ids.append(softbody_ids[i])
        deform_ptcloud = np.array(deform_ptcloud)
        deform_tracking_ids = np.array(deform_tracking_ids)
        return deform_ptcloud, deform_tracking_ids

    def render(
        sim,
        object_ids,
        cam_rolls=[0] * 7,
        cam_yaws=[-30, 10, 50, 90, 130, 170, 210],
        cam_pitches=[-70, -10, -65, -40, -10, -25, -60],
        cam_dist=0.85,
        cam_target=np.array([0.35, 0, 0]),
        fov=90,
        views=[2],
        width=100,
        height=100,
        return_seg=False,
        return_depth=False,
        debug=False,
    ):
        imgs, depths, segs = [], [], []
        cam_vals = MultiCamera.get_cam_vals(
            cam_rolls,
            cam_yaws,
            cam_pitches,
            cam_dist,
            cam_target,
            fov,
            aspect_ratio=float(width / height),
        )

        # render views
        for i in views:
            (
                view_matrix,
                proj_matrix,
                cam_forward,
                cam_horiz,
                cam_vert,
                cam_dist,
                cam_tgt,
            ) = cam_vals[i]
            w, h, rgba_px, depth_raw_cam_dists, segment_mask = sim.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                lightDirection=np.array([0.0, 0.0, 5.0]),
                lightColor=np.array([1.0, 1.0, 1.0]),
                lightDistance=2.0,
                shadow=1,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
                # renderer=pybullet.ER_TINY_RENDERER,
                flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            )
            imgs.append(np.array(rgba_px).reshape(height, width, 4))
            depths.append(np.array(depth_raw_cam_dists).reshape(height, width))
            segs.append(np.array(segment_mask).reshape(height, width))

        # prepare return
        return_dict = dict(images=imgs)
        if return_depth:
            for i, depth in enumerate(depths):
                depth = np.array(depth).reshape(height, width).T
                near_val = MultiCamera.PYBULLET_NEAR_VAL
                far_val = MultiCamera.PYBULLET_FAR_VAL
                depth = far_val * near_val / (far_val - (far_val - near_val) * depth)
                depths[i] = depth
            return_dict["depths"] = depths
        if return_seg:
            return_dict["segs"] = segs
        return return_dict
