import numpy as np


def vectorized_unproject(
    depth, intrinsics, rgb=None, depth_scale=1.0, filter_pixels=None
):
    # parse intrinsics and scale depth
    fx, fy, cx, cy = np.array(intrinsics)[[0, 1, 0, 1], [0, 1, 2, 2]]
    depth /= depth_scale

    # form the mesh grid
    if filter_pixels is None:
        xv, yv = np.meshgrid(
            np.arange(depth.shape[1], dtype=float),
            np.arange(depth.shape[0], dtype=float),
        )
    else:
        xv, yv = filter_pixels[:, 1], filter_pixels[:, 0]
        depth = depth[filter_pixels[:, 0], filter_pixels[:, 1]]

    # transform coordinates and concatenate xyz on 2nd axis
    xv = (xv - cx) / fx * depth
    yv = (yv - cy) / fy * depth
    points = np.c_[xv.flatten(), yv.flatten(), depth.flatten()]

    # attach rgb values if provided
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
        points = np.concatenate([points, rgb], axis=1)

    return points


def unproject_depth(
    depths,
    intrinsics,
    extrinsics,
    clip_radius=1.0,
    filter_xyz=True,
    min_x=-1.0,
    max_x=1.0,
    min_y=-1.0,
    max_y=1.0,
    min_z=-1.0,
    max_z=1.0,
    filter_pixels=None,
):
    """
    Converts depth images to point cloud in world coordinate system.
    Args:
        depths: all depth images
        intrinsics: all intrinsic matrices from camera coordinates to pixels
        extrinsics: all extrinsic matrices from camera to world coordinates
        clip_radius: clipping radius for filtering out sky box
        filter_xyz: perform point cloud filtering by axis limits
        [min,max]_[x,y,z]: axis limits for point cloud items
        filter_pixels: only unproject depth at the specified pixels
    """
    camera_xyzs = []
    world_xyzs = []
    if filter_pixels is None:
        filter_pixels = [None] * len(depths)
    for i in range(len(depths)):
        # filter mask
        if filter_pixels[i] is None:
            clip_mask = depths[i].flatten() < clip_radius
        else:
            clip_mask = (
                depths[i][filter_pixels[i][:, 0], filter_pixels[i][:, 1]] < clip_radius
            )

        # unproject depths into camera coordinate
        camera_xyz = vectorized_unproject(
            depths[i], intrinsics[i], filter_pixels=filter_pixels[i]
        )
        camera_xyz_homo = np.c_[camera_xyz, np.ones(len(camera_xyz))]
        camera_xyzs.append(camera_xyz[clip_mask])

        # convert positions to world coordinate
        world_xyz = np.dot(extrinsics[i], camera_xyz_homo.T).T
        filtered_world_xyz = world_xyz[clip_mask][:, :3]
        if filter_xyz:
            filtered_world_xyz = filtered_world_xyz[filtered_world_xyz[..., 0] > min_x]
            filtered_world_xyz = filtered_world_xyz[filtered_world_xyz[..., 0] < max_x]
            filtered_world_xyz = filtered_world_xyz[filtered_world_xyz[..., 1] > min_y]
            filtered_world_xyz = filtered_world_xyz[filtered_world_xyz[..., 1] < max_y]
            filtered_world_xyz = filtered_world_xyz[filtered_world_xyz[..., 2] > min_z]
            filtered_world_xyz = filtered_world_xyz[filtered_world_xyz[..., 2] < max_z]
        world_xyzs.append(filtered_world_xyz)

    return np.concatenate(world_xyzs, axis=0)
