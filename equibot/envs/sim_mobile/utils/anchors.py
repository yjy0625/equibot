import numpy as np
import pybullet

from .plan_control_traj import plan_min_jerk_trajectory


def make_anchors(anchor_config, sim):
    anchor_ids = []
    for anchor_info in anchor_config:
        radius, rgba = anchor_info["radius"], anchor_info["rgba"]
        mass, pos = 0.0, anchor_info["pos"]
        visual_shape = sim.createVisualShape(
            pybullet.GEOM_SPHERE, radius=radius, rgbaColor=rgba
        )
        anchor_id = sim.createMultiBody(
            baseMass=mass,
            basePosition=pos,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            useMaximalCoordinates=True,
        )
        anchor_ids.append(anchor_id)
    return anchor_ids


def get_closest_mesh_vertex(pos, obj_ids, sim):
    """Get the closest point from a position among several meshes."""
    min_dist, selected_obj_id, selected_vertex_id = np.inf, None, None
    kwargs = {}
    if hasattr(pybullet, "MESH_DATA_SIMULATION_MESH"):
        kwargs["flags"] = pybullet.MESH_DATA_SIMULATION_MESH
    for obj_id in obj_ids:
        _, mesh_vertices = sim.getMeshData(obj_id, **kwargs)
        vertex_id = get_closest(pos, mesh_vertices)[0]
        vertex_pos = np.array(mesh_vertices[vertex_id])
        vertex_dist = np.linalg.norm(pos - vertex_pos)
        if vertex_dist < min_dist:
            selected_obj_id = obj_id
            selected_vertex_id = vertex_id
            min_dist = vertex_dist
    if min_dist > 0.25:
        return None, None
    return selected_obj_id, selected_vertex_id


def get_closest(point, vertices, max_dist=None):
    """Find mesh points closest to the given point."""
    point = np.array(point).reshape(1, -1)
    vertices = np.array(vertices)
    num_pins_per_pt = max(1, vertices.shape[0] // 50)
    num_to_pin = min(vertices.shape[0], num_pins_per_pt)
    dists = np.linalg.norm(vertices - point, axis=1)
    closest_ids = np.argpartition(dists, num_to_pin)[0:num_to_pin]
    if max_dist is not None:
        closest_ids = closest_ids[dists[closest_ids] <= max_dist]
    return closest_ids


def create_trajectory(waypoints, steps_per_waypoint, frequency):
    """Creates a smoothed trajectory through the given waypoints."""
    assert len(waypoints) == len(steps_per_waypoint)
    num_wpts = len(waypoints)
    tot_steps = sum(steps_per_waypoint[:-1])
    dt = 1.0 / frequency
    traj = np.zeros([tot_steps, 3 + 3])  # 3D pos , 3D vel
    prev_pos = waypoints[0]  # start at the 0th waypoint
    t = 0
    for wpt in range(1, num_wpts):
        tgt_pos = waypoints[wpt]
        dur = steps_per_waypoint[wpt - 1]
        if dur == 0:
            continue
        Y, Yd, Ydd = plan_min_jerk_trajectory(prev_pos, tgt_pos, dur, dt)
        traj[t : t + dur, 0:3] = Y[:]
        traj[t : t + dur, 3:6] = Yd[:]  # vel
        # traj[t:t+dur,6:9] = Ydd[:]  # acc
        t += dur
        prev_pos = tgt_pos
    if t < tot_steps:
        traj[t:, :] = traj[t - 1, :]  # set rest to last entry
    # print('create_trajectory(): traj', traj)
    return traj
