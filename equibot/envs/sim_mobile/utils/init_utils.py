#
# Utilities for deform sim in PyBullet.
#
# @contactrika
#
import os
from glob import glob

import numpy as np

np.set_printoptions(precision=2, linewidth=150, threshold=10000, suppress=True)
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bclient

from .multi_camera import MultiCamera


def load_rigid_object(sim, obj_file_name, scale, init_pos, init_ori, mass=0.0):
    """Load a rigid object from file, create visual and collision shapes."""
    assert obj_file_name.endswith(".obj")  # assume mesh info
    viz_shape_id = sim.createVisualShape(
        shapeType=pybullet.GEOM_MESH,
        rgbaColor=None,
        fileName=obj_file_name,
        meshScale=scale,
    )
    col_shape_id = sim.createCollisionShape(
        shapeType=pybullet.GEOM_MESH, fileName=obj_file_name, meshScale=scale
    )
    rigid_custom_id = sim.createMultiBody(
        baseMass=mass,  # mass==0 => fixed at the position where it is loaded
        basePosition=init_pos,
        baseCollisionShapeIndex=col_shape_id,
        baseVisualShapeIndex=viz_shape_id,
        baseOrientation=pybullet.getQuaternionFromEuler(init_ori),
    )
    return rigid_custom_id


def load_soft_object(
    sim,
    obj_file_name,
    scale,
    init_pos,
    init_ori,
    bending_stiffness,
    damping_stiffness,
    elastic_stiffness,
    friction_coeff,
    mass=1.0,
    collision_margin=0.002,
    fuzz_stiffness=False,
    use_self_collision=True,
    debug=False,
):
    """Load object from obj file with pybullet's loadSoftBody()."""
    if fuzz_stiffness:
        elastic_stiffness += (np.random.rand() - 0.5) * 2 * 20
        bending_stiffness += (np.random.rand() - 0.5) * 2 * 20
        friction_coeff += (np.random.rand() - 0.5) * 2 * 0.3
        scale += (np.random.rand() - 0.5) * 2 * 0.2
        if elastic_stiffness < 10.0:
            elastic_stiffness = 10.0
        if bending_stiffness < 10.0:
            bending_stiffness = 10.0
        scale = np.clip(scale, 0.6, 1.5)
        print(
            "fuzzed",
            f"elastic_stiffness {elastic_stiffness:0.4f}",
            f"bending_stiffness {bending_stiffness:0.4f}",
            f"friction_coeff {friction_coeff:0.4f} scale {scale:0.4f}",
        )
    # Note: do not set very small mass (e.g. 0.01 causes instabilities).
    deform_id = sim.loadSoftBody(
        scale=scale,
        mass=mass,
        fileName=obj_file_name,
        basePosition=init_pos,
        baseOrientation=pybullet.getQuaternionFromEuler(init_ori),
        collisionMargin=collision_margin,
        springElasticStiffness=elastic_stiffness,
        springDampingStiffness=damping_stiffness,
        springBendingStiffness=bending_stiffness,
        frictionCoeff=friction_coeff,
        useSelfCollision=use_self_collision,
        useNeoHookean=0,
        useMassSpring=1,
        useBendingSprings=1,
    )
    kwargs = {}
    if hasattr(pybullet, "MESH_DATA_SIMULATION_MESH"):
        kwargs["flags"] = pybullet.MESH_DATA_SIMULATION_MESH
    num_mesh_vertices, _ = sim.getMeshData(deform_id, **kwargs)
    if debug:
        print("Loaded deform_id", deform_id, "with", num_mesh_vertices, "mesh vertices")
    # Pybullet will struggle with very large meshes, so we should keep mesh
    # sizes to a limited number of vertices and faces.
    # Large meshes will load on Linux/Ubuntu, but sim will run too slowly.
    # Meshes with >2^13=8196 verstices will fail to load on OS X due to shared
    # memory limits, as noted here:
    # https://github.com/bulletphysics/bullet3/issues/1965
    assert num_mesh_vertices < 2**13  # make sure mesh has less than ~8K verts
    return deform_id


def create_spheres(
    id,
    radius=0.01,
    mass=0.0,
    batch_positions=[[0, 0, 0]],
    visual=True,
    collision=True,
    rgba=[0, 1, 1, 1],
):
    """
    Reference: https://github.com/Healthcare-Robotics/assistive-gym/blob/
    41d7059f0df0976381b2544b6bcfc2b84e1be008/assistive_gym/envs/base_env.py#L127
    """
    sphere_collision = -1
    if collision:
        sphere_collision = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_SPHERE, radius=radius, physicsClientId=id
        )

    sphere_visual = (
        pybullet.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
            physicsClientId=id,
        )
        if visual
        else -1
    )

    last_sphere_id = pybullet.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=sphere_collision,
        baseVisualShapeIndex=sphere_visual,
        basePosition=[0, 0, 0],
        useMaximalCoordinates=False,
        batchPositions=batch_positions,
        physicsClientId=id,
    )

    spheres = []
    for body in list(
        range(last_sphere_id[-1] - len(batch_positions) + 1, last_sphere_id[-1] + 1)
    ):
        # sphere = Agent()
        # sphere.init(body, id, self.np_random, indices=-1)
        spheres.append(body)
    return spheres


def init_bullet(args, sim=None, cam_on=False, cam_configs={}):
    """Initialize pybullet simulation."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(curr_dir)
    args.data_path = os.path.join(parent_dir, "assets")
    if args.viz:
        if sim is None:
            sim = bclient.BulletClient(connection_mode=pybullet.GUI)
        # toggle aux menus in the gui
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, cam_on)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, cam_on
        )
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, cam_on
        )
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, cam_on
        )
        # don't render during init
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        # Keep camera distance and target same as defaults for MultiCamera,
        # so that debug views look similar to those recorded by the camera pans.
        # Camera pitch and yaw can be set as desired, since they don't
        # affect MultiCamera panning strategy.
        cam_args = {
            "cameraDistance": 0.85,
            "cameraYaw": -30,
            "cameraPitch": -70,
            "cameraTargetPosition": np.array([0.35, 0, 0]),
        }
        cam_args.update(cam_configs)
        sim.resetDebugVisualizerCamera(**cam_args)
    else:
        if sim is None:
            sim = bclient.BulletClient(connection_mode=pybullet.DIRECT)
    sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    sim.setGravity(0, 0, args.sim_gravity)
    sim.setTimeStep(1.0 / args.sim_frequency)
    return sim


def scale_mesh(obj_file, scale):
    # Load the OBJ file
    with open(obj_file, "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    v_lines_idxs = [k for k in range(len(lines)) if lines[k].startswith("v ")]
    v_lines = [lines[idx] for idx in v_lines_idxs]
    vertices = [[float(x) for x in l.split(" ")[1:]] for l in v_lines]
    vertices = np.array(vertices)
    vertices = vertices * np.array(scale)[None]
    for i, idx in enumerate(v_lines_idxs):
        x, y, z = vertices[i]
        lines[idx] = f"v {x:.6f} {y:.6f} {z:.6f}"

    # Save the scaled mesh to a new OBJ file
    pid = os.getpid()
    output_file = obj_file.replace(
        ".obj", f"_scaled_x{scale[0]:.2f}_y{scale[1]:.2f}_pid{pid}.obj"
    )
    existing_generated_files = glob(
        os.path.join(os.path.dirname(obj_file), f"*_scaled*_pid{pid}.obj")
    )
    for f in existing_generated_files:
        os.remove(f)
    with open(output_file, "w") as f:
        for line in lines:
            f.write(f"{line}\n")
    return output_file


def rotate_around_z(
    points,
    angle_rad=0.0,
    center=np.array([0.0, 0.0, 0.0]),
    scale=np.array([1.0, 1.0, 1.0]),
):
    # Check if the input points have the correct shape (N, 3)
    assert (len(points.shape) == 1 and len(points) == 3) or points.shape[-1] == 3
    p_shape = points.shape
    points = points.reshape(-1, 3) - center[None]

    # Create the rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )

    # Apply the rotation to all points using matrix multiplication
    rotated_points = np.dot(points, rotation_matrix.T) * scale[None] + center[None]
    rotated_points = rotated_points.reshape(p_shape)

    return rotated_points
