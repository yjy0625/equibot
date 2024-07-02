import numpy as np
import pybullet


DEFAULT_ANCHOR_CONFIG = [
    {"pos": np.array([0.0, 0.0, 0.0]), "radius": 0.02, "rgba": (0.0, 0.0, 0.0, 0.0)},
    {"pos": np.array([1.0, 0.0, 0.0]), "radius": 0.02, "rgba": (1.0, 0.0, 0.0, 0.0)},
    {"pos": np.array([0.0, 1.0, 0.0]), "radius": 0.02, "rgba": (1.0, 0.3, 0.0, 0.0)},
    {"pos": np.array([0.0, 0.0, -10.0]), "radius": 0.02, "rgba": (0.3, 0.8, 0.6, 1.0)},
]


KINOVA_HOME_QPOS = np.array([0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57])


SIM_ROBOT_INFO = {
    "kinova": {
        "file_name": "kinova/base_with_kinova_gripper.urdf",
        "ee_joint_name": "end_effector",
        "ee_link_name": "tool_frame",
        "rest_arm_qpos": KINOVA_HOME_QPOS,
    },
    "kinova_tta": {
        "file_name": "kinova/base_with_kinova_tta.urdf",
        "ee_joint_name": "end_effector",
        "ee_link_name": "tool_frame",
        "rest_arm_qpos": KINOVA_HOME_QPOS,
    },
    "kinova_ladle": {
        "file_name": "kinova/base_with_kinova_ladle.urdf",
        "ee_joint_name": "end_effector",
        "ee_link_name": "tool_frame",
        "rest_arm_qpos": KINOVA_HOME_QPOS,
    },
    "kinova_pan": {
        "file_name": "kinova/base_with_kinova_pan.urdf",
        "ee_joint_name": "end_effector",
        "ee_link_name": "tool_frame",
        "rest_arm_qpos": KINOVA_HOME_QPOS,
    },
}
