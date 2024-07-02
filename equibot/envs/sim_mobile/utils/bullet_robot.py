"""
PyBullet simulator class for manipulators.

From github.com/contactrika/dedo/blob/main/dedo/utils/bullet_manipulator.py

@contactrika, @yjy0625

Useful links:

What Euler angle is the used in the pybullet?
Answer: XYZ fixed axes (world frame, extrinsic).
github.com/bulletphysics/bullet3/issues/2334

"""

from copy import copy
import os
import time

import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bclient

from equibot.envs.sim_mobile.utils.console import suppress_stdout


def theta_to_sin_cos(rads):
    sin_cos = np.vstack([np.sin(rads), np.cos(rads)])  # [[sin,...],[cos,...]]
    return sin_cos.T.reshape(-1)  # [sin,cos,sin,cos,...]


def sin_cos_to_theta(sin_cos):
    assert len(sin_cos.shape) == 1
    assert (sin_cos.shape[0] % 2) == 0  # need [sin,cos,sin,cos,...]
    sin_cos = sin_cos.reshape(-1, 2)  # [[sin,cos],[sin,cos]...]
    theta = np.arctan2(sin_cos[:, 0], sin_cos[:, 1])
    return theta.reshape(-1)  # [theta0,theta1,...]


def quat_to_sin_cos(quat):
    assert len(quat.shape) == 1
    assert quat.shape[0] == 4  # [x,y,z,w]
    euler = pybullet.getEulerFromQuaternion(quat)
    euler = np.array(euler).reshape(-1)
    return theta_to_sin_cos(euler)  # [sin,cos,sin,cos,...]


def sin_cos_to_quat(sin_cos):
    assert len(sin_cos.shape) == 1
    assert (sin_cos.shape[0] % 2) == 0  # need [sin,cos,sin,cos,...]
    rads = sin_cos_to_theta(sin_cos)  # [theta0,theta1,theta2]
    assert len(rads) == 3
    quat = pybullet.getQuaternionFromEuler(rads.tolist())
    return quat


def normalize(x):
    return x / (np.linalg.norm(x) + 1e-12)


def sgn(error, threshold=0.005, preserve_scale=False):
    close_flag = np.abs(error) < threshold
    sgn_value = np.sign(error)
    if preserve_scale and not np.all(close_flag):
        max_value = np.max(np.abs(error))
        sgn_value = error / max_value
    return (1 - close_flag) * sgn_value


class BulletRobotInfo(object):
    def __init__(
        self,
        robot_id,
        joint_ids,
        joint_names,
        joint_minpos,
        joint_maxpos,
        joint_maxforce,
        joint_maxvel,
        ee_link_id,
        arm_jids_lst,
        ee_jid,
        finger_link_ids,
        finger_jids_lst,
        continuous_jids_lst=None,
        torso_jids_lst=None,
    ):
        self.robot_id = robot_id
        self.joint_ids = joint_ids
        self.joint_names = joint_names
        self.joint_minpos = joint_minpos
        self.joint_maxpos = joint_maxpos
        self.joint_maxforce = joint_maxforce
        self.joint_maxvel = joint_maxvel
        self.ee_link_id = ee_link_id
        self.arm_jids_lst = arm_jids_lst
        self.ee_jid = ee_jid
        self.finger_link_ids = finger_link_ids
        self.finger_jids_lst = finger_jids_lst
        self.torso_jids_lst = torso_jids_lst
        self.continuous_jids_lst = continuous_jids_lst
        self.dof = len(joint_ids)

    def print(self):
        print(
            "ManipulatorInfo: robot_id",
            self.robot_id,
            "\n joint_ids",
            self.joint_ids,
            "\n joint_names",
            self.joint_names,
            "\n joint_minpos",
            self.joint_minpos,
            "\n joint_maxpos",
            self.joint_maxpos,
            "\n joint_maxforce",
            self.joint_maxforce,
            "\n joint_maxvel",
            self.joint_maxvel,
            "\n ee_link_id",
            self.ee_link_id,
            "\n right_arm_jids_lst",
            self.arm_jids_lst,
            "\n ee_jid",
            self.ee_jid,
            "\n finger_link_ids",
            self.finger_link_ids,
            "\n finger_jids_lst",
            self.finger_jids_lst,
            "\n torso_jids_lst",
            self.torso_jids_lst,
            "\n continuous_jids_lst",
            self.continuous_jids_lst,
        )


class BulletRobot(object):
    BASE_TGT_LOWS = np.array([-1.5, -1.2])
    BASE_TGT_HIGHS = np.array([1.7, 1.2])
    BASE_SAFE_MARGIN = 0.15
    MIN_DIST_TO_FLOOR = 0.005
    ARM_MOUNTING_HEIGHT = 0 # 0.335

    def __init__(
        self,
        sim,
        robot_desc_file,
        ee_joint_name,
        ee_link_name,
        control_mode,
        base_pos,
        base_quat,
        global_scaling,
        rest_arm_pos=None,
        rest_arm_quat=None,
        use_fixed_base=False,
        use_fixed_arm=False,
        kp=0.1,
        kd=1.0,
        min_z=0.0,
        debug=False,
    ):
        assert control_mode in ["ee_position", "position", "velocity", "torque"]
        self.control_mode = control_mode
        self.kp, self.kd, self.min_z = kp, kd, min_z
        self.debug = debug
        self.sim = sim
        self.rest_arm_pos = rest_arm_pos
        self.rest_arm_quat = rest_arm_quat
        self.use_fixed_base = use_fixed_base
        self.use_fixed_arm = use_fixed_arm

        # load robot from URDF
        if not os.path.isabs(robot_desc_file):
            robot_desc_file = os.path.join(
                os.path.split(__file__)[0], "..", "envs", "data", robot_desc_file
            )
        if debug:
            print("robot_desc_file", robot_desc_file)
        self.info = self.load_robot(
            robot_desc_file,
            ee_joint_name,
            ee_link_name,
            base_pos=base_pos,
            base_quat=base_quat,
            use_fixed_base=use_fixed_base,
            use_fixed_arm=use_fixed_arm,
            global_scaling=global_scaling,
        )

        # Create a constraint for the mobile manipulator to stay on the floor.
        # Note this constraint is not as rigid as using use_fixed_base while
        # loading, but it looks okay to keep balance of the platform and could
        # allow us to move the basis as the VR pybullet example.
        self.base_cid = None
        if not use_fixed_base:
            self.base_cid = sim.createConstraint(
                self.info.robot_id,
                -1,
                -1,
                -1,
                sim.JOINT_FIXED,
                [0.0, 0, 0],
                [0.0, 0, 0],
                base_pos,
            )

        # set ik arguments
        self.rest_qpos = (self.info.joint_maxpos + self.info.joint_minpos) / 2
        self.default_ik_args = {
            "lowerLimits": self.info.joint_minpos.tolist(),
            "upperLimits": self.info.joint_maxpos.tolist(),
            "jointRanges": (self.info.joint_maxpos - self.info.joint_minpos).tolist(),
            "restPoses": self.rest_qpos.tolist(),
            "maxNumIterations": 500,
            "residualThreshold": 0.005,
        }

        # reset to initial position and visualize
        self.reset()

        # console logging
        if not self.use_fixed_arm and debug:
            ee_pos, ee_quat, *_ = self.get_ee_pos_quat_vel()
            print(
                "Initialized robot ee at pos",
                ee_pos,
                "euler quat",
                ee_quat,
                "sin/cos ori",
                quat_to_sin_cos(ee_quat),
            )

    def load_robot(
        self,
        robot_path,
        ee_joint_name,
        ee_link_name,
        base_pos,
        base_quat,
        use_fixed_base,
        use_fixed_arm,
        global_scaling,
    ):
        # with suppress_stdout():
        robot_id = self.sim.loadURDF(
            robot_path,
            basePosition=base_pos,
            baseOrientation=base_quat,
            useFixedBase=use_fixed_base,
            flags=pybullet.URDF_USE_SELF_COLLISION,
            globalScaling=global_scaling,
        )

        joint_ids, joint_names = [], []
        joint_minpos, joint_maxpos = [], []
        joint_maxforce, joint_maxvel = [], []
        ee_link_id, ee_jid = None, None
        finger_jids_lst = []
        finger_link_ids = []
        arm_jids_lst = []
        for j in range(self.sim.getNumJoints(robot_id)):
            (
                _,
                jname,
                jtype,
                _,
                _,
                _,
                _,
                _,
                jlowlim,
                jhighlim,
                jmaxforce,
                jmaxvel,
                link_name,
                _,
                _,
                _,
                _,
            ) = self.sim.getJointInfo(robot_id, j)
            jname = jname.decode("utf-8")
            link_name = link_name.decode("utf-8")
            # take care of continuous joints
            if jhighlim < jlowlim:
                jlowlim, jhighlim = -np.pi, np.pi
            if jtype in [pybullet.JOINT_REVOLUTE, pybullet.JOINT_PRISMATIC]:
                joint_ids.append(j)
                joint_names.append(jname)
                joint_minpos.append(jlowlim)
                joint_maxpos.append(jhighlim)
                joint_maxforce.append(jmaxforce * global_scaling)
                joint_maxvel.append(jmaxvel * global_scaling)
            jid = len(joint_ids) - 1  # internal index (not pybullet id)
            if jtype == pybullet.JOINT_REVOLUTE:
                arm_jids_lst.append(jid)
            if jtype == pybullet.JOINT_PRISMATIC:
                finger_jids_lst.append(jid)
            if jname == ee_joint_name:
                ee_jid = jid
            if link_name == ee_link_name:
                ee_link_id = j  # for IK
            if link_name.endswith("finger"):
                finger_link_ids.append(j)
        if not self.use_fixed_arm:
            assert ee_link_id is not None
            assert ee_jid is not None
        info = BulletRobotInfo(
            robot_id,
            np.array(joint_ids),
            np.array(joint_names),
            np.array(joint_minpos),
            np.array(joint_maxpos),
            np.array(joint_maxforce),
            np.array(joint_maxvel),
            ee_link_id,
            arm_jids_lst,
            ee_jid,
            finger_link_ids,
            finger_jids_lst,
        )
        if self.debug:
            info.print()
        return info

    def reset(self):
        self.reset_to_pose(self.rest_arm_pos, self.rest_arm_quat)

    def reset_to_pose(self, ee_pos, ee_quat):
        qpos = self.sim.calculateInverseKinematics(
            self.info.robot_id,
            self.info.ee_link_id,
            targetPosition=ee_pos,
            targetOrientation=ee_quat,
            **self.default_ik_args,
        )
        qpos = np.array(qpos)
        qpos = self.clip_qpos(qpos)
        for jid in range(self.info.dof):
            self.sim.resetJointState(
                bodyUniqueId=self.info.robot_id,
                jointIndex=self.info.joint_ids[jid],
                targetValue=qpos[jid],
                targetVelocity=0,
            )
        self.clear_motor_control()

    def clear_motor_control(self):
        # We need these to be called after every reset. This fact is not
        # documented, and bugs resulting from not zeroing-out velocity and
        # torque control values this way are hard to reproduce.
        self.sim.setJointMotorControlArray(
            self.info.robot_id,
            self.info.joint_ids.tolist(),
            pybullet.VELOCITY_CONTROL,
            targetVelocities=[0] * self.info.dof,
        )
        self.sim.setJointMotorControlArray(
            self.info.robot_id,
            self.info.joint_ids.tolist(),
            pybullet.TORQUE_CONTROL,
            forces=[0] * self.info.dof,
        )

    def set_joint_limits(self, minpos, maxpos):
        self.info.joint_minpos[:] = minpos[:]
        self.info.joint_maxpos[:] = maxpos[:]

    def get_minpos(self):
        return self.info.joint_minpos

    def get_maxpos(self):
        return self.info.joint_maxpos

    def get_maxforce(self):
        return self.info.joint_maxforce

    def get_maxvel(self):
        return self.info.joint_maxvel

    def get_max_fing_dist(self):
        farr = np.array(self.info.finger_jids_lst)
        return self.info.joint_maxpos[farr].sum()  # for symmetric commands

    def get_qpos(self):
        joint_states = self.sim.getJointStates(self.info.robot_id, self.info.joint_ids)
        qpos = [joint_states[i][0] for i in range(self.info.dof)]
        return np.array(qpos)

    def get_qvel(self):
        joint_states = self.sim.getJointStates(self.info.robot_id, self.info.joint_ids)
        qvel = [joint_states[i][1] for i in range(self.info.dof)]
        return np.array(qvel)

    def get_fing_dist(self):
        joint_states = self.sim.getJointStates(self.info.robot_id, self.info.joint_ids)
        fing_dists = [joint_states[i][0] for i in self.info.finger_jids_lst]
        return np.array(fing_dists).sum()

    def get_ee_pos(self):
        pos, _, _, _ = self.get_ee_pos_quat_vel()
        return pos

    def get_ee_pos_quat_vel(self):
        # Returns (in world coords): EE 3D position, quaternion orientation,
        # linear and angular velocity.
        ee_link_id = self.info.ee_link_id
        ee_state = self.sim.getLinkState(
            self.info.robot_id, ee_link_id, computeLinkVelocity=1
        )
        pos = np.array(ee_state[0])
        quat = np.array(ee_state[1])
        lin_vel = np.array(ee_state[6])
        ang_vel = np.array(ee_state[7])
        return pos, quat, lin_vel, ang_vel

    def _ee_pos_to_qpos_raw(self, ee_pos, ee_quat=None, fing_dist=0.0, debug=False):
        qpos = self.sim.calculateInverseKinematics(
            self.info.robot_id,
            self.info.ee_link_id,
            targetPosition=ee_pos.tolist(),
            targetOrientation=ee_quat,
            **self.default_ik_args,
        )
        qpos = np.array(qpos)
        for jid in self.info.finger_jids_lst:
            qpos[jid] = np.clip(  # finger info (not set by IK)
                fing_dist / 2.0,
                self.info.joint_minpos[jid],
                self.info.joint_maxpos[jid],
            )

        # IK will find solutions outside of joint limits, so clip.
        qpos = self.clip_qpos(qpos)
        return qpos

    def move_to_qpos(self, tgt_qpos, mode, kp=None, kd=None):
        tgt_qvel = np.zeros_like(tgt_qpos)
        self.move_to_qposvel(
            tgt_qpos,
            tgt_qvel,
            mode,
            self.kp if kp is None else kp,
            self.kd if kd is None else kd,
        )

    def reset_joint(self, jid, jpos, jvel):
        self.sim.resetJointState(
            bodyUniqueId=self.info.robot_id,
            jointIndex=self.info.joint_ids[jid],
            targetValue=jpos,
            targetVelocity=jvel,
        )

    def get_ok_qvel(self, tgt_qvel):
        ok_tgt_qvel = np.copy(tgt_qvel)
        LIM_SC = 0.95
        JOINT_MINPOS = np.copy(self.info.joint_minpos[0:7]) * LIM_SC
        JOINT_MAXPOS = np.copy(self.info.joint_maxpos[0:7]) * LIM_SC
        qpos = self.get_qpos()
        for jid in range(JOINT_MINPOS.shape[0]):
            if qpos[jid] < JOINT_MINPOS[jid] or qpos[jid] > JOINT_MAXPOS[jid]:
                ok_tgt_qvel[jid] = 0.0
                self.reset_joint(jid, qpos[jid], ok_tgt_qvel[jid])
        # Stop all motion if the robot is too close to the ground/table.
        ee_pos = self.get_ee_pos()
        if ee_pos[2] <= self.min_z:
            ok_tgt_qvel = None
            for jid in range(JOINT_MINPOS.shape[0]):
                self.reset_joint(jid, qpos[jid], 0.0)
        return ok_tgt_qvel

    def move_with_qvel(self, tgt_qvel, mode, kp=None, kd=None):
        if kp is None:
            kp = self.kp
        if kd is None:
            kd = self.kd
        tgt_qpos = np.zeros_like(tgt_qvel)
        ok_tgt_qvel = self.get_ok_qvel(tgt_qvel)
        if ok_tgt_qvel is not None:
            self.move_to_qposvel(tgt_qpos, ok_tgt_qvel, mode, kp, kd)
        else:
            self.move_to_qposvel(tgt_qpos, np.zeros_like(tgt_qvel), mode, kp, kd)

    def move_to_qposvel(self, tgt_qpos, tgt_qvel, mode, kp, kd):
        assert mode in [
            pybullet.POSITION_CONTROL,
            pybullet.VELOCITY_CONTROL,
            pybullet.PD_CONTROL,
        ]
        kps = kp if type(kp) == list else [kp] * self.info.dof
        kds = kd if type(kd) == list else [kd] * self.info.dof
        rbt_tgt_qpos = self.clip_qpos(tgt_qpos)
        rbt_tgt_qvel = np.clip(
            tgt_qvel, -1.0 * self.info.joint_maxvel, self.info.joint_maxvel
        )
        # PD example: https://github.com/bulletphysics/bullet3/issues/2152
        # ATTENTION: it is extremely important to set maximum forces when
        # executing PD control. This is not documented, but PyBullet seems
        # to have a memory corruption problem (when high torques are
        # applied the simulation can get permanently and silently corrupted
        # without any warnings). Save/restore state does not help, need to
        # delete and re-instantiate the whole simulation.
        if mode == pybullet.POSITION_CONTROL:
            self.sim.setJointMotorControlArray(
                bodyUniqueId=self.info.robot_id,
                jointIndices=self.info.joint_ids.tolist(),
                targetPositions=rbt_tgt_qpos,
                targetVelocities=rbt_tgt_qvel,
                controlMode=pybullet.POSITION_CONTROL,
                positionGains=kps,  # e.g. 0.1
                velocityGains=kds,  # e.g. 1.0
                forces=self.info.joint_maxforce,
            )  # see page 22 of pybullet docs
        elif mode == pybullet.VELOCITY_CONTROL:
            self.sim.setJointMotorControlArray(
                bodyUniqueId=self.info.robot_id,
                jointIndices=self.info.joint_ids.tolist(),
                targetPositions=rbt_tgt_qpos,
                targetVelocities=rbt_tgt_qvel,
                controlMode=pybullet.VELOCITY_CONTROL,
                positionGains=kps,  # e.g. 0.1
                velocityGains=kds,  # e.g. 1.0
                forces=self.info.joint_maxforce,
            )  # see page 22 of pybullet docs
        else:  # PD_CONTROL
            self.sim.setJointMotorControlArray(
                bodyUniqueId=self.info.robot_id,
                jointIndices=self.info.joint_ids.tolist(),
                targetPositions=rbt_tgt_qpos.tolist(),
                targetVelocities=rbt_tgt_qvel.tolist(),
                controlMode=pybullet.PD_CONTROL,
                positionGains=kps,  # e.g. 100.0
                velocityGains=kds,  # e.g. 10.0
                forces=self.info.joint_maxforce.tolist(),
            )  # see docs page 22
        self.obey_joint_limits()

    def move_to_ee_pos(
        self,
        tgt_ee_pos,
        tgt_ee_quat=None,
        fing_dist=0.0,
        mode=pybullet.POSITION_CONTROL,
        kp=None,
        kd=None,
        debug=True,
    ):
        qpos = None
        num_tries = 10
        if tgt_ee_quat is None:
            _, tgt_ee_quat, _, _ = self.get_ee_pos_quat_vel()
        for i in range(num_tries):
            qpos = self.ee_pos_to_qpos(tgt_ee_pos, tgt_ee_quat, fing_dist)
            if qpos is not None:
                # ok solution found
                break
        if qpos is None:
            if debug:
                print("ee pos not good:", tgt_ee_pos, tgt_ee_quat)
        else:
            self.move_to_qpos(qpos, mode=mode, kp=kp, kd=kd)

    def action_low_high_ranges(self):
        if self.control_mode == "ee_position":
            # EE pos, quat, fing dist
            low = np.array([-1, -1, 0, -1, -1, -1, -1, 0.0])
            high = np.array([1, 1, 1, 1, 1, 1, 1, self.get_max_fing_dist()])
        elif self.control_mode == "position":
            low = self.get_minpos()
            high = self.get_maxpos()
        elif self.control_mode == "velocity":
            low = -self.get_maxvel()
            high = self.get_maxvel()
        elif self.control_mode == "torque":
            low = -self.get_maxforce()
            high = self.get_maxforce()
        else:
            assert ValueError("unknown control mode")
        return low, high

    def apply_joint_torque(self, torque, compensate_gravity=True):
        if np.allclose(torque, 0):
            return

        torque = np.copy(torque)
        if compensate_gravity:
            gcomp_torque = self.inverse_dynamics(np.zeros_like(torque))
            torque += gcomp_torque

        # clip check and command torques
        torque = np.clip(
            torque, -1.0 * self.info.joint_maxforce, self.info.joint_maxforce
        )
        self.sim.setJointMotorControlArray(
            bodyIndex=self.info.robot_id,
            jointIndices=self.info.joint_ids,
            controlMode=pybullet.TORQUE_CONTROL,
            forces=torque.tolist(),
        )

    def get_ee_jacobian(self):
        qpos = self.get_qpos()
        qvel = self.get_qvel()
        ee_link_id = self.info.ee_link_id
        J_lin, J_ang = self.sim.calculateJacobian(
            bodyUniqueId=self.info.robot_id,
            linkIndex=ee_link_id,
            localPosition=[0, 0, 0],
            objPositions=qpos.tolist(),
            objVelocities=qvel.tolist(),
            objAccelerations=[0] * self.info.dof,
        )
        return np.array(J_lin), np.array(J_ang)

    def inverse_dynamics(self, des_acc):
        qpos = self.get_qpos()
        qvel = self.get_qvel()
        torques = self.sim.calculateInverseDynamics(
            self.info.robot_id, qpos.tolist(), qvel.tolist(), des_acc.tolist()
        )
        return np.array(torques)

    def clip_qpos(self, qpos):
        if (qpos >= self.info.joint_minpos).all() and (
            qpos <= self.info.joint_maxpos
        ).all():
            return qpos
        clipped_qpos = np.copy(qpos)
        for jid in range(self.info.dof):
            jpos = qpos[jid]
            if jpos < self.info.joint_minpos[jid]:
                jpos = self.info.joint_minpos[jid]
            if jpos > self.info.joint_maxpos[jid]:
                jpos = self.info.joint_maxpos[jid]
            clipped_qpos[jid] = jpos
        return clipped_qpos

    def obey_joint_limits(self):
        qpos = self.get_qpos()
        clipped_qpos = self.clip_qpos(qpos)
        neq_ids = np.nonzero(qpos - clipped_qpos)[0]
        for jid in neq_ids:
            if self.debug:
                print("fix jid", jid, qpos[jid], "->", clipped_qpos[jid])
            self.sim.resetJointState(
                bodyUniqueId=self.info.robot_id,
                jointIndex=self.info.joint_ids[jid],
                targetValue=clipped_qpos[jid],
                targetVelocity=0,
            )

    def ee_pos_to_qpos(self, ee_pos, ee_quat, fing_dist=0.0, debug=False):
        qpos = self._ee_pos_to_qpos_raw(ee_pos, ee_quat, fing_dist, debug=debug)
        return qpos

    def get_base_pose_vel(self):
        # Returns 2D base position and 1D rotation angle.
        assert self.base_cid is not None
        base_state = self.sim.getBasePositionAndOrientation(self.info.robot_id)
        base_pos = np.array(base_state[0])
        base_ori = self.sim.getEulerFromQuaternion(base_state[1])
        lin_vel, ang_vel = self.sim.getBaseVelocity(self.info.robot_id)
        return np.array([*base_pos[0:2], base_ori[2]]), lin_vel, ang_vel

    def get_base_pose(self):
        return self.get_base_pose_vel()[0]

    def move_base(self, xy_and_rot):
        # Input: [x, y, rot] (rot in radians)
        # Change fixed constraint spec to move the base: assuming the mobile
        # base is omnidirectional, we can also animate differential drive by
        # having the wheel distance.
        base_pose = self.sim.getBasePositionAndOrientation(self.info.robot_id)
        curr_quat = base_pose[1]
        curr_z = 0.05  # base_pose[0][2]
        target_base_pos = [*xy_and_rot[0:2], curr_z]
        target_rot = xy_and_rot[2]
        target_base_quat = pybullet.getQuaternionFromEuler([0, 0, target_rot])
        if np.dot(curr_quat, target_base_quat) < 0:
            target_base_quat *= -1
        assert self.base_cid is not None
        self.sim.changeConstraint(
            self.base_cid, target_base_pos, target_base_quat, maxForce=2000
        )
        # self.sim.resetBasePositionAndOrientation(    # TODO: remove
        #     self.info.robot_id, target_base_pos, target_base_quat)

    def open_gripper(self):
        qpos = self.get_qpos()
        for i in self.info.finger_jids_lst:
            qpos[i] = self.info.joint_maxpos[i]
        self.move_to_qpos(qpos, pybullet.POSITION_CONTROL)

    def close_gripper(self):
        qpos = self.get_qpos()
        for i in self.info.finger_jids_lst:
            qpos[i] = self.info.joint_minpos[i]
        self.move_to_qpos(qpos, pybullet.POSITION_CONTROL)

    def arm_home(self):
        pass
