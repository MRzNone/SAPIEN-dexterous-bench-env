from collections import defaultdict

from sapien.core import Pose
import numpy as np
from transforms3d import quaternions as quat, euler

from agent import PandaArm
from env.dexEnv import DexEnv, ARM_NAME
from sapien_interfaces import Task, Solution
from task.dexTapTask import DexTapTask


def comp_pose(pose1: Pose, pose2: Pose):
    p1 = pose1.p
    p2 = pose2.p

    dist = np.linalg.norm(p1 - p2)

    q1 = pose1.q
    q2 = pose2.q
    d = q1 @ q2
    d = np.clip(d, -1, 1)
    theta = np.arccos(d)

    return dist, theta


class DexHackySolution(Solution):
    def __init__(self):
        self.robot = None
        self.timestep = None
        self.skip_frame = None
        self.task = None

        self.hold_qpos = defaultdict(lambda: None)
        self.st = defaultdict(lambda: 0)
        self.drive = defaultdict(lambda: True)
        self.last_pose = defaultdict(lambda: None)
        self.not_move = defaultdict(lambda: 0)

    def init(self, env: DexEnv, task: Task):
        self.timestep = env.timestep
        self.skip_frame = env.n_substeps
        self.task = task

        robot = env.agents[ARM_NAME]
        self.prep_arm_drive(robot)

    def prep_arm_drive(self, robot):
        self.hold_qpos = defaultdict(lambda: None)
        self.st = defaultdict(lambda: 0)
        self.drive = defaultdict(lambda: True)
        self.last_pose = defaultdict(lambda: None)
        self.not_move = defaultdict(lambda: 0)
        ps = np.array([1000] * 9)
        ds = np.array([400] * 9)
        robot.configure_controllers(ps, ds)

    def diff_drive(self, robot: PandaArm, index, target_pose, max_v=0.1, max_w=np.pi, active_joints_ids=[], override=None):
        obs = robot.observation
        qpos, qvel, poses = obs['qpos'], obs['qvel'], obs['poses']
        current_pose: Pose = poses[index]
        delta_p = target_pose.p - current_pose.p
        delta_q = quat.qmult(target_pose.q, quat.qinverse(current_pose.q))

        axis, theta = quat.quat2axangle(delta_q)
        if theta > np.pi:
            theta -= np.pi * 2

        t1 = np.linalg.norm(delta_p) / max_v
        t2 = theta / max_w
        t = max(np.abs(t1), np.abs(t2), 0.01)
        thres = 0.1
        if t < thres:
            k = (np.exp(thres) - 1) / thres
            t = np.log(k * t + 1)
        v = delta_p / t
        w = theta / t * axis
        index = index if index >= 0 else len(robot._robot.get_joints()) + index
        cal_qvel = robot.get_compute_functions()['cartesian_diff_ik'](np.concatenate((v, w)), index, active_joints_ids)
        target_qvel = np.zeros_like(qvel)
        indx = np.arange(len(qvel)) if active_joints_ids == [] else active_joints_ids
        target_qvel[indx] = cal_qvel

        pf = robot.get_compute_functions()['passive_force']()

        if override is not None:
            override_indx, override_qpos, override_qvel = override
            qpos[override_indx] = override_qpos
            target_qvel[override_indx] = override_qvel

        robot.set_action(qpos, target_qvel, pf)

    def hold(self, robot: PandaArm):
        pf = robot.get_compute_functions()['passive_force']()

        # ps = [6000] * 7
        # ds = [3000] * 7
        #
        # robot.configure_controllers(ps, ds)

        if self.hold_qpos[robot] is None:
            self.hold_qpos[robot] = robot.observation['qpos']

        robot.set_action(self.hold_qpos[robot], [0] * robot.dof, pf)

    def drive_to_pose(self, robot: PandaArm, target_pose, dist_thresh=5e-4, theta_thresh=1e-3,
                      dist_abs_thresh=1e-1, theta_abs_thresh=1e-2,
                      max_v=0.1, max_w=np.pi, joint_index=-3, active_joints_ids=[], max_iters=2000,
                      override=None):
        if self.st[robot] < max_iters:
            if self.drive[robot]:
                self.diff_drive(robot, joint_index, target_pose, max_v, max_w, active_joints_ids, override)
            else:
                self.hold(robot)

            self.st[robot] += 1

        pose = robot.observation['poses'][joint_index]

        # check not moving
        if self.last_pose[robot] is not None and self.drive[robot] and self.st[robot] > 10:
            dist1, theta1 = comp_pose(pose, self.last_pose[robot])

            if dist1 < dist_thresh and theta1 < theta_thresh:
                self.not_move[robot] += 1
            else:
                self.not_move[robot] = 0

        if self.st[robot] < max_iters and self.not_move[robot] < 5:
            dist2, theta2 = comp_pose(pose, target_pose)

            if dist2 > dist_abs_thresh or theta2 > theta_abs_thresh:
                self.drive[robot] = True
        else:
            self.drive[robot] = False

        self.last_pose[robot] = pose
