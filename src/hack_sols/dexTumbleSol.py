import enum

from sapien.core import Pose
from transforms3d import euler
import numpy as np

from agent import PandaArm, Box
from env.dexEnv import DexEnv, ARM_NAME, BOX_NAME
from hack_sols import DexHackySolution
from sapien_interfaces import Task
from task.dexTumbleTask import DexTumbleTask


class Phase(enum.Enum):
    rotate_toward_box = -1
    move_above_box = 0
    move2box = 1
    hold_box = 2
    tumble = 3
    openGripper = 4
    evaluation = 5

    idle = 100


class DexTumbleSolution(DexHackySolution):
    def __init__(self):
        super().__init__()
        self.phase = Phase.rotate_toward_box
        self.init_control = True
        self.tg_pose = None

        self.up_right_q = euler.euler2quat(np.pi, 0, 0)
        self.target_count = 0

        self.grab_pos = 0.015
        self.box2grip_dist = 0.1
        self.ini_box_edg_tumble = None
        self.tumble_intervs = 40
        self.tumble_phase = 0

    def init(self, env: DexEnv, task: Task):
        super().init(env, task)
        self.phase = Phase.rotate_toward_box
        self.init_control = True
        self.tg_pose = None

        robot = env.agents[ARM_NAME]
        self.prep_drive(robot)

    def prep_drive(self, robot):
        super().prep_arm_drive(robot)

        self.init_control = True
        self.tg_pose = None
        self.target_count = 0

    def before_step(self, env, task):
        robot: PandaArm = env.agents[ARM_NAME]
        box: Box = env.agents[BOX_NAME]

        if self.phase is Phase.rotate_toward_box:
            if self.init_control is True:
                box2base = robot.observation['poses'][0].inv() * box.observation['pose']
                p = box2base.p
                theta = np.arctan(p[1] / p[0])
                if p[0] < 0 and p[1] < 0:
                    theta -= np.pi
                elif p[0] < 0 and p[1] > 0:
                    theta += np.pi

                self.init_control = False

                self.tg_pose = Pose(robot.observation['poses'][0].p, euler.euler2quat(0, 0, theta))

            self.drive_to_pose(robot, self.tg_pose, joint_index=1, theta_thresh=1e-4, theta_abs_thresh=1e-5)

            if self.drive[robot] is False:
                self.phase = Phase.move_above_box
                self.prep_drive(robot)
        elif self.phase is Phase.move_above_box:
            if self.init_control is True:
                self.tg_pose = robot.observation['poses'][-1]
                self.tg_pose.set_p(box.observation['pose'].p)
                self.tg_pose = Pose([0, 0, 0.2 + box.box_size]) * self.tg_pose
                self.init_control = False

            self.drive_to_pose(robot, self.tg_pose)

            if self.drive[robot] is False:
                self.phase = Phase.move2box
                self.prep_drive(robot)
        elif self.phase is Phase.move2box:
            if self.init_control is True:
                box_pose = box.observation['pose']
                self.tg_pose = Pose([0,0,self.box2grip_dist]) * box_pose * Pose([0]*3, euler.euler2quat(np.pi,0,0))
                self.init_control = False

            self.drive_to_pose(robot, self.tg_pose, override=([-1, -2], [0.6, 0.6], [0, 0]))

            if self.drive[robot] is False:
                self.phase = Phase.hold_box
                self.prep_drive(robot)
        elif self.phase is Phase.hold_box:
            if self.init_control is True:
                self.init_control = False
                self.target_count = env.current_step + 50 * env.n_substeps

            if env.current_step < self.target_count:
                qpos = robot.observation['qpos']
                qvel = [0] * robot.dof
                qvel[-2:] = np.clip(qpos[-2:] - self.grab_pos, 0, 0.1)
                qpos[-2:] = self.grab_pos
                pf = robot.get_compute_functions()['passive_force']()
                robot.set_action(qpos, qvel, pf)
            else:
                self.phase = Phase.tumble
                box2edge = Pose([box.box_size, 0, -box.box_size])
                self.ini_box_edg_tumble = box.observation['pose'] * box2edge
                self.prep_drive(robot)
                self.tumble_phase = 0
        elif self.phase is Phase.tumble:
            box2edge = Pose([box.box_size, 0, -box.box_size])

            if self.init_control:
                theta = self.tumble_phase * np.pi / 2 / self.tumble_intervs
                self.tg_pose = self.ini_box_edg_tumble * Pose([0]*3, euler.euler2quat(0, theta, 0)) * box2edge.inv() * Pose([0,0,self.box2grip_dist], euler.euler2quat(np.pi,0,0))
                self.init_control = False

            self.drive_to_pose(robot, self.tg_pose, override=([-1, -2], [self.grab_pos, self.grab_pos], [0,0]),
                               theta_abs_thresh=1e-3, dist_abs_thresh=1e-5, max_v=0.05, max_w=np.pi/3)

            if self.drive[robot] is False:
                self.tumble_phase += 1
                self.prep_drive(robot)

                if self.tumble_phase > 35:
                    self.phase = Phase.openGripper
        elif self.phase is Phase.openGripper:
            if self.init_control is True:
                self.init_control = False
                self.target_count = env.current_step + 50 * env.n_substeps

            if env.current_step < self.target_count:
                qpos = robot.observation['qpos']
                qvel = [0] * robot.dof
                qvel[-2:] = np.clip(qpos[-2:] - 0.06, 0, 0.1)
                qpos[-2:] = 0.06
                pf = robot.get_compute_functions()['passive_force']()
                robot.set_action(qpos, qvel, pf)
            else:
                self.phase = Phase.evaluation
        elif self.phase is Phase.evaluation:
            print(task.get_trajectory_status())
            self.phase = Phase.idle



if __name__ == '__main__':
    np.random.seed(6)
    env = DexEnv()
    task = DexTumbleTask()
    env.add_task(task)
    sol = DexTumbleSolution()
    task.register_slotion(sol)
    sol.init(env, task)

    while not env.should_quit:
        env.step()
