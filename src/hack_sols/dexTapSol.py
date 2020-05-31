import enum

from sapien.core import Pose
from transforms3d import euler
import numpy as np

from agent import PandaArm, Box
from env.dexEnv import DexEnv, ARM_NAME, BOX_NAME
from hack_sols import DexHackySolution
from sapien_interfaces import Task
from task.dexTapTask import DexTapTask


class Phase(enum.Enum):
    rotate_toward_box = -1
    move_above_box = 0
    tap_down = 1
    tap_up = 2
    evaluate = 3
    done = 4


class DexTapSolution(DexHackySolution):
    def __init__(self):
        super().__init__()
        self.phase = Phase.rotate_toward_box
        self.init_control = True
        self.tg_pose = None
        self.count = 0

        self.up_right_q = euler.euler2quat(np.pi, 0, 0)

    def init(self, env: DexEnv, task: Task):
        super().init(env, task)
        self.phase = Phase.rotate_toward_box
        self.init_control = True
        self.tg_pose = None
        self.count = 0

        robot = env.agents[ARM_NAME]
        self.prep_drive(robot)

    def prep_drive(self, robot):
        super().prep_arm_drive(robot)

        self.init_control = True
        self.tg_pose = None
        self.count = 0

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

            self.drive_to_pose(robot, self.tg_pose, override=([-1, -2], [0, 0.6], [0, 0]))

            if self.drive[robot] is False:
                self.phase = Phase.tap_down
                self.prep_drive(robot)
        elif self.phase is Phase.tap_down:
            if self.init_control is True:
                self.tg_pose = robot.observation['poses'][-1]
                self.tg_pose.set_p(box.observation['pose'].p)
                self.tg_pose = Pose([0, 0, 0.1 + box.box_size]) * self.tg_pose
                self.init_control = False
                self.count = round(0.5 / self.timestep / env.n_substeps)

            self.drive_to_pose(robot, self.tg_pose, override=([-1, -2], [0, 0.6], [0, 0]), max_v=5e-2)

            if task.get_trajectory_status()['touched'] is True:
                self.phase = Phase.tap_up
                self.prep_drive(robot)
        elif self.phase is Phase.tap_up:
            if self.init_control is True:
                self.tg_pose = robot.observation['poses'][-1]
                self.tg_pose.set_p(box.observation['pose'].p)
                self.tg_pose = Pose([0, 0, 0.2 + box.box_size]) * self.tg_pose
                self.init_control = False

            self.drive_to_pose(robot, self.tg_pose, override=([-1, -2], [0, 0.6], [0, 0]))

            if self.drive[robot] is False:
                self.phase = Phase.evaluate
                self.prep_drive(robot)
        elif self.phase is Phase.evaluate:
            print(task.get_trajectory_status())
            self.phase = Phase.done


if __name__ == '__main__':
    env = DexEnv()
    task = DexTapTask()
    env.add_task(task)
    sol = DexTapSolution()
    task.register_slotion(sol)
    sol.init(env, task)

    while not env.should_quit:
        env.step()
