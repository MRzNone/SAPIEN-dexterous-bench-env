import enum
import numpy as np
from sapien.core import Pose
from transforms3d import euler

from agent import PandaArm, Box
from env.dexEnv import DexEnv, ARM_NAME, BOX_NAME
from hack_sols import DexHackySolution
from sapien_interfaces import Task
from task.dexPushTask import DexPushTask
from task.dexSlidingTask import DexSlidingTask


class Phase(enum.Enum):
    rotate_toward_box = -1
    move2above = 0
    move2box = 1
    push2goal = 2
    moveUp = 3
    evaluate = 4
    done = 5


class DexSlideSolution(DexHackySolution):
    def __init__(self):
        super().__init__()
        self.phase = Phase.rotate_toward_box
        self.init_control = True
        self.tg_pose = None
        self.plan2wd = None
        self.goal2wd = None
        self.push2wd = None

        self.up_right_q = euler.euler2quat(np.pi, 0, 0)

    def init(self, env: DexEnv, task: Task):
        super().init(env, task)
        self.phase = Phase.rotate_toward_box
        self.init_control = True
        self.tg_pose = None
        self.goal2wd = task.parameters['goal_pos']
        self.plan2wd = None
        self.push2wd = None

        robot = env.agents[ARM_NAME]
        self.prep_drive(robot)

    def prep_drive(self, robot):
        super().prep_arm_drive(robot)

        self.init_control = True
        self.tg_pose = None

    def before_step(self, env, task: DexSlidingTask):
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
                self.phase = Phase.move2above
                self.prep_drive(robot)
        elif self.phase is Phase.move2above:
            if self.init_control is True:
                box2wd = box.observation['pose']
                goal2box = box2wd.inv() * Pose(self.goal2wd)
                robot2box = box2wd.inv() * robot.observation['poses'][0]

                direct = np.array([0, 0, 0])
                if np.min(np.abs(goal2box.p)) > 5e-3:
                    indx = np.argmin(np.abs(robot2box.p[:2]))
                    if np.abs(goal2box.p[indx]) < 8e-4:
                        indx = 1 - indx
                else:
                    indx = np.argmax(np.abs(goal2box.p[:2]))
                direct[indx] = 1
                dist = goal2box.p @ direct
                dist = np.clip(abs(dist), 0.05, 0.2) * np.sign(dist)

                quat = Pose([0]*3, self.up_right_q)
                if direct[1] == 1:
                    quat = quat * Pose([0]*3, euler.euler2quat(0, 0, np.pi/2))
                ee2pt = Pose([0,0,0.07], quat.q)

                self.plan2wd = box2wd * Pose(dist * direct) * ee2pt
                self.push2wd = box2wd * ee2pt
                self.tg_pose = Pose([0,0,0.07]) * box2wd * ee2pt

                self.init_control = False

            self.drive_to_pose(robot, self.tg_pose, override=([-1, -2], [0, 0], [0, 0]), dist_abs_thresh=1e-6, dist_thresh=1e-6, theta_abs_thresh=1e-3, theta_thresh=1e-3)

            if self.drive[robot] is False:
                self.phase = Phase.move2box
                self.prep_drive(robot)
        elif self.phase is Phase.move2box:
            if self.init_control is True:
                self.tg_pose = self.push2wd

            self.drive_to_pose(robot, self.tg_pose, override=([-1, -2], [0, 0], [0, 0]), dist_abs_thresh=1e-5, dist_thresh=1e-5, theta_abs_thresh=1e-3, theta_thresh=1e-3)

            if self.drive[robot] is False:
                self.prep_drive(robot)
                self.phase = Phase.push2goal
        elif self.phase is Phase.push2goal:
            if self.init_control is True:
                self.tg_pose = self.plan2wd

            self.drive_to_pose(robot, self.tg_pose, override=([-1, -2], [0, 0], [0, 0]), max_v=0.05, dist_abs_thresh=1e-5, dist_thresh=1e-5, theta_abs_thresh=1e-3, theta_thresh=1e-3)

            if self.drive[robot] is False or\
                    task.get_trajectory_status()['tilt_theta'] > task.parameters['box_tilt_threshold_rad'] * 0.9 or\
                    task.get_trajectory_status()['succeeded'] is True:
                self.prep_drive(robot)
                self.phase = Phase.moveUp
        elif self.phase is Phase.moveUp:
            if self.init_control is True:
                self.tg_pose = Pose([0,0,0.1]) * robot.observation['poses'][-3]
                self.init_control = False

            self.drive_to_pose(robot, self.tg_pose, override=([-1, -2], [0, 0], [0, 0]), dist_abs_thresh=1e-4)

            if self.drive[robot] is False:
                self.prep_drive(robot)
                status = task.get_trajectory_status()

                if status['succeeded'] is False and status['valid_slide'] is True:
                    self.phase = Phase.move2above
                else:
                    self.phase = Phase.evaluate
        elif self.phase is Phase.evaluate:
            print(task.get_trajectory_status())
            self.phase = Phase.done


if __name__ == '__main__':
    np.random.seed(26546)
    env = DexEnv()
    task = DexSlidingTask()
    env.add_task(task)
    sol = DexSlideSolution()
    task.register_slotion(sol)
    sol.init(env, task)

    while not env.should_quit:
        env.step()
