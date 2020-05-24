import numpy as np
from sapien.core import Pose
import sapien.core as sapien
from transforms3d.quaternions import quat2mat

from agent import Box, PandaArm
from env.dexEnv import BOX_NAME, DexEnv, ARM_NAME
from sapien_interfaces import Task, Env

TASK_NAME = "Dexterous Box Sliding"
GOAl_POS = np.array([0.5, 0.3, -1])
TILT_THRESHHOLD = np.pi / 180 * 20
BOX_AT_GOAL_THRESHOLD = 5e-3
BOX_HEIGHT_THRESHOLD = 2e-2

FAIL_REASON_TILT = "box was tilted"
FAIL_REASON_LIFTED = "box was lifted off-ground"
FAIL_REASON_TOUCHED_SIDES = "touched box at side"


def build_goal(scene, side_len) -> sapien.Actor:
    size = [side_len, side_len, 0.001]
    builder = scene.create_actor_builder()
    builder.add_box_visual(size=size, color=[0, 1, 0, 0.2])
    return builder.build(True, "Goal")


class DexSlidingTask(Task):

    def __init__(self):
        self._box_valid_push = True
        self._suceeded = False

        self._box: Box = None
        self._arm: PandaArm = None
        self._goal: sapien.Actor = None
        self._box_height_thresh = None
        self._tracking = True
        self._failed_reason = None

        self._initialized = False
        self._parameters = None

    def init(self, env: Env) -> None:
        if self._initialized:
            print(f"Task: {TASK_NAME} already initialized")
            return

        # add box
        self._box = env.agents[BOX_NAME]
        assert self._box is not None and isinstance(self._box, Box), \
            f"{BOX_NAME} does not exist in env"
        self._box_height_thresh = self._box.observation['pose'].p[2] + BOX_HEIGHT_THRESHOLD

        # add arm
        self._arm = env.agents[ARM_NAME]
        assert self._arm is not None and isinstance(self._arm, PandaArm), \
            f"{ARM_NAME} does not exist in env"

        # build goal
        self._goal = build_goal(env.scene, self._box.box_size)
        self._goal.set_pose(Pose(GOAl_POS))
        self._box_valid_push = True
        self._suceeded = False
        self._tracking = True
        self._failed_reason = None

        self._parameters = {
            "box_size": self._box.box_size,
            "box_tilt_threshold_rad": TILT_THRESHHOLD,
            "box_at_goal_threshold_meter": BOX_AT_GOAL_THRESHOLD,
            "box_height_threshold_global": self._box_height_thresh,
            "goal_pos": GOAl_POS,
        }

        self._initialized = True
        print(f"Task: {TASK_NAME} initialized")

    def reset(self, env) -> None:
        self._box.reset(env)
        self._arm.reset(env)

    def before_step(self, env) -> None:
        pass

    def after_step(self, env) -> None:
        pass

    def before_substep(self, env) -> None:
        pass

    def after_substep(self, env: Env) -> None:
        assert self._initialized, "task not initialized"

        if self._box_valid_push is False or self._suceeded is True or not self._tracking:
            self._tracking = False
            return

        box_pose = self._box.observation['pose']
        # calcilate how much "flipped"
        up_right_pose = np.array([0, 0, 1])
        rot_mat = quat2mat(box_pose.q)
        box_up = rot_mat[:, 2]
        box_up /= np.linalg.norm(box_up)

        theta = np.arccos(box_up @ up_right_pose)

        if theta > TILT_THRESHHOLD:
            self._box_valid_push = False
            self._failed_reason = FAIL_REASON_TILT
            return
        elif box_pose.p[2] > self._box_height_thresh:
            self._box_valid_push = False
            self._failed_reason = FAIL_REASON_LIFTED
            return

        # check contacts all at top
        contacts = env.scene.get_contacts()

        def if_valid_contact(c):
            s = {c.actor1.name, c.actor2.name}
            return 'ground' not in s and BOX_NAME in s and c.separation < 1e-4
        box_center = box_pose.p
        valid_pos = np.array([c.point for c in contacts if if_valid_contact(c)])

        if len(valid_pos) > 0:
            valid_pos_to_box_center = valid_pos - box_center
            valid_pos_local_z = valid_pos_to_box_center @ box_up
            if np.any(valid_pos_local_z < self._box.box_size - 1e-4):
                self._box_valid_push = False
                self._failed_reason = FAIL_REASON_TOUCHED_SIDES
                return

        # check if succeeded
        dist = np.linalg.norm(GOAl_POS[:2] - box_pose.p[:2])
        if dist < BOX_AT_GOAL_THRESHOLD:
            self._suceeded = True

    @property
    def parameters(self) -> dict:
        return self._parameters

    def get_trajectory_status(self) -> dict:
        assert self._initialized, "task not initialized"

        status = {
            "succeeded": self._suceeded,
            "valid_push": self._box_valid_push,
            "tracking": self._tracking
        }

        if not self._suceeded and not self._tracking:
            status['failed_reason'] = self._failed_reason

        return status


if __name__ == '__main__':
    env = DexEnv()
    task = DexSlidingTask()
    env.add_task(task)

    i = 0

    while not env.should_quit:
        env.step()
        print(task.get_trajectory_status())
