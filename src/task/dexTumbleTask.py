import numpy as np
from sapien.core import Pose
import sapien.core as sapien
from transforms3d.quaternions import quat2mat

from agent import Box, PandaArm
from env.dexEnv import BOX_NAME, DexEnv, ARM_NAME
from sapien_interfaces import Task, Env

TASK_NAME = "Dexterous Box Push"
TILT_THRESHOLD = np.pi / 180 * 10
FULL_TUMBLE_THRESHOLD = np.pi / 180 * 85
BOX_EDGE_DISP_THRESHOLD = 3e-3

FAIL_REASON_TOO_HIGH = "box was lifted too high off ground"
FAIL_REASON_EDGE_MOVED = "flipping edge moved during tumbling"
FAIL_REASON_DIAG_FLIP = "box was flipped not along the sides"


class DexTumbleTask(Task):

    def __init__(self):
        self._box_valid_tumble = True
        self._suceeded = False

        self._box: Box = None
        self._arm: PandaArm = None
        self._goal: sapien.Actor = None
        self._box_height_thresh = None
        self._tracking = True
        self._last_edge_pt1 = None
        self._last_edge_pt2 = None
        self._last_basis = None
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

        # add arm
        self._arm = env.agents[ARM_NAME]
        assert self._arm is not None and isinstance(self._arm, PandaArm), \
            f"{ARM_NAME} does not exist in env"

        self._box_valid_tumble = True
        self._suceeded = False
        self._tracking = True
        self._last_edge_pt1 = None
        self._last_edge_pt2 = None
        self._box_height_thresh = self._box.observation['pose'].p[-1] + self._box.box_size * 1.5
        self._last_basis = None
        self._failed_reason = None

        self._parameters = {
            "box_size": self._box.box_size,
            "box_tilt_threshold_rad": TILT_THRESHOLD,
            "box_height_threshold_global": self._box_height_thresh,
            "box_edge_center_displace_threshold": BOX_EDGE_DISP_THRESHOLD,
            "box_full_tumble_threshold_rad": FULL_TUMBLE_THRESHOLD,
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

    def after_substep(self, env) -> None:
        assert self._initialized, "task not initialized"

        if self._box_valid_tumble is False or self._suceeded or not self._tracking:
            self._tracking = False
            return

        box_pose = self._box.observation['pose']
        # calculate how much "flipped"
        rot_mat = quat2mat(box_pose.q)
        box_up = rot_mat[:3, 2]
        box_up /= np.linalg.norm(box_up)

        # check height
        if box_pose.p[2] > self._box_height_thresh:
            self._suceeded = False
            self._box_valid_tumble = False
            self._failed_reason = FAIL_REASON_TOO_HIGH
            return

        tiled = np.arccos(box_up[2]) > TILT_THRESHOLD

        theta_x, theta_y = np.pi / 2, np.pi / 2

        if tiled:
            theta_x, theta_y = np.arccos(self._last_basis @ box_up)
        else:
            self._last_basis = rot_mat[:3, :2]
            self._last_basis[2, :] = 0
            self._last_basis /= np.linalg.norm(self._last_basis, axis=0)
            self._last_basis = self._last_basis.T

        tumble_x = theta_x < np.pi / 2 - TILT_THRESHOLD
        tumble_y = theta_y < np.pi / 2 - TILT_THRESHOLD

        # check tumble
        self._box_valid_tumble = not (tumble_x and tumble_y)

        if self._box_valid_tumble is False:
            self._suceeded = False
            self._failed_reason = FAIL_REASON_DIAG_FLIP
            return

        # check edge center
        if tumble_x or tumble_y:
            center2pt1 = np.array([0, 0, -self._box.box_size])
            center2pt2 = center2pt1.copy()

            if tumble_x:
                center2pt1[0] = np.sign(theta_x) * self._box.box_size
                center2pt1[1] = self._box.box_size/2

                center2pt2[0] = np.sign(theta_x) * self._box.box_size
                center2pt2[1] = -self._box.box_size/2
            else:
                center2pt1[1] = np.sign(theta_y) * self._box.box_size
                center2pt1[0] = self._box.box_size/2

                center2pt2[1] = np.sign(theta_y) * self._box.box_size
                center2pt2[0] = -self._box.box_size/2

            center2pt1_pose = Pose(center2pt1)
            edge_pt1 = (box_pose * center2pt1_pose).p
            center2pt2_pose = Pose(center2pt2)
            edge_pt2 = (box_pose * center2pt2_pose).p

            if self._last_edge_pt1 is not None and self._last_edge_pt2 is not None:
                dist1 = np.linalg.norm(self._last_edge_pt1 - edge_pt1)
                dist2 = np.linalg.norm(self._last_edge_pt2 - edge_pt2)

                if dist1 > BOX_EDGE_DISP_THRESHOLD or dist2 > BOX_EDGE_DISP_THRESHOLD:
                    self._box_valid_tumble = False
                    self._failed_reason = FAIL_REASON_EDGE_MOVED
                    return
            else:
                self._last_edge_pt1 = edge_pt1
                self._last_edge_pt2 = edge_pt2
        else:
            self._last_edge_pt1 = None
            self._last_edge_pt2 = None

        # check succeeded
        if tumble_x ^ tumble_y and \
                (theta_x < np.pi/2 - FULL_TUMBLE_THRESHOLD or theta_y < np.pi/2 - FULL_TUMBLE_THRESHOLD):
            self._suceeded = True

    @property
    def parameters(self) -> dict:
        return self._parameters

    def get_trajectory_status(self) -> dict:
        assert self._initialized, "task not initialized"

        status = {
            "succeeded": self._suceeded,
            "valid_tumble": self._box_valid_tumble,
            "tracking": self._tracking,
        }

        if not self._suceeded and not self._tracking:
            status['failed_reason'] = self._failed_reason

        return status


if __name__ == '__main__':
    env = DexEnv()
    task = DexTumbleTask()
    env.add_task(task)

    i = 0

    while not env.should_quit:
        env.step()
        print(task.get_trajectory_status())
