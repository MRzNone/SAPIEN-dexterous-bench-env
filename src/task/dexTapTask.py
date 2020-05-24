import numpy as np
from sapien.core import Pose
import sapien.core as sapien
from transforms3d.quaternions import quat2mat

from agent import Box, PandaArm
from env.dexEnv import BOX_NAME, DexEnv, ARM_NAME
from sapien_interfaces import Task, Env

TASK_NAME = "Dexterous Box Tap"
MAX_TAP_SEC = 1

FAIL_REASON_TOUCHED_TOO_LONG = "contact too long"
FAIL_REASON_MULTI_CONTACT = "box was touched at multiple locations"
FAIL_REASON_CONTACT_MOVED = "contact point moved too much during contact"


def build_goal(scene, side_len) -> sapien.Actor:
    size = [side_len, side_len, 0.001]
    builder = scene.create_actor_builder()
    builder.add_box_visual(size=size, color=[0, 1, 0, 0.2])
    return builder.build(True, "Goal")


class DexTapTask(Task):

    def __init__(self):
        self._suceeded = False

        self._box: Box = None
        self._arm: PandaArm = None
        self._tracking = True

        self._last_contact = None
        self._contact_start_step = None
        self._failed_reason = None
        self._max_tap_steps = None

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

        self._suceeded = False
        self._tracking = True
        self._failed_reason = None
        self._last_contact = None
        self._contact_start_step = None
        self._max_tap_steps = MAX_TAP_SEC / env.timestep

        self._parameters = {
            "box_size": self._box.box_size,
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

        if self._suceeded is True or self._failed_reason is not None or not self._tracking:
            self._tracking = False
            return

        contacts = env.scene.get_contacts()

        # get valid contacts
        def if_valid_contact(c):
            s = {c.actor1.name, c.actor2.name}
            return 'ground' not in s and BOX_NAME in s and c.separation < 1e-4
        valid_contacts = np.array([c for c in contacts if if_valid_contact(c)])

        # group
        groups = {}
        for c in valid_contacts:
            other_actor = ({c.actor1.name, c.actor2.name} - {BOX_NAME}).pop()
            pos = c.point

            if other_actor not in groups:
                groups[other_actor] = [pos]
            else:
                groups[other_actor].append(pos)

        avg_contacts = {}
        for name, poses in groups.items():
            avg_contacts[name] = np.average(poses, axis=0)

        if len(avg_contacts) > 1:
            self._suceeded = False
            self._failed_reason = FAIL_REASON_MULTI_CONTACT
            return
        elif len(avg_contacts) == 1:
            contact_pos = list(avg_contacts.values())[0]
            contact_actor = list(avg_contacts.keys())[0]
            if self._last_contact is None:
                self._last_contact = (contact_actor, contact_pos)
                self._contact_start_step = env.current_step
            else:
                # check same contact actor and not moving a lot
                if contact_actor != self._last_contact[0]:
                    self._suceeded = False
                    self._failed_reason = FAIL_REASON_MULTI_CONTACT
                    return
                elif np.linalg.norm(contact_pos - self._last_contact[1]) > 1e-3:
                    self._suceeded = False
                    self._failed_reason = FAIL_REASON_CONTACT_MOVED
                    return

        # check success and over-time failure
        if self._contact_start_step is not None and len(avg_contacts) == 0:
            self._suceeded = True
            return
        elif self._contact_start_step is not None and env.current_step - self._contact_start_step > self._max_tap_steps:
            self._suceeded = False
            self._failed_reason = FAIL_REASON_TOUCHED_TOO_LONG
            return


    @property
    def parameters(self) -> dict:
        return self._parameters

    def get_trajectory_status(self) -> dict:
        assert self._initialized, "task not initialized"

        status = {
            "succeeded": self._suceeded,
            "touched": self._last_contact is not None,
            "tracking": self._tracking
        }

        if not self._suceeded and not self._tracking:
            status['failed_reason'] = self._failed_reason

        return status


if __name__ == '__main__':
    env = DexEnv()
    task = DexTapTask()
    env.add_task(task)

    i = 0

    while not env.should_quit:
        env.step()
        print(task.get_trajectory_status())
