import os

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from environment import EnvManager
from Tools import misc
from environment.agents import AgentBase


class PandaArm(AgentBase):
    def __init__(self, env_man: EnvManager, pose: Pose = None, damping=None):
        if pose is None:
            pose = Pose([0, 0, -0.95])

        if damping is None:
            damping = 20

        self.env_man = env_man
        # load arm
        loader = self.env_man.get_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load("../../Assets/Arm/panda.urdf")
        self.robot.set_pose(pose)
        self.robot.set_qpos(np.array([0, 0, 0, -1.5, 0, 1.5, 0.7, 0.4, 0.4]))

        # adjust damping
        if damping is not None:
            if isinstance(damping, int):
                damping = np.array([damping] * self.robot.dof)

            assert damping.shape == (self.robot.dof,)

            for joint, d in zip(self.robot.get_joints(), damping):
                joint.set_drive_property(stiffness=0, damping=d)

        self.mId = self.env_man.register_agent(self)
        self.robot.set_name(f"PandaArm_{self.mId}")

        self.mount_camera()

        print(f"Created: PandaArm {self.mId} created")

    def mount_camera(self):
        pass

    def get_observation(self) -> np.ndarray:
        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel()
        root_pose = self.robot.get_root_pose()

        obs = np.concatenate((qpos, qvel, root_pose.p, root_pose.q))

        return obs

    def set_action(self, u: np.ndarray) -> None:
        self.robot.set_qf(u)

    def get_robot(self) -> sapien.Articulation:
        return self.robot
