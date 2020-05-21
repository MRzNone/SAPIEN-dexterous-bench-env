import os
import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from sapien_interfaces import Agent, Env
from Tools import config

INI_POSE = Pose(np.array([0., 0., -0.95]))
INI_QPOS = np.array([0, 0, 0, -1.5, 0, 1.5, 0.7, 0.4, 0.4])


class PandaArm(Agent):
    def __init__(self, env: Env, name: str, damping: int = 50):
        super().__init__()

        self._damping = damping
        self.name = name

        self._robot: sapien.Articulation = None
        self._initialized = False

        self._action_spec = None
        self._observation_spec = None

    def init(self, env: Env):
        if self._initialized:
            print(f"{self.name} already initialized")
            return

        # load arm
        loader = env.scene.create_urdf_loader()
        loader.fix_root_link = True
        arm_path = os.path.join(config.ASSET_DIR, "Arm/panda.urdf")
        self._robot = loader.load(arm_path)
        print()

        # adjust damping
        for joint, d in zip(self._robot.get_joints(), np.array([self._damping] * self._robot.dof)):
            joint.set_drive_property(stiffness=0, damping=d)

        self._robot.set_pose(INI_POSE)
        self._robot.set_qpos(INI_QPOS)
        self._robot.set_name(self.name)

        # set specs
        self._action_spec = {
            "qlimits": self._robot.get_qlimits()
        }

        self._observation_spec = None

        self._initialized = True
        print(f"CREATED: {self.name} created")

    def reset(self, env: Env) -> None:
        self._robot.set_pose(INI_POSE)
        self._robot.set_qpos(INI_QPOS)

    @property
    def action_spec(self) -> dict:
        return self._action_spec

    @property
    def observation_spec(self) -> dict:
        return self._observation_spec

    @property
    def observation(self) -> dict:
        """
        Return qpos, qvel and link poses
        """

        poses = map(lambda link: link.get_pose() , self._robot.get_links())

        return {
            "qpos": self._robot.get_qpos(),
            "qvel": self._robot.get_qvel(),
            "link_poses": poses
        }

    def set_action(self, action):
        """
        Set the joint torque

        :param action: the joint torque
        """
        self._robot.set_qf(action)

    def step(self, env, step: int):
        pass
