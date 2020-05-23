import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from sapien_interfaces import Agent, Env


class Box(Agent):
    def __init__(self, name: str, pose: Pose, size: int = 0.02):
        self._ini_pose = pose
        self._size = np.array([size] * 3)

        self._initialized = False
        self._box: sapien.Actor = None
        self.name = name

    def init(self, env: Env):
        if self._initialized:
            print(f"{self.name} already initialized")
            return

        builder = env.scene.create_actor_builder()
        builder.add_box_shape(size=self._size)
        builder.add_box_visual(size=self._size, color=np.array([0.2, 0.4, 0.6]))
        self._box = builder.build()
        self._box.set_name(self.name)

        self._box.set_pose(self._ini_pose)

        self._initialized = True
        print(f"CREATED: {self.name} created")

    def reset(self, env) -> None:
        self._box.set_pose(self._ini_pose)

    @property
    def box_size(self) -> float:
        return self._size[0]

    @property
    def action_spec(self) -> dict:
        return None

    @property
    def observation_spec(self) -> dict:
        return None

    @property
    def observation(self) -> dict:
        return {
            "pose": self._box.get_pose()
        }

    def set_action(self, action):
        return None

    def step(self, env, step: int):
        return None