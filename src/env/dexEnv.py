import sapien.core as sapien
from sapien.core import Pose
from typing import Dict, List
import numpy as np

from agent import PandaArm, Box
from sapien_interfaces import Env, Task, Agent

ENV_DESCRIPTION = 'Environment for Dexterous tasks including pushing, tapping, sliding and tumbling'
ARM_NAME = "DexArm"
BOX_NAME = "DexBox"


def random_pose_circle(theta_range=None, len=None):
    if len is None:
        len = [0.3, 0.6]
    if theta_range is None:
        theta_range = [0, 2 * np.pi]

    theta = np.random.uniform(low=theta_range[0], high=theta_range[1])
    r = np.random.uniform(low=len[0], high=len[1])

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return Pose(np.array([x, y, -0.99]))


class DexEnv(Env):
    def __init__(self, timestep: float = 1 / 60, frame_skip: int = 1, visual: bool = True, cam_pos=None, cam_rot=None):
        if cam_rot is None:
            cam_rot = [-np.pi, -np.pi/9]
        if cam_pos is None:
            cam_pos = [2, 0, 0]

        # initialize sapien
        self._sim = sapien.Engine()

        if visual:
            self._renderer = sapien.OptifuserRenderer()
            self._sim.set_renderer(self._renderer)
            self._render_controller = sapien.OptifuserController(self._renderer)
            self._render_controller.set_camera_position(*cam_pos)
            self._render_controller.set_camera_rotation(*cam_rot)

        sceneConfig = sapien.SceneConfig()
        sceneConfig.sleep_threshold = 0.00002
        self._scene = self._sim.create_scene(config=sceneConfig)
        self._scene.add_ground(-1)
        self._scene.set_timestep(timestep)

        if visual:
            self._scene.set_ambient_light([0.5, 0.5, 0.5])
            self._scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
            self._render_controller.set_current_scene(self._scene)

        # set up variables
        self._timestep = timestep
        self._frame_skip = frame_skip
        self._step = 0
        self._tasks: [Task] = []
        self._agents: Dict[str, Agent] = {}
        self._shown_window = False

        # set up basic environment
        self._set_up_basic_env()

        for _ in range(100):
            self._scene.step()
        self._render_controller.render()

    def _set_up_basic_env(self):
        # arm
        arm = PandaArm(self, ARM_NAME)
        arm.init(self)
        self._agents[ARM_NAME] = arm

        # box
        box_pose = random_pose_circle()
        box = Box(BOX_NAME, box_pose)
        box.init(self)
        self._agents[BOX_NAME] = box

    @property
    def scene(self) -> sapien.Scene:
        return self._scene

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def timestep(self) -> float:
        return self._timestep

    @property
    def n_substeps(self) -> int:
        return self._frame_skip

    @property
    def tasks(self) -> List[Task]:
        return self._tasks

    def add_task(self, t: Task):
        self._tasks.append(t)
        t.init(self)

    @property
    def agents(self) -> Dict[str, Agent]:
        return self._agents

    def step(self):
        if self._shown_window is False:
            self._render_controller.show_window()

        # pre step
        for t in self._tasks:
            t.before_step(self)

        # substeps
        for _ in range(self.n_substeps):
            for t in self._tasks:
                t.before_substep(self)

            self._scene.step()
            self._step += 1

            for t in self._tasks:
                t.after_substep(self)

        self._scene.update_render()
        self._render_controller.render()

        # after step
        for t in self._tasks:
            t.after_step(self)

    @property
    def metadata(self) -> dict:
        """Useful data to describe the environment"""
        return dict()

    @property
    def description(self) -> str:
        return ENV_DESCRIPTION

    @property
    def should_quit(self) -> bool:
        return self._render_controller.should_quit


if __name__ == '__main__':
    env = DexEnv()

    while not env.should_quit:
        env.step()