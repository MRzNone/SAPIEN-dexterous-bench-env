import sapien.core as sapien
from sapien.core import Pose
import numpy as np

from environment import EnvManager
from environment.agents import StickyAnt


class LadderCliff:
    def __init__(self, ants_config: [(np.ndarray, Pose)] = None,
                 cam_pos=None, cam_rot=None,
                 time_step: float = 1 / 100):
        if cam_rot is None:
            cam_rot = [0, -0.3]
        if cam_pos is None:
            cam_pos = [-15, 0, 5]
        if ants_config is None:
            ants_config = [
                ([1, 0, 0, 1], Pose([-5, 3, 1])),
                ([0, 0, 1, 1], Pose([-5, -3, 1])),
            ]
        self.sim = sapien.Engine()
        self.env_man = EnvManager(self.sim, time_step, cam_pos=cam_pos, cam_rot=cam_rot)

        self.build_cliffs()

        self.ladder, self.roller = self.build_tools_for_crossing()
        self.ladder.set_pose(Pose([0, 5, 0]))
        self.roller.set_pose(Pose([0, -5, 0]))

        # create ants
        self.ants = []
        for color, pose in ants_config:
            self.ants.append(StickyAnt(self.env_man, damping=5, pose=pose, color=color))

    def build_tools_for_crossing(self) -> [sapien.Actor]:
        builder = self.env_man.scene.create_actor_builder()
        ladder_size = [3, 1.5, 0.05]
        ladder_color = [0.5, 0.5, 0.5, 0.8]
        builder.add_box_visual(size=ladder_size, color=ladder_color)
        builder.add_box_shape(size=ladder_size, density=500)
        ladder = builder.build()

        builder = self.env_man.scene.create_actor_builder()
        roller_radius = 0.6
        roller_length = 1
        roller_color = [0.7, 0.3, 0.5, 0.8]
        builder.add_capsule_shape(radius=roller_radius, half_length=roller_length)
        builder.add_capsule_visual(radius=roller_radius, half_length=roller_length, color=roller_color)
        roller = builder.build()

        return ladder, roller

    def build_cliffs(self) -> sapien.Actor:
        # cliff 1
        builder = self.env_man.scene.create_actor_builder()
        size = np.array([2.5, 2.5, 1.0])
        builder.add_box_visual(size=size, color=[0.9, 0.9, 0.9, 0.9])
        builder.add_box_shape(size=size)

        return builder.build(True, "cliff")

    def get_ants(self) -> [StickyAnt]:
        return self.ants

    def show_window(self):
        self.env_man.show_window()

    def step(self):
        self.env_man.step()

    def should_quit(self):
        return self.env_man.render_controller.should_quit


if __name__ == "__main__":
    env = LadderCliff()

    env.show_window()

    while not env.should_quit():
        env.step()
