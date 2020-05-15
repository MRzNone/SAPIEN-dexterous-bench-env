import sapien.core as sapien
from sapien.core import Pose
import numpy as np

from environment import EnvManager
from environment.agents import StickyAnt


class CrossCliff:
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
        self.env_man = EnvManager(self.sim, time_step, cam_pos=cam_pos, cam_rot=cam_rot, ground=False)

        self.build_cliffs()

        self.ladder = self.build_ladder_for_crossing()
        self.ladder.set_pose(Pose([-4, 0, 0]))

        # create ants
        self.ants = []
        for color, pose in ants_config:
            self.ants.append(StickyAnt(self.env_man, damping=5, pose=pose, color=color))

    def build_ladder_for_crossing(self) -> sapien.Actor:
        builder = self.env_man.scene.create_actor_builder()
        ladder_size = [3, 1.5, 0.05]
        ladder_color = [0.5, 0.5, 0.5, 0.8]

        builder.add_box_visual(size=ladder_size, color=ladder_color)
        builder.add_box_shape(size=ladder_size, density=500)
        ladder = builder.build()

        return ladder

    def build_cliffs(self) -> sapien.Actor:
        saperation = 4

        # cliff 1
        builder = self.env_man.scene.create_actor_builder()
        size1 = np.array([5.0, 5.0, 5.0])
        pose1 = Pose([-5 - saperation/2, 0, -5])
        builder.add_box_visual(pose=pose1, size=size1)
        builder.add_box_shape(pose=pose1, size=size1)
        size2 = [5, 5, 5]
        pose2 = Pose([5 + saperation/2, 0, -5])
        builder.add_box_visual(pose=pose2, size=size2)
        builder.add_box_shape(pose=pose2, size=size2)

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
    env = CrossCliff()

    env.show_window()

    while not env.should_quit():
        env.step()
