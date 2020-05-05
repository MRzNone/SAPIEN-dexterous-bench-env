import sapien.core as sapien
from sapien.core import Pose
import numpy as np

from environment import EnvManager
from environment.agents import StickyAnt


class DragTable:
    def __init__(self, ants_config: [(np.ndarray, Pose)] = None,
                 cam_pos=None, cam_rot=None,
                 time_step: float = 1 / 100):
        if cam_rot is None:
            cam_rot = [0, -0.3]
        if cam_pos is None:
            cam_pos = [-10, 0, 5]
        if ants_config is None:
            ants_config = [
                ([1, 0, 0, 1], Pose([0, 3, 0])),
                ([0, 0, 1, 1], Pose([0, -3, 0])),
            ]
        self.sim = sapien.Engine()
        self.env_man = EnvManager(self.sim, time_step, cam_pos=cam_pos, cam_rot=cam_rot)

        # create table
        self.table = self.env_man.get_point_net_urdf(24152, 1.8)

        # create ants
        self.ants = []
        for color, pose in ants_config:
            self.ants.append(StickyAnt(self.env_man, damping=5, pose=pose, color=color))

        # create goal
        self.goal = self.build_goal()
        self.goal.set_pose(Pose([7, 0, -1]))

    def build_goal(self):
        size = [1.5, 0.8, 0.01]
        builder = self.env_man.scene.create_actor_builder()
        builder.add_box_visual(size=size, color=[0, 1, 0, 0.2])
        return builder.build(True, "Goal")

    def get_ants(self) -> [StickyAnt]:
        return self.ants

    def show_window(self):
        self.env_man.show_window()

    def step(self):
        self.env_man.step()

    def should_quit(self):
        return self.env_man.render_controller.should_quit


if __name__ == "__main__":
    env = DragTable()

    env.show_window()

    while not env.should_quit():
        env.step()
