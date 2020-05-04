import sapien.core as sapien
from sapien.core import Pose
import numpy as np

from environment import EnvManager
from environment.agents import StickyAnt


class DragStone:
    def __init__(self):
        self.sim = sapien.Engine()
        self.env_man = EnvManager(self.sim, 1/60)

        color_1 = np.random.random(4)
        color_1[-1] = np.clip(color_1[-1], 0.8, 1)
        color_2 = np.random.random(4)
        color_2[-1] = np.clip(color_2[-1], 0.8, 1)

        self.ant1 = StickyAnt(self.env_man, damping=5, pose=Pose([1, 1, 0]), color=color_1)
        # self.ant2 = StickyAnt(self.env_man, damping=5, pose=Pose([-1, 1, 0]), color=color_2)

    def show_window(self):
        self.env_man.show_window()

    def step(self):
        self.env_man.step()

    def should_quit(self):
        return self.env_man.render_controller.should_quit


if __name__ == "__main__":
    env = DragStone()
    env.show_window()

    ant = env.ant1

    ant.robot.set_pose(Pose([0, 0, 0.5]))
    env.step()

    builder = env.env_man.scene.create_actor_builder()
    builder.add_sphere_visual(radius=0.3)
    builder.add_sphere_shape(radius=0.3)
    ball = builder.build()
    ball.set_pose(Pose([0, -1, -0.5]))

    env.step()

    for _ in range(30):
        u = np.array([0] * 12)
        u[-1] = 1
        ant.set_action(u)
        env.step()

    i = 0

    while not env.should_quit():
        u = np.concatenate((np.random.random(8) * 4000 - 2000, [0,0,0,0]))
        i += 1
        if i == 300:
            u[-4:] = -1
            print("unstick")
        ant.set_action(u)
        env.step()

