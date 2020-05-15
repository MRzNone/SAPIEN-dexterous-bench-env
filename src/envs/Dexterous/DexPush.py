import sapien.core as sapien
from sapien.core import Pose
import numpy as np

from environment import EnvManager
from environment.agents import PandaArm
from transforms3d.quaternions import quat2mat


class DexPush:
    def __init__(self, cam_pos=None, cam_rot=None,
                 time_step: float = 1 / 100):
        if cam_rot is None:
            cam_rot = [np.pi, -0.7]
        if cam_pos is None:
            cam_pos = [2, 0, 0.7]
        self.sim = sapien.Engine()
        self.env_man = EnvManager(self.sim, time_step, cam_pos=cam_pos, cam_rot=cam_rot)
        self.box_flipped = False
        self.TILT_THRESHHOLD = np.pi / 180 * 20
        self.GOAl_POS = np.array([0.5, 0.3, -1])

        # create arm
        self.arm = PandaArm(self.env_man)

        # place box
        self.box = self.build_box()
        self.box.set_pose(Pose([0.5, -0.3, -0.9]))

        # create goal
        self.goal = self.build_goal()
        self.goal.set_pose(Pose(self.GOAl_POS))

        self.env_man.add_callback(self.check_not_flip)

    def check_not_flip(self):
        if self.box_flipped is True:
            return

        box_pose = self.box.get_pose()
        # calcilate how much "flipped"
        up_right_pose = np.array([0, 0, 1])
        rot_mat = quat2mat(box_pose.q)
        tilted = rot_mat[:, 2]

        theta = np.arccos(tilted @ up_right_pose)

        if theta > self.TILT_THRESHHOLD:
            self.box_flipped = True

    def reward(self):
        if self.box_flipped:
            return 0

        box_pos = self.box.get_pose().p
        dist = np.linalg.norm(self.GOAl_POS[:2] - box_pos[:2])

        if dist < 5e-3:
            return 1

        return 0

    def build_box(self):
        box_size = [0.02] * 3
        box_color = [0.2, 0.2, 0.6]
        builder = self.env_man.scene.create_actor_builder()
        builder.add_box_visual(size=box_size, color=box_color)
        builder.add_box_shape(size=box_size)
        return builder.build(name="play_box")

    def build_goal(self):
        size = [0.02, 0.02, 0.001]
        builder = self.env_man.scene.create_actor_builder()
        builder.add_box_visual(size=size, color=[0, 1, 0, 0.2])
        return builder.build(True, "Goal")

    def show_window(self):
        self.env_man.show_window()

    def step(self):
        self.env_man.step()

    def should_quit(self):
        return self.env_man.render_controller.should_quit


if __name__ == "__main__":
    env = DexPush()

    env.show_window()

    while not env.should_quit():
        env.step()
