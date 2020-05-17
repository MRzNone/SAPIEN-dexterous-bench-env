import sapien.core as sapien
from sapien.core import Pose
import numpy as np
from transforms3d.quaternions import quat2mat

from Tools import misc
from environment import EnvManager
from environment.agents import PandaArm

from envs import EnvBase


class DexPush(EnvBase):
    def __init__(self, cam_pos=None, cam_rot=None,
                 time_step: float = 1 / 300):
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
        self.box.set_pose(self.random_pose())

        # create goal
        self.goal = self.build_goal()
        self.goal.set_pose(Pose(self.GOAl_POS))

        self.env_man.add_callback(self.check_not_flip)

        self.ini_pack = self.save()

    def random_pose(self, theta_range=[0, 2 * np.pi], len=[0.2, 0.6]):
        theta = np.random.uniform(low=theta_range[0], high=theta_range[1])
        r = np.random.uniform(low=len[0], high=len[1])

        print(theta / np.pi * 180, r)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return Pose([x, y, -0.99])

    def reset(self) -> []:
        self.load(self.ini_pack)
        self.box_flipped = False

        return self.step()

    def save(self) -> []:
        box_data = misc.save_actor(self.box)
        arm_data = self.arm.save()

        return [box_data, arm_data]

    def load(self, data: []) -> None:
        box_data = data[0]
        arm_data = data[1]

        misc.load_actor(self.box, box_data)
        self.arm.load(arm_data)

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
            return np.array([0])

        box_pos = self.box.get_pose().p
        dist = np.linalg.norm(self.GOAl_POS[:2] - box_pos[:2])

        if dist < 5e-3:
            return np.array([1])

        return np.array([0])

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

    def get_obs(self):
        box_obs = [self.box.get_pose().p.tolist(), self.box.get_pose().q.tolist()]
        arm_obs = self.arm.get_observation().tolist()
        return [arm_obs, box_obs]

    def step(self):
        self.env_man.step()

        return self.get_obs()

    def get_metadata(self) -> dict:
        None

    def should_quit(self):
        return self.env_man.render_controller.should_quit


if __name__ == "__main__":
    env = DexPush()

    env.show_window()

    while not env.should_quit():
        env.step()
