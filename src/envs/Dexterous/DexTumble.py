import sapien.core as sapien
from sapien.core import Pose
import numpy as np

from Tools import misc
from environment import EnvManager
from environment.agents import PandaArm
from transforms3d.quaternions import quat2mat

from envs import EnvBase


class DexTumble(EnvBase):
    def __init__(self, cam_pos=None, cam_rot=None,
                 time_step: float = 1 / 300):
        if cam_rot is None:
            cam_rot = [np.pi, -0.7]
        if cam_pos is None:
            cam_pos = [2, 0, 0.7]
        self.sim = sapien.Engine()
        self.env_man = EnvManager(self.sim, time_step, cam_pos=cam_pos, cam_rot=cam_rot)
        self.box_tumble_transform = True
        self.TILT_THRESHHOLD = np.pi / 180 * 75
        self.GOAl_POS = np.array([0.5, 0.3, -1])

        # create arm
        self.arm = PandaArm(self.env_man)

        # place box
        self.box = self.build_box()
        self.box.set_pose(Pose([0.5, -0.3, -0.99]))

        self.env_man.add_callback(self.check_tumble)

        self.ini_pack = self.save()

    def reset(self) -> []:
        self.load(self.ini_pack)
        self.box_tumble_transform = True

        return self.step()

    def load(self, data: []) -> None:
        box_data = data[0]
        arm_data = data[1]

        misc.load_actor(self.box, box_data)
        self.arm.load(arm_data)

    def save(self) -> []:
        box_data = misc.save_actor(self.box)
        arm_data = self.arm.save()

        return [box_data, arm_data]

    def check_tumble(self):
        if self.box_tumble_transform is False:
            return

        box_pose = self.box.get_pose()
        # calcilate how much "flipped"
        rot_mat = quat2mat(box_pose.q)
        tilted = rot_mat[:3, 2]
        tilted /= np.linalg.norm(tilted)

        theta_x = np.arccos(tilted[0])
        theta_y = np.arccos(tilted[1])

        tumble_x = theta_x < self.TILT_THRESHHOLD
        tumble_y = theta_y < self.TILT_THRESHHOLD

        if tumble_x and tumble_y:
            print("Failed to tumble")
            self.box_tumble_transform = False

    def reward(self):
        if not self.box_tumble_transform:
            return np.array([0])

        box_pose = self.box.get_pose()
        rot_mat = quat2mat(box_pose.q)
        tilted = rot_mat[:3, 2]
        tilted /= np.linalg.norm(tilted)

        theta = np.arccos(tilted[2])

        if theta > np.pi * 0.49:
            return np.array([1])

        return np.array([0])

    def build_box(self):
        box_size = [0.02] * 3
        box_color = [0.2, 0.2, 0.6]
        builder = self.env_man.scene.create_actor_builder()
        builder.add_box_visual(size=box_size, color=box_color)
        builder.add_box_shape(size=box_size)
        return builder.build(name="play_box")

    def get_metadata(self) -> dict:
        return None

    def show_window(self):
        self.env_man.show_window()

    def get_obs(self):
        box_obs = [self.box.get_pose().p.tolist(), self.box.get_pose().q.tolist()]
        arm_obs = self.arm.get_observation().tolist()
        return [arm_obs, box_obs]

    def step(self):
        self.env_man.step()

        return self.get_obs()

    def should_quit(self):
        return self.env_man.render_controller.should_quit


if __name__ == "__main__":
    env = DexTumble()

    env.show_window()

    env.step()

    while not env.should_quit():
        env.step()
