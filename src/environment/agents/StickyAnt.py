import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from environment import EnvManager
from Tools import misc
from environment.agents import AgentBase


class StickyAnt(AgentBase):
    def __init__(self, env_man: EnvManager, pose: Pose = Pose(), color=None, damping=None):
        super().__init__()

        assert env_man is not None

        self.env_man = env_man
        ant_builder = misc.create_ant_builder(self.env_man.get_scene(), env_man.sim, color)
        self.robot: sapien.Articulation = ant_builder.build()
        self.robot.set_pose(pose)

        # adjust damping
        if damping is not None:
            if isinstance(damping, int):
                damping = np.array([damping] * self.robot.dof)

            assert damping.shape == (self.robot.dof,)

            for joint, d in zip(self.robot.get_joints(), damping):
                joint.set_drive_property(stiffness=0, damping=d)

        self.mId = self.env_man.register_agent(self)
        self.robot.set_name(f"Sticky_ant_{self.mId}")

        # get foot for sticky
        self.num_foot = 4
        self.foot = self.robot.get_links()[-self.num_foot:]
        self.sticks: [sapien.Drive] = [None] * self.num_foot
        self.touch_radius = 0.15

        self.camera = self.mount_camera()

        print(f"{self.mId}: Sticky Ant created")

    def mount_camera(self) -> sapien.ICamera:
        body = self.robot.get_links()[0]
        near, far = 0.05, 100
        camera = self.env_man.scene.add_mounted_camera(f"ant_{self.mId}_cam", body, Pose([0, 0, 3], [0, -0.7071068, 0, 0.7071068]),
                                                       1920, 1080, np.deg2rad(50), np.deg2rad(50),
                                                       near, far)

        return camera

    def get_observation(self) -> np.ndarray:
        """
        return qpos, qvel, root_pose concatnated

        :return: [qpos, qvel, root_pose] concacnated
        """
        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel()
        root_pose = self.robot.get_root_pose()

        obs = np.concatenate((qpos, qvel, root_pose.p, root_pose.q))

        return obs

    def stick(self, foot_index: int):
        if self.sticks[foot_index] is not None:
            return

        feet = self.foot[foot_index]

        contacts = self.env_man.scene.get_contacts()

        touches = np.array([c for c in contacts if feet in {c.actor1, c.actor2}])

        # no touch
        if len(touches) is 0:
            return

        # check if at end
        feet_pos = (feet.get_pose() * Pose([-0.282, 0, 0])).p
        touches_pos = np.array([c.point for c in touches])
        touch_dist = np.linalg.norm(touches_pos - feet_pos, axis=1)
        valid_sticks = touches[np.argwhere(touch_dist < self.touch_radius)]

        if len(valid_sticks) is 0:
            return
        valid_sticks = valid_sticks[0]

        # just using the first
        valid_stick = valid_sticks[0]
        other_actor = ({valid_stick.actor1, valid_stick.actor2} - {feet}).pop()
        pt = valid_stick.point

        # pose1 = Pose([-0.282, 0, 0])
        pose1 = feet.get_pose().inv() * Pose(pt)
        pose2 = other_actor.get_pose().inv() * Pose(pt)

        drive = self.env_man.scene.create_drive(feet, pose1, other_actor, pose2)
        drive.set_properties(1000, 0, is_acceleration=False)

        self.sticks[foot_index] = drive
        print(f"foot {foot_index+1} sticks")

    def unstick(self, foot_index: int) -> None:
        if self.sticks[foot_index] is not None:
            self.sticks[foot_index].destroy()
        self.sticks[foot_index] = None

    def set_action(self, action: np.ndarray) -> None:
        """
        set force on joints and if stick (8+4)
        sticks:
            1:          stick
            -1:         unstick
            otherwise:  unchange

        :param action: the generalised force and 4 int representing if stick
        """
        assert action.shape == (self.robot.dof + self.num_foot, )

        u = action[:self.robot.dof]
        sticks = action[-self.num_foot:]
        for i in range(4):
            if sticks[i] == -1:
                self.unstick(i)
            elif sticks[i] == 1:
                self.stick(i)

        self.robot.set_qf(u)

    def get_robot(self) -> sapien.Articulation:
        return self.robot
