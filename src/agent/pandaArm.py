import os
import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from sapien_interfaces import Agent, Env
from Tools import config

INI_POSE = Pose(np.array([0., 0., -0.95]))
INI_QPOS = np.array([0, 0, 0, -1.5, 0, 1.5, 0.7, 0.4, 0.4])


class PandaArm(Agent):
    def __init__(self, env: Env, name: str, damping: int = 50, pusher=False):
        super().__init__()

        self._damping = damping
        self.name = name

        self.dof = None

        self._robot: sapien.Articulation = None
        self._initialized = False

        self._action_spec = None
        self._observation_spec = None
        self._active_joints = None
        self._pusher = pusher

    def init(self, env: Env):
        if self._initialized:
            print(f"{self.name} already initialized")
            return

        # load arm
        loader = env.scene.create_urdf_loader()
        loader.fix_root_link = True
        arm_path = os.path.join(config.ASSET_DIR, "Arm/panda.urdf")
        print()
        if not self._pusher:
            self._robot = loader.load(arm_path)
        else:
            builder = loader.load_file_as_articulation_builder(arm_path)
            lb = builder.get_link_builders()
            lb[-1].remove_all_shapes()
            lb[-1].remove_all_visuals()
            lb[-2].remove_all_shapes()
            lb[-2].remove_all_visuals()
            lb[-3].remove_all_shapes()
            lb[-3].remove_all_visuals()

            pusher_size = [0.06, 0.06, 0.03]
            pusher_pose = Pose([0,0,0.03])

            material = env._sim.create_physical_material(0.9, 0.9, 0)

            lb[-3].add_box_shape(pusher_pose, pusher_size, material=material)
            lb[-3].add_box_visual_complex(pusher_pose, pusher_size)

            self._robot = builder.build(True)

        self.dof = self._robot.dof

        # adjust damping
        for joint, d in zip(self._robot.get_joints(), np.array([self._damping] * self._robot.dof)):
            joint.set_drive_property(stiffness=0, damping=d)

        self._robot.set_pose(INI_POSE)
        self._robot.set_qpos(INI_QPOS)
        self._robot.set_name(self.name)

        self._active_joints = [j for j in self._robot.get_joints() if j.get_dof() == 1]

        qpos = np.array([0, 0, 0, -1.5, 0, 1.5, 0.7, 0.4, 0.4])
        self._robot.set_qpos(qpos)
        self._robot.set_qvel([0] * self._robot.dof)
        self.set_action(qpos, [0] * self._robot.dof, self._robot.compute_passive_force())

        # set specs
        self._action_spec = {
            "qlimits": self._robot.get_qlimits()
        }

        self._observation_spec = None

        self._initialized = True
        print(f"CREATED: {self.name} created")
        
    def get_compute_functions(self):
        """
        provides various convenience functions
        """
        return {
            'forward_dynamics': self._robot.compute_forward_dynamics,
            'inverse_dynamics': self._robot.compute_inverse_dynamics,
            'adjoint_matrix': self._robot.compute_adjoint_matrix,
            'spatial_twist_jacobian': self._robot.compute_spatial_twist_jacobian,
            'world_cartesian_jacobian': self._robot.compute_world_cartesian_jacobian,
            'manipulator_inertia_matrix': self._robot.compute_manipulator_inertia_matrix,
            'transformation_matrix': self._robot.compute_transformation_matrix,
            'passive_force': self._robot.compute_passive_force,
            'twist_diff_ik': self._robot.compute_twist_diff_ik,
            'cartesian_diff_ik': self._robot.compute_cartesian_diff_ik
        }

    def reset(self, env: Env) -> None:
        self._robot.set_pose(INI_POSE)
        self._robot.set_qpos(INI_QPOS)

    @property
    def action_spec(self) -> dict:
        return self._action_spec

    @property
    def observation_spec(self) -> dict:
        return self._observation_spec

    @property
    def observation(self) -> dict:
        """
        Return qpos, qvel and link poses
        """
        return {
            "qpos": self._robot.get_qpos(),
            "qvel": self._robot.get_qvel(),
            "poses": [link.get_pose() for link in self._robot.get_links()]
        }

    def configure_controllers(self, ps, ds):
        """
        Set parameters for the PD controllers for the robot joints
        """
        assert len(ps) == self.dof
        assert len(ds) == self.dof
        for j, p, d in zip(self._active_joints, ps, ds):
            j.set_drive_property(p, d)

    def set_action(self, drive_target, drive_velocity, additional_force):
        """
        action includes 3 parts
        drive_target: qpos target for PD controller
        drive_velocity: qvel target for PD controller
        additional_force: additional qf applied to the joints
        """
        assert len(drive_target) == self.dof
        assert len(drive_velocity) == self.dof
        assert len(additional_force) == self.dof
        for j, t, v in zip(self._active_joints, drive_target, drive_velocity):
            j.set_drive_target(t)
            j.set_drive_velocity_target(v)
        self._robot.set_qf(additional_force)

    def step(self, env, step: int):
        pass

    def close(self, env) -> None:
        env.scene.remove_articulation(self._robot)
