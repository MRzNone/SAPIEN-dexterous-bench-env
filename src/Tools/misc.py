import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.quaternions import axangle2quat as aa

ant_poses = {
    'j1': (
        Pose([0.282, 0, 0], [0.7071068, 0, 0.7071068, 0]),
        Pose([0.141, 0, 0], [-0.7071068, 0, 0.7071068, 0])),
    'j2': (
        Pose([-0.282, 0, 0], [0, -0.7071068, 0, 0.7071068]),
        Pose([0.141, 0, 0], [-0.7071068, 0, 0.7071068, 0])),
    'j3': (
        Pose([0, 0.282, 0], [0.5, -0.5, 0.5, 0.5]),
        Pose([0.141, 0, 0], [0.7071068, 0, -0.7071068, 0])),
    'j4': (
        Pose([0, -0.282, 0], [0.5, 0.5, 0.5, -0.5]),
        Pose([0.141, 0, 0], [0.7071068, 0, -0.7071068, 0])),
    'j11': (
        Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
        Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0])),
    'j21': (
        Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
        Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0])),
    'j31': (
        Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
        Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0])),
    'j41': (
        Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
        Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0])),
}


def create_ant_builder(scene: sapien.Scene, sim: sapien.Engine, rgba=None) -> sapien.pysapien.ArticulationBuilder:
    if rgba is None:
        rgba = [0.875, 0.553, 0.221, 1]
    copper = sapien.PxrMaterial()
    copper.set_base_color(rgba)
    copper.metallic = 1
    copper.roughness = 0.2
    density = 2000

    ant_material = sim.create_physical_material(0.9, 0.8, 1.0)

    builder = scene.create_articulation_builder()
    body = builder.create_link_builder()
    body.add_sphere_shape(Pose(), 0.25, material=ant_material, density=density)
    body.add_sphere_visual_complex(Pose(), 0.25, copper)
    body.add_capsule_shape(Pose([0.141, 0, 0]), 0.08, 0.141, material=ant_material, density=density)
    body.add_capsule_visual_complex(Pose([0.141, 0, 0]), 0.08, 0.141, copper)
    body.add_capsule_shape(Pose([-0.141, 0, 0]), 0.08, 0.141, material=ant_material, density=density)
    body.add_capsule_visual_complex(Pose([-0.141, 0, 0]), 0.08, 0.141, copper)
    body.add_capsule_shape(Pose([0, 0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141, material=ant_material,
                           density=density)
    body.add_capsule_visual_complex(Pose([0, 0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141, copper)
    body.add_capsule_shape(Pose([0, -0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141, material=ant_material,
                           density=density)
    body.add_capsule_visual_complex(Pose([0, -0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141, copper)
    body.set_name("body")

    l1 = builder.create_link_builder(body)
    l1.set_name("l1")
    l1.set_joint_name("j1")
    l1.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                            ant_poses['j1'][0], ant_poses['j1'][1], 0.1)
    l1.add_capsule_shape(Pose(), 0.08, 0.141, material=ant_material, density=density)
    l1.add_capsule_visual_complex(Pose(), 0.08, 0.141, copper)

    l2 = builder.create_link_builder(body)
    l2.set_name("l2")
    l2.set_joint_name("j2")
    l2.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                            ant_poses['j2'][0], ant_poses['j2'][1], 0.1)
    l2.add_capsule_shape(Pose(), 0.08, 0.141, material=ant_material)
    l2.add_capsule_visual_complex(Pose(), 0.08, 0.141, copper)

    l3 = builder.create_link_builder(body)
    l3.set_name("l3")
    l3.set_joint_name("j3")
    l3.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                            ant_poses['j3'][0], ant_poses['j3'][1], 0.1)
    l3.add_capsule_shape(Pose(), 0.08, 0.141, material=ant_material, density=density)
    l3.add_capsule_visual_complex(Pose(), 0.08, 0.141, copper)

    l4 = builder.create_link_builder(body)
    l4.set_name("l4")
    l4.set_joint_name("j4")
    l4.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                            ant_poses['j4'][0], ant_poses['j4'][1], 0.1)
    l4.add_capsule_shape(Pose(), 0.08, 0.141, material=ant_material, density=density)
    l4.add_capsule_visual_complex(Pose(), 0.08, 0.141, copper)

    f1 = builder.create_link_builder(l1)
    f1.set_name("f1")
    f1.set_joint_name("j11")
    f1.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                            ant_poses['j11'][0], ant_poses['j11'][1], 0.1)
    f1.add_capsule_shape(Pose(), 0.08, 0.282, material=ant_material, density=density)
    f1.add_capsule_visual_complex(Pose(), 0.08, 0.282, copper)

    f2 = builder.create_link_builder(l2)
    f2.set_name("f2")
    f2.set_joint_name("j21")
    f2.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                            ant_poses['j21'][0], ant_poses['j21'][1], 0.1)
    f2.add_capsule_shape(Pose(), 0.08, 0.282, material=ant_material, density=density)
    f2.add_capsule_visual_complex(Pose(), 0.08, 0.282, copper)

    f3 = builder.create_link_builder(l3)
    f3.set_name("f3")
    f3.set_joint_name("j31")
    f3.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                            ant_poses['j31'][0], ant_poses['j31'][1], 0.1)
    f3.add_capsule_shape(Pose(), 0.08, 0.282, material=ant_material, density=density)
    f3.add_capsule_visual_complex(Pose(), 0.08, 0.282, copper)

    f4 = builder.create_link_builder(l4)
    f4.set_name("f4")
    f4.set_joint_name("j41")
    f4.set_joint_properties(sapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                            ant_poses['j41'][0], ant_poses['j41'][1], 0.1)
    f4.add_capsule_shape(Pose(), 0.08, 0.282, material=ant_material, density=density)
    f4.add_capsule_visual_complex(Pose(), 0.08, 0.282, copper)

    return builder


def save_actor(actor: sapien.Actor):
    p = actor.get_pose()
    v = actor.get_velocity()
    l = actor.get_angular_velocity()

    return [p, v, l]


def load_actor(actor: sapien.Actor, data: []):
    p, v, l = data

    actor.set_pose(p)
    actor.set_velocity(v)
    actor.set_angular_velocity(l)
