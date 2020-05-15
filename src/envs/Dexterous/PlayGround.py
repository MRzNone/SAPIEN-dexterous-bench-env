import sapien.core as sapien
from sapien.core import Pose
import numpy as np

if __name__ == "__main__":
    sim = sapien.Engine()

    renderer = sapien.OptifuserRenderer()
    sim.set_renderer(renderer)
    render_controller = sapien.OptifuserController(renderer)
    render_controller.set_camera_position(2, 0, 0.7)
    render_controller.set_camera_rotation(np.pi, -0.7)

    scene = sim.create_scene()
    scene.add_ground(-1)
    scene.set_timestep(1/500)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
    render_controller.set_current_scene(scene)

    def build_box():
        box_size = [0.02] * 3
        box_color = [0.2, 0.2, 0.6]
        builder = scene.create_actor_builder()
        builder.add_box_visual(size=box_size, color=box_color)
        builder.add_box_shape(size=box_size)
        return builder.build(name="play_box")

    # place box
    box = build_box()
    box.set_pose(Pose([0.5, -0.3, -0.9]))

    render_controller.show_window()
    while not render_controller.should_quit:
        scene.step()
        scene.update_render()
        render_controller.render()