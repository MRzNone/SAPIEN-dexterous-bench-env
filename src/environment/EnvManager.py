import sapien.core as sapien

from environment.agents import AgentBase


class EnvManager:
    def __init__(self, sim: sapien.Engine, timestep: float, visual: bool = True, *args):
        self.agents = []
        self.dumbs = []

        self.sim = sim

        if visual:
            self.renderer = sapien.OptifuserRenderer()
            self.sim.set_renderer(self.renderer)
            self.render_controller = sapien.OptifuserController(self.renderer)
            self.render_controller.set_camera_position(-5, 0, 0)

        self.scene = self.sim.create_scene(*args)
        self.scene.add_ground(-1)
        self.scene.set_timestep(timestep)

        if visual:
            self.scene.set_ambient_light([0.5, 0.5, 0.5])
            self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
            self.render_controller.set_current_scene(self.scene)

    def register_agent(self, agent: AgentBase) -> None:
        self.agents.append(agent)

    def get_scene(self) -> sapien.Scene:
        return self.scene

    def show_window(self):
        self.render_controller.show_window()

    def step(self, visual=True):
        self.scene.step()

        if visual:
            self.scene.update_render()
            self.render_controller.render()