import os

import sapien.core as sapien
from sapien.asset import download_partnet_mobility

from environment.agents import AgentBase

PART_NET_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Inc5emVuZ0B1Y3NkLmVkdSIsImlwIjoiNzAuOTUuMTc1LjIyNiIsInByaXZpbGVnZSI6MSwiaWF0IjoxNTg4NjM3ODQwLCJleHAiOjE2MjAxNzM4NDB9.hmgf19Rzbmrm8ZO-bzGLChsU7Vr3B5RMx6uIp3voFXA"
ASSET_DIR = os.path.abspath("../../Assets")


class EnvManager:
    def __init__(self, sim: sapien.Engine, timestep: float, visual: bool = True,
                 cam_pos=None, cam_rot=None, ground=True, *args):
        self.call_backs = []
        if cam_rot is None:
            cam_rot = [0, 0]
        if cam_pos is None:
            cam_pos = [0, 0, 0]
        self.agents = {}
        self.dumbs = {}
        self.id_counter = 0

        self.sim = sim

        if visual:
            self.renderer = sapien.OptifuserRenderer()
            self.sim.set_renderer(self.renderer)
            self.render_controller = sapien.OptifuserController(self.renderer)
            self.render_controller.set_camera_position(*cam_pos)
            self.render_controller.set_camera_rotation(*cam_rot)

        self.scene = self.sim.create_scene(*args)
        if ground:
            self.scene.add_ground(-1)
        self.scene.set_timestep(timestep)

        if visual:
            self.scene.set_ambient_light([0.5, 0.5, 0.5])
            self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
            self.render_controller.set_current_scene(self.scene)

        self.asset_dir = ASSET_DIR

    def get_urdf_loader(self):
        return self.scene.create_urdf_loader()

    def get_point_net_urdf(self, asset_id, scale=1, PxMaterial=None):
        loader = self.scene.create_urdf_loader()
        urdf = download_partnet_mobility(asset_id, PART_NET_TOKEN, os.path.join(ASSET_DIR, "part_net"))
        loader.fix_root_link = False
        loader.scale = scale
        asset = loader.load(urdf, PxMaterial)

        assert asset, f"Failed loading {asset_id}"

        return asset

    def register_agent(self, agent: AgentBase) -> int:
        agent_id = self.id_counter
        self.id_counter += 1
        self.agents[agent_id] = agent

        return agent_id

    def get_scene(self):
        return self.scene

    def show_window(self):
        self.render_controller.show_window()

    def add_callback(self, method):
        self.call_backs.append(method)

    def step(self, visual=True):
        self.scene.step()
        if visual:
            self.scene.update_render()
            self.render_controller.render()

        for cb in self.call_backs:
            cb()
