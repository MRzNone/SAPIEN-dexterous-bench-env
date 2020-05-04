import numpy as np
import sapien.core as sapien


class AgentBase:
    def get_observation(self) -> np.ndarray:
        raise NotImplementedError

    def set_action(self, u: np.ndarray) -> None:
        raise NotImplementedError

    def get_robot(self) -> sapien.Articulation:
        raise NotImplementedError