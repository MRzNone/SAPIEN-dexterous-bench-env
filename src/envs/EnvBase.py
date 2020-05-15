import numpy as np


class EnvBase:
    def step(self) -> []:
        raise NotImplementedError

    def reset(self) -> []:
        raise NotImplementedError

    def save(self) -> []:
        raise NotImplementedError

    def load(self, data: []) -> None:
        raise NotImplementedError

    def reward(self) -> np.ndarray:
        raise NotImplementedError

    def get_metadata(self) -> dict:
        raise NotImplementedError
