from typing import List, Dict
import sapien.core as sapien

from .task import Task
from .agent import Agent


class Env:
    @property
    def current_step(self) -> int:
        """Current step in an episode."""
        raise NotImplementedError()

    @property
    def timestep(self) -> float:
        """Simulation frequency."""
        raise NotImplementedError()

    @property
    def n_substeps(self) -> int:
        """Number of physical steps per control step, also known as frame skip."""
        raise NotImplementedError()

    @property
    def tasks(self) -> List[Task]:
        """Get all tasks in this environment"""
        raise NotImplementedError()

    def add_task(self, task: Task):
        """Add a task to the environment"""
        raise NotImplementedError()

    @property
    def agents(self) -> Dict[str, Agent]:
        raise NotImplementedError()

    @property
    def scene(self) -> sapien.Scene:
        raise NotImplementedError

    def step(self) -> None:
        """This function should respect the n_substeps property and also call proper
        task life cycle functions.

        The following is a template for this function

        for t in self.tasks:
            t.before_step(self)
        # do stuff
        for _ in self.n_substeps:
            for t in self.tasks:
                t.before_step(self)
            # do stuff
            # do physical step
            # increase current_step
            # do stuff
            for t in self.tasks:
                t.after_step(self)
        # do stuff
        for t in self.tasks:
            t.after_step(self)
        """
        raise NotImplementedError()

    @property
    def metadata(self) -> dict:
        """Useful data to describe the environment"""
        return dict()

    @property
    def description(self) -> str:
        """Human readable description of the environment"""
        return ''


def private(func):
    def wrapper(self, *args, **kwargs):
        print('nope')
        return func(*args, **kwargs)

    return wrapper
