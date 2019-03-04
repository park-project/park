import abc

from park.envs.circuit_sim.utility.learn import ObjSpace

__all__ = ['Env']


class Env(object, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def obs_space(self) -> ObjSpace:
        pass

    @property
    @abc.abstractmethod
    def act_space(self) -> ObjSpace:
        pass

    @abc.abstractmethod
    def step(self, action):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    @property
    def total_steps(self):
        raise ValueError('This environment has not defined total_steps.')
