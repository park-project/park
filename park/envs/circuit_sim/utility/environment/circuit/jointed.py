import abc

import numpy as np

from park.envs.circuit_sim.circuit import Evaluator
from park.envs.circuit_sim.utility.environment.circuit import CircuitEnv
from park.envs.circuit_sim.utility.learn import Box

__all__ = ['CircuitJointedEnv', 'CircuitPointedEnv', 'CircuitSteppedEnv']


class CircuitJointedEnv(CircuitEnv, metaclass=abc.ABCMeta):
    def __init__(self, evaluator: Evaluator, benchmark, total_steps, obs_space=True):
        super().__init__(evaluator, benchmark, obs_space)
        self._total_steps = total_steps

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def act_space(self) -> Box:
        return Box.from_bound(self._evaluator.bound)


class CircuitSteppedEnv(CircuitJointedEnv):
    def _reset_internal_state(self):
        super(CircuitSteppedEnv, self)._reset_internal_state()
        self._running_values = self.act_space.zeros()

    def step(self, action):
        self._running_values += np.asarray(action)
        return self._execute(action, self._running_values)


class CircuitPointedEnv(CircuitJointedEnv):
    def step(self, action):
        return self._execute(action, np.asarray(action))
