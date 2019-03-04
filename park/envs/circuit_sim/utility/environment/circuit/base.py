import abc

from park.envs.circuit_sim.circuit import Evaluator
from park.envs.circuit_sim.utility.environment import Env
from park.envs.circuit_sim.utility.learn import SpaceResolver
from park.envs.circuit_sim.utility.misc import nested_select

__all__ = ['CircuitEnv']


class CircuitEnv(Env, metaclass=abc.ABCMeta):
    def __init__(self, evaluator: Evaluator, benchmark, obs_space=True):
        if not callable(benchmark):
            raise ValueError("Benchmark is not a callable object.")

        self._evaluator = evaluator
        self._benchmark = benchmark
        self._obs_space, _ = nested_select(self._evaluator.out_space, obs_space)
        self._obs_resolver = SpaceResolver(self._obs_space)

    @property
    def obs_space(self):
        return self._obs_space

    def _reset_internal_state(self):
        self._running_scores = []
        self._running_params = []
        self._running_actions = []

    def reset(self):
        self._reset_internal_state()
        return self._obs_resolver.zeros()

    def _get_reward_from_scores(self):
        old_score = self._running_scores[-2] if len(self._running_scores) > 1 else 0
        now_score = self._running_scores[-1]
        return now_score - old_score

    def _get_obsinfo_from_features(self, features):
        if features is None:
            return self._obs_resolver.zeros(), None
        else:
            return nested_select(features, self._obs_space)

    def _execute(self, action, values):
        features = self._evaluator(values)
        obs, info = self._get_obsinfo_from_features(features)

        self._running_params.append(values)
        self._running_scores.append(self._benchmark(values, features))
        self._running_actions.append(action)

        reward = self._get_reward_from_scores()

        terminated = len(self._running_actions) == self.total_steps

        return obs, reward, terminated, info

    @property
    def scores(self):
        return self._running_scores.copy()

    @property
    def params(self):
        return self._running_params.copy()

    @property
    def actions(self):
        return self._running_actions.copy()

    @property
    def last_score(self):
        return self._running_scores[-1]

    @property
    def last_param(self):
        return self._running_params[-1]

    def __str__(self):
        return f'{self.__class__.__name__}(total_steps={self.total_steps})'

    __repr__ = __str__
