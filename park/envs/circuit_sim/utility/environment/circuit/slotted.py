import abc

from park.envs.circuit_sim.circuit import Evaluator
from park.envs.circuit_sim.utility.environment.circuit import CircuitEnv
from park.envs.circuit_sim.utility.learn import Box
from park.envs.circuit_sim.utility.misc import AttrDict
import numpy as np

__all__ = ['CircuitSlottedEnv', 'CircuitInitSlottedEnv', 'CircuitVoidSlottedEnv']


class CircuitSlottedEnv(CircuitEnv, metaclass=abc.ABCMeta):
    @property
    def act_space(self) -> Box:
        return Box.from_bound(bound=(-1, 1))

    @property
    def total_steps(self):
        return len(self._evaluator.parameters)


class CircuitVoidSlottedEnv(CircuitSlottedEnv):
    def _get_reward_from_scores(self):
        return 0 if len(self._running_actions) < self.total_steps else self._running_scores[-1]

    def _reset_internal_state(self):
        super(CircuitVoidSlottedEnv, self)._reset_internal_state()
        self._running_values = self.act_space.zeros()

    def step(self, action):
        counter = len(self._running_actions)
        param = self._evaluator.parameters[counter - 1]
        value = self._evaluator.denormalize(param, action, source_bound=(-1, 1))
        self._running_values[counter - 1] = value
        self._running_actions.append(action)
        return self._execute(action, self._running_values)


class CircuitInitSlottedEnv(CircuitSlottedEnv):
    def __init__(self, evaluator: Evaluator, benchmark, obs_space=True, *, initial_values):
        super().__init__(evaluator, benchmark, obs_space)
        self._initial_values = self._evaluator.formalize_as_numpy_array(initial_values)

    def _reset_internal_state(self):
        super(CircuitInitSlottedEnv, self)._reset_internal_state()
        self._running_values = self.origin.param.copy()

    @property
    def origin(self):
        if not hasattr(self, '__cache_initial__'):
            initial_features = self._evaluator(self._initial_values)
            initial_score = self._benchmark(self._initial_values, initial_features)
            initial_obs, initial_info = self._get_obsinfo_from_features(initial_features)

            assert initial_features is not None, "Initial point cannot be evaluated with failure."

            data = AttrDict(features=initial_features, score=initial_score,
                            obs=initial_obs, info=initial_info, param=self._initial_values)
            setattr(self, '__cache_initial__', data)

        return getattr(self, '__cache_initial__')

    def reset(self):
        self._reset_internal_state()
        return self.origin.obs

    # def _get_reward_from_scores(self):
    #     return self._running_scores[-1] - self.origin.score
    #
    def _get_reward_from_scores(self):
        return 0 if len(self._running_actions) < self.total_steps else self._running_scores[-1]

    # def _get_reward_from_scores(self):
    #     old_score = self._running_scores[-2] if len(self._running_scores) > 1 else 0
    #     now_score = self._running_scores[-1]
    #     return now_score - old_score

    def step(self, action):
        params_dynamic = np.ones(5) * 10000
        params_dynamic = np.concatenate((params_dynamic, np.ones(5) * 1000))
        params_dynamic = np.concatenate((params_dynamic, np.ones(6) * 5))
        params_dynamic = np.concatenate((params_dynamic, [50, 2500]))

        # upper bounder
        params_upper = np.ones(5) * 150000
        params_upper = np.concatenate((params_upper, np.ones(5) * 2000))
        params_upper = np.concatenate((params_upper, np.ones(6) * 20))
        params_upper = np.concatenate((params_upper, [100, 5000]))

        # set lower bounds
        params_lower = np.ones(5) * 220
        params_lower = np.concatenate((params_lower, np.ones(5) * 180))
        params_lower = np.concatenate((params_lower, np.ones(6) * 1))
        params_lower = np.concatenate((params_lower, [1, 10]))
        # print(params_lower, params_upper)

        # set parameter steps
        params_step = np.ones(5) * 10
        params_step = np.concatenate((params_step, np.ones(5) * 50))
        params_step = np.concatenate((params_step, np.ones(6) * 1))
        params_step = np.concatenate((params_step, [1, 10]))

        def round_params(params):
            # round w
            for i in range(len(params)):
                # round to steps
                params[i] = params_lower[i] + (params[i] - params_lower[i]) // params_step[i] * params_step[i]

                # constrain max and min
                params[i] = min(params[i], params_upper[i])
                params[i] = max(params[i], params_lower[i])
                params[i] = int(params[i])

            return params

        counter = len(self._running_actions)
        param = self._evaluator.parameters[counter - 1]
        print(action)
        value = self._evaluator.denormalize(param, action, source_bound=(-1, 1))
        self._running_values[counter - 1] = self._initial_values[counter - 1] + params_dynamic[counter - 1] * value * 0.2
        self._running_values = round_params(self._running_values)
        print(self._running_values)
        print(self.scores)
        print(value)

        obs, reward, terminated, info = self._execute(action, self._running_values)
        print(obs, reward)
        return obs, reward, terminated, info
