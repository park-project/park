import numpy as np

from park.envs.circuit_sim.circuit import Circuit


class Evaluator(object):
    def __init__(self, circuit: Circuit, logger=None):
        self._circuit = circuit
        self._fixed_values = {}
        self._lbound = {}
        self._ubound = {}
        self._logger = logger or get_default_logger(self.__class__.__name__)

    @property
    def circuit(self):
        return self._circuit

    @property
    def lower_bound(self):
        return [self._lbound[k] for k in self.parameters]

    @property
    def upper_bound(self):
        return [self._ubound[k] for k in self.parameters]

    @property
    def bound(self):
        return self.lower_bound, self.upper_bound

    def set_bound(self, parameter, lower_bound, upper_bound):
        if lower_bound is not None:
            self._lbound[parameter] = lower_bound
        if upper_bound is not None:
            self._ubound[parameter] = upper_bound

    def random_values(self):
        return {k: np.random.rand() * (self._ubound[k] - self._lbound[k]) + self._lbound[k] for k in self.parameters}

    def sample(self, debug=None):
        values = self.random_values()
        return self(values, debug)

    def sample_batch(self, size, debug=None):
        values = [self.random_values() for _ in range(size)]
        return self.batch(values, debug)

    @property
    def parameters(self) -> tuple:
        return self._circuit.parameters

    @property
    def result_meta(self):
        return self._circuit.meta

    def set_fixed(self, parameter, value):
        self._fixed_values[parameter] = value

    def formalize(self, values):
        if isinstance(values, dict):
            assert set(values.keys()) == set(self.parameters)
            return values
        else:
            values = tuple(values)
            assert len(values) == len(self.parameters)
            return dict(zip(self.parameters, values))

    def __call__(self, values, debug=None):
        values = self.formalize(values)
        try:
            return self._circuit.evaluate(values, debug)
        except KeyboardInterrupt:
            raise
        except:
            self._logger.exception("An exception occurred when evaluate single value.")
            return None

    def batch(self, values, debug=None):
        values = tuple(map(self.formalize, values))
        try:
            return self._circuit.evaluate_batch(values, debug)
        except KeyboardInterrupt:
            raise
        except:
            self._logger.exception("An exception occurred when evaluate batch values.")
            return None
