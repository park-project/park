import abc
import os
import subprocess

from park.envs.circuit_sim.evaluator import Evaluator
from park.envs.circuit_sim.context import Context


class Circuit(object, metaclass=abc.ABCMeta):
    __UNITS__ = {'a': 1e-18, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3, 'k': 1e3, 'x': 1e6, 'g': 1e9}

    def __init__(self, context: Context = None):
        self._context = context

    @property
    def context(self):
        return self._context or Context.current_context()

    @property
    @abc.abstractmethod
    def parameters(self) -> tuple:
        pass

    @abc.abstractmethod
    def run(self, tmp_path, values):
        pass

    def evaluate(self, values, debug=None):
        context = self.context
        assert context is not None and context.opened, "Context must be specified and opened"
        return context.evaluate(self, values, debug)

    def evaluate_batch(self, values, debug=None):
        context = self.context
        assert context is not None and context.opened, "Context must be specified and opened"
        return context.evaluate_batch(self, values, debug)

    @staticmethod
    def _run_hspice(name, work_path) -> str:
        pipe = subprocess.Popen(['hspice', name + ' > result'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_path)
        pipe.communicate()
        with open(os.path.join(work_path, 'result'), 'r') as reader:
            return reader.read()

    @staticmethod
    def _run_spectre(name, work_path) -> str:
        pipe = subprocess.Popen(['spectre', name, '-format=psfascii'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_path)
        pipe.communicate()
        with open(os.path.join(work_path, 'spectre.dc')) as reader:
            return reader.read()

    @classmethod
    def number_from_string(cls, string: str):
        string = string.strip()
        for unit, value in cls.__UNITS__.items():
            if string.endswith(unit):
                return eval(string[:-1]) * value
        return eval(string)

    def evaluator(self, **kwargs):
        return Evaluator(self, **kwargs)

    @property
    @abc.abstractmethod
    def meta(self):
        pass
