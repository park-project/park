import os

import math
import re
import shutil

from park.envs.circuit_sim.circuit import Circuit
from park.envs.circuit_sim.context import LocalContext
from park.envs.circuit_sim.evaluator import CircuitSimIncrementalEnv
from park.envs.circuit_sim.utils import AttrDict
from park.spaces import Box

__all__ = ['OneStageVoltageAmplifier', 'make_env']


class OneStageVoltageAmplifier(Circuit):
    @property
    def parameters(self) -> tuple:
        return 'W0', 'L0', 'W1', 'L1', 'W2', 'L2', 'W4', 'L4'

    @classmethod
    def _get_metrics(cls, offset, values, result):
        metrics = AttrDict(
            power=cls.number_from_string(re.findall(r'.*=\s*([+\-.\w\d]+).*', result[offset + 276])[0]),
            gain=cls.number_from_string(re.findall(r'.*1\.00000\s+([+\-.\d\w]+).*', result[offset + 330])[0])
        )
        gain_10g = cls.number_from_string(re.findall('.*10\.00000g\s*([+\-.\w\d]+).*', result[offset + 10330])[0])
        if metrics.gain < 1:
            metrics.gain_bandwidth = 1
        elif gain_10g > 1:
            metrics.gain_bandwidth = 1e10
        else:
            gain_bandwidth = re.findall(r'.*gbw=\s+([+\-.\d\w]+).*', result[offset + 10335])[0]
            metrics.gain_bandwidth = cls.number_from_string(gain_bandwidth)

        if gain_10g >= metrics.gain / math.sqrt(2):
            metrics.bandwidth = 1e10
        else:
            bandwidth = re.findall(r'.*f3db=\s+([+\-.\d\w]+).*', result[offset + 10334])[0]
            metrics.bandwidth = cls.number_from_string(bandwidth)

        metrics.area = 0
        metrics.area += values.W0 * values.L0 + values.W1 * values.L1
        metrics.area += (values.W2 * values.L2 + values.W4 * values.L4) * 2

        regex = re.compile(r'.*region\s*Saturati\s*Saturati\s*Saturati\s*Saturati\s*Saturati\s*Saturati.*')
        metrics.saturated = bool(regex.match(result[offset + 298]))
        return metrics

    @classmethod
    def _get_offset(cls, result):
        return 4 if re.findall(r'.*(no\s+convergence)', result[255]) else 0

    @classmethod
    def _get_curve(cls, offset, result):
        regex = re.compile(r'.*\s+([\d\w.+]+)\s+([\d\w.+\-]+)\s+([\d\w.+\-]+).*')
        curve = AttrDict(frequency=[], magnitude=[], phase=[])
        for line in result[offset + 330: offset + 10331]:
            data = regex.findall(line)
            curve.frequency.append(cls.number_from_string(data[0][0]))
            curve.magnitude.append(cls.number_from_string(data[0][1]))
            curve.phase.append(cls.number_from_string(data[0][2]))
        return curve

    @classmethod
    def _get_transistor_states(cls, offset, result):
        transistor_states = AttrDict(M0=[1, 0], M1=[1, 0], M2=[1, 0], M3=[1, 0], M4=[0, 1], M5=[0, 1])
        regex = re.compile(r'.*[\s\w]+\s+([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+'
                           r'([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+([+\-.\d\w]+).*')
        for line in result[offset + 299: offset + 319]:
            states = regex.findall(line)
            assert states
            for i in range(6):
                state = cls.number_from_string(states[0][i])
                getattr(transistor_states, f'M{i}').append(state)
        return transistor_states

    @classmethod
    def _get_voltages(cls, auxiliaries):
        voltages = auxiliaries.split('\n')[12:20]
        regex = re.compile(r'.*=\s*(\d+\.\d*\w*)')
        for i in range(len(voltages)):
            match = regex.search(voltages[i])
            voltages[i] = cls.number_from_string(match.group(1))
        return voltages

    def run(self, tmp_path, values):
        with open('library/voltage/one_stage.circuit') as reader, \
                open(os.path.join(tmp_path, 'one_amp.sp'), 'w') as writer:
            data = reader.read()
            arg_units = 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u'
            for key, tail in zip(self.parameters, arg_units):
                data = data.replace(f'$(params:{key})', f'%f{tail}' % getattr(values, key))
            writer.write(data)
        shutil.copy('library/sm046005-1j.hspice', tmp_path)

        result = self._run_hspice('one_amp.sp', tmp_path).split('\n')
        offset = self._get_offset(result)
        with open(os.path.join(tmp_path, 'one_amp.ic0'), 'r') as reader:
            voltages = self._get_voltages(reader.read())

        return AttrDict(
            voltages=voltages,
            curve=self._get_curve(offset, result),
            metrics=self._get_metrics(offset, values, result),
            transistor_states=self._get_transistor_states(offset, result),
            tmp_path=os.path.abspath(tmp_path)
        )

    @property
    def meta(self):
        return AttrDict(
            voltages=Box(shape=(8,)),
            curve=AttrDict(
                frequency=Box(shape=(10001,)),
                magnitude=Box(shape=(10001,)),
                phase=Box(shape=(10001,)),
            ),
            metrics=AttrDict(
                power=Box(shape=()),
                gain=Box(shape=()),
                gain_bandwidth=Box(shape=()),
                bandwidth=Box(shape=()),
                area=Box(shape=()),
                saturated=OptSpace(2, 'bool')
            ),
            transistor_states=AttrDict(
                M0=Box(shape=(22,)),
                M1=Box(shape=(22,)),
                M2=Box(shape=(22,)),
                M3=Box(shape=(22,)),
                M4=Box(shape=(22,)),
                M5=Box(shape=(22,)),
            )
        )


def make_env(env_type='incremental', path='./tmp', context=None, benchmark=None, specs=None, total_steps=3):
    context = context or LocalContext(path, debug=True)
    circuit = OneStageVoltageAmplifier(context)
    if env_type == 'incremental':
        environment = CircuitSimIncrementalEnv(circuit.evaluator(), specs, benchmark, total_steps)
    else:
        raise ValueError(f'invalid environment type of "{env_type}"')
    return environment
