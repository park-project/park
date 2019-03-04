import os
import re

import park
from park.envs.circuit_sim.circuit import Circuit, export_circuit
from park.envs.circuit_sim.utility.io import load_txt, dump_txt
from park.envs.circuit_sim.utility.misc import AttrDict

__all__ = ['TwoStageTransimpedanceAmplifier']


@export_circuit
class TwoStageTransimpedanceAmplifier(Circuit):
    @property
    def simdata(self):
        if not hasattr(self, '__cache_simdata__'):
            data = load_txt(park.envs.circuit_sim.__file__ + 'library/transimpedance/two_stage.circuit')
            setattr(self, '__cache_simdata__', data)
        return getattr(self, '__cache_simdata__')

    @property
    def libdata(self):
        if not hasattr(self, '__cache_libdata__'):
            data = load_txt(park.envs.circuit_sim.__file__ + 'library/transimpedance/two_stage.circuit')
            setattr(self, '__cache_libdata__', data)
        return getattr(self, '__cache_libdata__')


    @property
    def parameters(self):
        return 'W1', 'W3', 'W5', 'W6', 'L1', 'L3', 'L5', 'L6', 'RF', 'R6', 'A'

    @classmethod
    def _get_offset(cls, result):
        regex = re.compile(r'.*\*\*warning\*\*.*')
        return sum(bool(regex.findall(line)) for line in result[22: 30])

    @classmethod
    def _get_transistor_states(cls, offset, result):
        states = AttrDict(M1=[1, 0], M3=[0, 1], M2=[1, 0], M4=[0, 1], M5=[1, 0], M6=[1, 0])
        regex = re.compile(r'.*[\s\w]+\s+([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+'
                           r'([+\-.\d\w]+)\s+([+\-.\d\w]+)\s+([+\-.\d\w]+).*')
        for line in result[offset + 86: offset + 106]:
            data = regex.findall(line)[0]
            states.M1.append(cls.number_from_string(data[0]))
            states.M3.append(cls.number_from_string(data[1]))
            states.M2.append(cls.number_from_string(data[2]))
            states.M4.append(cls.number_from_string(data[3]))
            states.M5.append(cls.number_from_string(data[4]))
            states.M6.append(cls.number_from_string(data[5]))
        return states

    @classmethod
    def _get_curve(cls, offset, result):
        regex = re.compile(r'.*\s+([\d\w.+]+)\s+([\d\w.+\-]+)\s+([\d\w.+\-]+).*')
        curve = AttrDict(frequency=[], magnitude=[], phase=[])
        for line in result[offset + 117: offset + 4118]:
            data = regex.findall(line)[0]
            curve.frequency.append(cls.number_from_string(data[0]))
            curve.magnitude.append(cls.number_from_string(data[1]))
            curve.phases.append(cls.number_from_string(data[2]))
        return curve

    @classmethod
    def _get_metrics(cls, values, offset, result):
        proto_midband_closedloop_transimped = re.findall(r'.*cl0=\s*([+\-.\d\w]+).*', result[offset + 4119])[0]
        proto_magnitude_peaking = re.findall(r'.*peaking=\s*([+\-.\d\w]+).*', result[offset + 4122])[0]
        proto_bandwidth = re.findall(r'.*clf3db=\s*([+\-.\d\w]+).*', result[offset + 4123])[0]
        proto_noise_atf3db = re.findall(r'.*noisef3db=\s*([+\-.\d\w]+).*', result[offset + 4124])[0]
        proto_power = re.findall(r'.*dissipation=\s*([+\-.\d\w]+).*', result[offset + 53])[0]

        metrics = AttrDict(
            midband_closedloop_transimped=cls.number_from_string(proto_midband_closedloop_transimped),
            magnitude_peaking=cls.number_from_string(proto_magnitude_peaking),
            bandwidth=cls.number_from_string(proto_bandwidth),
            noise_atf3db=cls.number_from_string(proto_noise_atf3db),
            power=cls.number_from_string(proto_power)
        )
        metrics.area = 0
        metrics.area += 2 * (values.W1 * values.L1 + values.W3 * values.L3)
        metrics.area += values.W5 * values.L5 + values.W6 * values.L6
        regex = re.compile(r'.*region\s*Saturati\s*Saturati\s*Saturati\s*Saturati\s*Saturati\s*Saturati.*')
        metrics.saturated = bool(regex.match(result[offset + 85]))
        return metrics

    @classmethod
    def _get_voltages(cls, auxiliaries):
        voltages = []
        regex = re.compile(r'.*=\s*(\d+\.\d*\w*)')
        for line in auxiliaries[12: 18]:
            voltages.append(cls.number_from_string(regex.search(line).group(1)))
        return voltages

    def run(self, tmp_path, values):
        with open(os.path.join(tmp_path, 'two_amp.sp'), 'w') as writer:
            args_units = list('uuuuuuuukk') + ['']
            data = self.simdata
            for key, tail in zip(self.parameters, args_units):
                data = data.replace(f'$(params:{key})', f'%f{tail}' % getattr(values, key))
            writer.write(data)
        dump_txt(self.libdata, os.path.join(tmp_path, 'ee214.hspice'))
        try_out = 30
        while True:
            try_out -= 1
            if try_out == 0:
                raise RuntimeError("Have tried several times, ensure failed")
            result = self._run_hspice('two_amp.sp', tmp_path).split('\n')
            with open(os.path.join(tmp_path, 'two_amp.ic0'), 'r') as reader:
                auxiliaries = reader.read().split('\n')

            offset = self._get_offset(result)
            transistor_states = self._get_transistor_states(offset, result)
            try:
                curve = self._get_curve(offset, result)
            except:
                print('failed!!!')
                continue
            metrics = self._get_metrics(values, offset, result)
            voltages = self._get_voltages(auxiliaries)
            break
        return AttrDict(
            transistor_states=transistor_states,
            curve=curve,
            metrics=metrics,
            voltages=voltages,
            tmp_path=os.path.abspath(tmp_path)
        )

    @property
    def out_space(self):
        return {}
