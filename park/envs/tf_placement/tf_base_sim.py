import networkx as nx
from simulator import *
from utils import get_op_costs
import copy

class ImportantOpsSimulator(Simulator):

  def __init__(self, mg, op_perf, step_stats, devices, params=dict(),
                cost_d=None, out_d=None):
    """
      Init for Simulator class
      Args:
        devices: list of device names
        params: dictionary of parameters to use in simulator
          some parameters can be missing
    """
    if cost_d is None:
      cost_d, _ = get_op_costs(step_stats)

    temp_mem = {}
    if out_d is None:
      out_d = {}
      for op in op_perf:
        out_d[op.node] = op.op_memory.output_memory
        temp_mem[op.node] = op.temporary_memory_size

      for dev_stats in step_stats.dev_stats:
          for node_stats in dev_stats.node_stats:
                  node = node_stats.node_name
                  for output in node_stats.output:
                      allocation = output.tensor_description.allocation_description
                      num_bytes = allocation.requested_bytes
                      out_d[node] = [num_bytes]
                      break

    for i, dev in enumerate(devices):
      devices[i] = '/' + dev.split('/')[-1]

    for node in mg.graph_def.node:
      d = node.device
      # if 'CPU' in d: print(node, d)
      node.device = '/' + d.split('/')[-1]

    self.temp_mem = temp_mem
    Simulator.__init__(self, mg, cost_d, out_d, devices, params, disable_cpu=True)

  def simulate(self, pl, old=False, sim_mem_usage=False):
    
    for k, v in pl.items():
      pl[k] = self.devices[int(v)]

    if old:
      r, f = Simulator.old_simulate(self, pl)
    else:
      r, f = Simulator.simulate(self, pl)

    self.f = f

    start_t = {}
    for node in self.metagraph.graph_def.node:
      n = node.name
      start_t[n] = f[n].start_time

    if sim_mem_usage:

        mem_q = []

        for n, t in start_t.items():

            mem = sum(self.output_dict[n])
            if mem == 0:
                continue

            # temp_mem = self.temp_mem[n]

            dev = self.devices.index(f[n].device)

            # mem_q.append((t, mem, dev))
            mem_q.append((t, '+', mem, dev))
            # mem_q.append((t, '+', mem + temp_mem, dev))
            # mem_q.append((t + f[n].compute_cost, '-', temp_mem, dev))

            t_out_done = t
            for c in f[n].children:
                t_out_done = max(t_out_done, int(f[c].start_time) + int(f[c].compute_cost) - 1)

            # mem_q.append((t_out_done, -mem, dev))
            mem_q.append((t_out_done, '-', -mem, dev))

        mem_q.sort()

        mem_utils = [0]* len(self.devices)
        peak_utils = [0]* len(self.devices)

        for (t, _, mem, dev) in mem_q:
            mem_utils[dev] += mem

            if mem_utils[dev] > peak_utils[dev]:
              peak_utils[dev] = mem_utils[dev]

            # assert mem_utils[dev] >= 0

        return r, start_t, peak_utils
      
    return r, start_t
 
  def add_sim_to_step_stats(self, step_stats):
    return Simulator.add_sim_to_step_stats(self, self.f, step_stats)
