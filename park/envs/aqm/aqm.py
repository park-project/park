import park 
from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.spaces.box import Box

import numpy as np
from subprocess import Popen
import os

from time import time, sleep

from park.envs.aqm.mahimahi_interface import MahimahiInterface

class AQMEnv(core.Env):
	"""
	TODO: write a description
	"""
	def __init__(self):
		
		# Setup Mahimahi

		# TODO: check os

		mahimahi_path = park.__path__[0]+"/envs/aqm/mahimahi"

		if os.path.exists (mahimahi_path) == False :
			# create folder 
			Popen("mkdir %s" % mahimahi_path, shell=True).wait()
			# get mahimahi
			Popen("cd %s; git clone https://github.com/songtaohe/mahimahi.git" % (park.__path__[0]+"/envs/aqm/"), shell=True).wait()

			# Make mahimahi
			Popen("cd %s; git fetch; git checkout mahimahi_stable2; ./autogen.sh; ./configure; make; sudo make install" % mahimahi_path, shell=True).wait()


		#self.mm_delay = mahimahi_path + "/src/frontend/mm-delay"
		#self.mm_link = mahimahi_path + "/src/frontend/mm-link"
		
		self.mm_delay = "mm-delay"
		self.mm_link = "mm-link"
		
		# Setup link 

		self.linkDelay = config.aqm_link_delay
		self.uplinkTraceFile = config.aqm_uplink_trace
		self.downlinkTraceFile = config.aqm_downlink_trace


		# Setup workload generator

		self.workloadGeneratorSender   = "iperf -c 100.64.0.1 -P 1 -i 2 -t 10000"
		self.workloadGeneratorReceiver = "iperf -s -w 16m"
		self.workloadGeneratorKiller   = "pkill -9 iperf"

		with open("sender.sh","w") as fout:
			fout.write(self.workloadGeneratorSender+"\n")

		Popen("chmod a+x sender.sh", shell=True).wait()


		# Setup RL
		self.aqm_step_interval = config.aqm_step_interval
		self.aqm_step_num = config.aqm_step_num 

		self.state_space = Box(low=np.array([0, 0, 0]), high=np.array([1e4, 1e4, 1e3]))
		self.action_space = Box(low=np.array([0]), high=np.array([1]))

		self.step_counter = 0

		# Start 
		# self.reset()

		self.last_action_ts = None

	def _observe(self):
		ret = self.mahimahi.GetState()
		obs = np.array([ret['enqueued_packet'], ret['dequeued_packet'], ret['average_queueing_delay']])
		reward = -ret["average_queueing_delay"]
		info = {'message': ret['info']}
		return obs, reward, info

	def render(self):
		# TODO: depends on a switch (in __init__), visualize the mahimahi console
		pass


	def step(self, action):
		
		assert self.action_space.contains(action)

		self.step_counter += 1


		if self.last_action_ts is None:
			t_sleep = self.aqm_step_interval/1000.0
			sleep(t_sleep)
		else:
			t0 = time()
			t_sleep = self.last_action_ts + self.aqm_step_interval/1000.0 - t0

			if t_sleep > 0 :
				sleep(t_sleep)

		self.mahimahi.SetDropRate(action[0])
		self.last_action_ts = time()

		obs, reward, info = self._observe()
		assert self.state_space.contains(obs)
		info['wall_time_elapsed'] = t_sleep

		done = False
		if self.step_counter >= self.aqm_step_num:
			done = True

		return obs, reward, done, info
		

	def reset(self):
		# kill Mahimahi and workload generator

		Popen("pkill mm-delay", shell=True).wait()
		Popen(self.workloadGeneratorKiller, shell=True).wait()

		sleep(1.0)  # pkill has delay

		# start workload generator receiver 
		Popen(self.workloadGeneratorReceiver, shell=True)


		# start Mahimahi
		config_dict = {}

		config_dict["mmdelay"] = self.mm_delay
		config_dict["mmlink"] = self.mm_link
		config_dict["delay"] = int(self.linkDelay)

		config_dict["uplinktrace"] = self.uplinkTraceFile
		config_dict["downlinktrace"] = self.downlinkTraceFile

		config_dict["workloadSender"] = "./sender.sh"

		start_mahimahi_cmd = \
		"%(mmdelay)s %(delay)d %(mmlink)s %(uplinktrace)s %(downlinktrace)s  \
		--meter-uplink --meter-uplink-delay --uplink-queue=pie --downlink-queue=infinite --uplink-queue-args=\"packets=2000, qdelay_ref=20, max_burst=1\" \
		%(workloadSender)s "% config_dict

		Popen(start_mahimahi_cmd, shell=True)

		sleep(1.0)  # mahimahi start delay

		# Connect to Mahimahi
		self.mahimahi = MahimahiInterface()

		self.mahimahi.ConnectToMahimahi()


		# Gain control
		self.mahimahi.SetRLState(1)

		# get obs 
		obs, _, _ = self._observe()
		self.step_counter = 0

		return obs 



	def seed(self, seed=None):
		# no controllable randomness
		pass

	def clean(self):
		Popen("pkill mm-delay", shell=True).wait()
		Popen(self.workloadGeneratorKiller, shell=True).wait()

		Popen("rm mahimahi_pipe1 mahimahi_pipe2 sender.sh", shell=True).wait()

		sleep(1.0)