# /usr/bin/python
# python 2.7 only because of protobuf

import os
import dill
import json
import wget
import string
import zipfile
import subprocess
import numpy as np
from sys import platform
from urllib.request import urlopen

import park
from park import core, spaces, logger
from park.param import config
from park.utils import seeding


class ABREnv(core.SysEnv):
    """
    Adapt bitrate during a video playback with varying network conditions.
    The objective is to (1) reduce stall (2) increase video quality and
    (3) reduce switching between bitrate levels. Ideally, we would want to
    *simultaneously* optimize the objectives in all dimensions.

    * STATE *
        [The throughput estimation of the past chunk (chunk size / elapsed time),
        download time (i.e., elapsed time since last action), current buffer ahead,
        number of the chunks until the end, the bitrate choice for the past chunk,
        current chunk size of bitrate 1, chunk size of bitrate 2,
        ..., chunk size of bitrate 5]

        Note: we need the selected bitrate for the past chunk because reward has
        a term for bitrate change, a fully observable MDP needs the bitrate for past chunk

    * ACTIONS *
        Which bitrate to choose for the current chunk, represented as an integer in [0, 4]

    * REWARD *
        At current time t, the selected bitrate is b_t, the stall time between
        t to t + 1 is s_t, then the reward r_t is
        b_{t} - 4.3 * s_{t} - |b_t - b_{t-1}|
        Note: there are different definitions of combining multiple objectives in the reward,
        check Section 5.1 of the first reference below.
    
    * REFERENCE *
        Section 4.2, Section 5.1
        Neural Adaptive Video Streaming with Pensieve
        H Mao, R Netravali, M Alizadeh
        https://dl.acm.org/citation.cfm?id=3098843

        Figure 1b, Section 6.2 and Appendix J
        Variance Reduction for Reinforcement Learning in Input-Driven Environments.
        H Mao, SB Venkatakrishnan, M Schwarzkopf, M Alizadeh.
        https://openreview.net/forum?id=Hyg1G2AqtQ
    
        A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP
        X Yin, A Jindal, V Sekar, B Sinopoli 
        https://dl.acm.org/citation.cfm?id=2787486
    """
    def __init__(self):
        # check if the operating system is ubuntu
        if platform != 'linux' and platform != 'linux2':
            raise OSError('Real ABR environment only tested with Ubuntu 16.04.')

        # check/download the video files
        if not os.path.exists(park.__path__[0] + '/envs/abr/video_server/'):
            wget.download(
                'https://www.dropbox.com/s/t1igk37y4qtmtgt/video_server.zip?dl=1',
                out=park.__path__[0] + '/envs/abr/')
            with zipfile.ZipFile(
                 park.__path__[0] + '/envs/abr/video_server.zip', 'r') as zip_f:
                zip_f.extractall(park.__path__[0] + '/envs/abr/')

        # check/download the browser files
        if not os.path.exists(park.__path__[0] + '/envs/abr/abr_browser_dir/'):
            wget.download(
                'https://www.dropbox.com/s/a3vadqokeg3x60l/abr_browser_dir.zip?dl=1',
                out=park.__path__[0] + '/envs/abr/')
            with zipfile.ZipFile(
                 park.__path__[0] + '/envs/abr/abr_browser_dir.zip', 'r') as zip_f:
                zip_f.extractall(park.__path__[0] + '/envs/abr/')
            os.system('chmod 777 ' + park.__path__[0] + '/envs/abr/abr_browser_dir/chromedriver')

        # check/download the trace files
        if not os.path.exists(park.__path__[0] + '/envs/abr/cooked_traces/'):
            wget.download(
                'https://www.dropbox.com/s/qw0tmgayh5d6714/cooked_traces.zip?dl=1',
                out=park.__path__[0] + '/envs/abr/')
            with zipfile.ZipFile(
                 park.__path__[0] + '/envs/abr/cooked_traces.zip', 'r') as zip_f:
                zip_f.extractall(park.__path__[0] + '/envs/abr/')

        # check if the manifest file is copied to the right place (last step in setup.py)
        if not os.path.exists('/var/www/html/Manifest.mpd'):
            os.system('python ' + park.__path__[0] + '/envs/abr/setup.py')

        # observation and action space
        self.setup_space()

        # load all trace file names
        self.all_traces = os.listdir(park.__path__[0] + '/envs/abr/cooked_traces/')

        # random seed
        self.seed(config.seed)

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.obs_low = np.array([0] * 11)
        self.obs_high = np.array([
            10e6, 100, 100, 500, 5, 10e6, 10e6, 10e6, 10e6, 10e6, 10e6])
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    def run(self, agent, *args, **kwargs):
        # kill all previously running programs
        os.system("ps aux | grep -ie mm-delay | awk '{print $2}' | xargs kill -9")
        os.system("ps aux | grep -ie mm-link | awk '{print $2}' | xargs kill -9")
        os.system("ps aux | grep -ie abr | awk '{print $2}' | xargs kill -9")
        os.system("sudo sysctl -w net.ipv4.ip_forward=1")

        trace_file = self.np_random.choice(self.all_traces)

        ip_data = json.loads(urlopen("http://ip.jsontest.com/").read())
        ip = str(ip_data['ip'])

        curr_path = os.getcwd()

        # pickle the agent constructor into a tmp file, so that the rl server
        # *inside* the mahimahi shell can load and use it
        agent_file_name = '/tmp/abr_agent_' + ''.join(
            self.np_random.choice([i for i in string.ascii_letters + string.digits]) for _ in range(10))
        with open(agent_file_name, 'wb') as f:
            dill.dump((agent, self.observation_space, self.action_space, args, kwargs), f)

        # start real ABR environment
        p = subprocess.Popen('mm-delay 40' +
            ' mm-link ' + park.__path__[0] + '/envs/abr/12mbps ' +
            park.__path__[0] + '/envs/abr/cooked_traces/' + trace_file +
            ' /usr/bin/python ' + park.__path__[0] + '/envs/abr/run_video.py ' +
            ip + ' ' + '320' + ' ' + '0' + ' ' + '1' + ' ' +
            curr_path + ' ' + agent_file_name,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            shell=True)

        p.wait()
