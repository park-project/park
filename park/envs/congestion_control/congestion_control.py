import numpy as np
import subprocess as sh
import os
import socket
import sys
from time import time, sleep
import threading

import park
from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.spaces.box import Box

import capnp
capnp.remove_import_hook()
ccp_capnp = capnp.load(park.__path__[0]+"/envs/congestion_control/park/ccp.capnp")

class CcpRlAgentImpl(ccp_capnp.RLAgent.Server):
    def __init__(self, agent):
        self.agent = agent

    def getAction(self, observation, _context, **kwargs):
        obs = [
            observation.bytesAcked,
            observation.bytesMisordered,
            observation.ecnBytes,
            observation.packetsAcked,
            observation.packetsMisordered,
            observation.ecnPackets,
            observation.loss,
            observation.timeout,
            observation.bytesInFlight,
            observation.packetsInFlight,
            observation.bytesPending,
            observation.rtt,
            observation.rin,
            observation.rout,
        ]

        act = self.agent.get_action(obs, 0, 0, 0)
        c, r = act
        action = ccp_capnp.Action.new_message(cwnd=int(c), rate=int(r))
        return action

def run_forever(sock, agent):
    connection, _ = sock.accept()
    server = capnp.TwoPartyServer(connection, bootstrap=CcpRlAgentImpl(agent))
    server.on_disconnect().wait()

class CongestionControlEnv(core.SysEnv):
    def __init__(self):
        # check if the operating system is ubuntu
        if sys.platform != 'linux' and sys.platform != 'linux2':
            raise OSError('Congetsion control environment only tested on Linux.')

        print("====> Install Dependencies")
        if os.getuid() != 0:
            sh.call("sudo apt install -y git build-essential autoconf automake capnproto", shell=True)
            sh.call("sudo add-apt-repository -y ppa:keithw/mahimahi", shell=True)
            sh.call("sudo apt-get -y update", shell=True)
            sh.call("sudo apt-get -y install mahimahi", shell=True)
            sh.call("sudo sysctl -w net.ipv4.ip_forward=1", shell=True)
        else:
            sh.call("apt install -y git build-essential autoconf automake capnproto", shell=True)
            sh.call("add-apt-repository -y ppa:keithw/mahimahi", shell=True)
            sh.call("apt-get -y update", shell=True)
            sh.call("apt-get -y install mahimahi", shell=True)
            sh.call("sysctl -w net.ipv4.ip_forward=1", shell=True)
        print("======> Done")

        self.setup_ccp_shim()
        self.setup_mahimahi()

        sh.run("sudo rm -rf /tmp/park-ccp", shell=True)

        # state_space
        #
        # biggest BDP = 1200 packets = 1.8e6 Bytes
        #
        # bytesAcked         UInt64; at most one BDP
        # bytesMisordered    UInt64; at most one BDP
        # ecnBytes           UInt64; at most one BDP
        # packetsAcked       UInt64; at most one BDP / MSS
        # packetsMisordered  UInt64; at most one BDP / MSS
        # ecnPackets         UInt64; at most one BDP / MSS
        # loss               UInt64; at most one BDP / MSS
        # timeout            Bool;
        # bytesInFlight      UInt64; at most one BDP
        # packetsInFlight    UInt64; at most one BDP / MSS
        # bytesPending       UInt64; ignore
        # rtt                UInt64; [0ms, 300ms]
        # rin                UInt64; [0 Byte/s, 1GByte/s]
        # rout               UInt64; [0 Byte/s, 1GByte/s]
        self.state_space = Box(
            low=np.array([0] * 14),
            high=np.array([1.8e6, 1.8e6, 1.8e6, 1200, 1200, 1200, 1200, 1, 1.8e6, 1200, 0, 300e3, 1e9, 1e9]),
        )

        # action_space
        # cwnd = [0, 4800 = 4BDP]
        # rate = [0, 2e9 = 2 * max rate]
        self.action_space = Box(low=np.array([0, 0]), high=np.array([4800, 2e9]))

        print("====> Done with init")

    def run(self, agent_constructor, agent_parameters):
        self.agent = agent_constructor(self.observation_space, self.action_space, *agent_parameters)

        # setup capnp
        sock = socket.socket(family=socket.AF_UNIX)
        sock.bind("/tmp/park-ccp")
        sock.listen()

        # start rlagent rpc server that ccp talks to
        threading.Thread(target=run_forever, args=(sock, self.agent)).start()

        # start ccp shim
        cong_env_path = park.__path__[0] + "/envs/congestion_control"
        sh.Popen("sudo " + os.path.join(cong_env_path, "park/target/release/park"), shell=True)

        # Start
        self.reset()

    def reset(self):
        # kill Mahimahi and workload generator
        sh.Popen("pkill mm-delay", shell=True).wait()
        sh.Popen(self.workloadGeneratorKiller, shell=True).wait()

        sleep(1.0)  # pkill has delay

        # start workload generator receiver
        sh.Popen(self.workloadGeneratorReceiver, shell=True)

        # start Mahimahi
        config_dict = {}

        config_dict["mmdelay"] = "mm-delay"
        config_dict["mmlink"] = "mm-link"
        config_dict["delay"] = int(self.linkDelay)

        config_dict["uplinktrace"] = self.uplinkTraceFile
        config_dict["downlinktrace"] = self.downlinkTraceFile

        config_dict["workloadSender"] = "./sender.sh"

        start_mahimahi_cmd = \
                "%(mmdelay)s %(delay)d %(mmlink)s %(uplinktrace)s %(downlinktrace)s  \
                --uplink-queue=droptail --downlink-queue=droptail --uplink-queue-args=\"packets=2000\" --downlink-queue-args=\"packets=2000\"\
                %(workloadSender)s "% config_dict

        sh.Popen(start_mahimahi_cmd, shell=True)

        sleep(1.0)  # mahimahi start delay

    def setup_mahimahi(self):
        print("====> Mahimahi setup")

        env_path = park.__path__[0] + "/envs/congestion_control"

        # Setup link
        #self.linkDelay = config.congestion_control_link_delay
        #self.uplinkTraceFile = config.congestion_control_uplink_trace
        #self.downlinkTraceFile = config.congestion_control_downlink_trace
        self.linkDelay = 25
        self.uplinkTraceFile = os.path.join(env_path, "bw48.mahi")
        self.downlinkTraceFile = os.path.join(env_path, "bw48.mahi")

        # Setup workload generator
        self.workloadGeneratorSender   = "iperf -c 100.64.0.1 -Z ccp -P 1 -i 2 -t 10000"
        self.workloadGeneratorReceiver = "iperf -s -w 16m"
        self.workloadGeneratorKiller   = "pkill -9 iperf"

        with open("sender.sh","w") as fout:
            fout.write(self.workloadGeneratorSender+"\n")

        sh.Popen("chmod a+x sender.sh", shell=True).wait()

    def setup_ccp_shim(self):
        cong_env_path = park.__path__[0] + "/envs/congestion_control"

        # ccp-kernel
        if not os.path.exists(cong_env_path + "/ccp-kernel"):
            print("====> Downloading ccp-kernel")
            sh.run("git clone --recursive https://github.com/ccp-project/ccp-kernel.git {}".format(cong_env_path + "/ccp-kernel"), shell=True)

        loaded = sh.check_output("lsmod | grep -c ccp", shell=True)
        if int(loaded.strip()) == 0:
            print("====> Loading ccp-kernel")
            sh.run("make && sudo ./ccp_kernel_load ipc=0", cwd=cong_env_path + "/ccp-kernel", shell=True)

        try:
            print("====> Building ccp shim")
            sh.check_call("cargo build --release", cwd=cong_env_path + "/park", shell=True)
        except sh.CalledProcessError:
            print("====> Installing rust")
            sh.check_call("bash rust-install.sh", cwd=cong_env_path, shell=True)
            print("====> Building ccp shim")
            sh.check_call("~/.cargo/bin/cargo build --release", cwd=cong_env_path + "/park", shell=True)
