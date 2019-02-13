import os
import sys
import zmq
import numpy as np
import multiprocessing as mp
from collections import OrderedDict

import park
from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.utils.ordered_set import OrderedSet
from park.utils.directed_graph import DirectedGraph
from park.envs.spark.dags_database import DAGsDatabase
from park.envs.spark.environment import Environment
from park.envs.spark.executor_tracking import ExecutorMap

try:
    import park.envs.spark.ipc_msg_pb2
except:
    os.system("protoc -I=./park/envs/spark/ --python_out=./park/envs/spark/ ./park/envs/spark/ipc_msg.proto")
    import park.envs.spark.ipc_msg_pb2


class SparkEnv(core.Env):
    """
    Interacting with a modified scheduling module in Apache Spark.
    See reference for more details.

    * STATE *
        Graph type of observation. It consists of features associated with each node (
        a tensor of dimension n * m, where n is number of nodes, m is number of features),
        and adjacency matrix (a sparse 0-1 matrix of dimension n * n).
        The features on each node is
        [number_of_executors_currently_in_this_job, is_current_executor_local_to_this_job,
         number_of_free_executors, total_work_remaining_on_this_node,
         number_of_tasks_remaining_on_this_node]

    * ACTIONS *
        Two dimensional action, [node_idx_to_schedule_next, number_of_executors_to_assign]
        Note: the set of available nodes has to contain node_idx, and the number of
        executors to assign must not exceed the limit. Both the available set and the limit
        are provided in the (auxiliary) state.

    * REWARD *
        Negative time elapsed for each job in the system since last action.
        For example, the virtual time was 0 for the last action, 4 jobs
        was in the system (either in the queue waiting or being processed),
        job 1 finished at time 1, job 2 finished at time 2.4 and job 3 and 4
        are still running at the next action. The next action is taken at
        time 5. Then the reward is - (1 * 1 + 1 * 2.4 + 2 * 5).
        Thus, the sum of the rewards would be negative of total
        (waiting + processing) time for all jobs.
    
    * REFERENCE *
        Section 6.1
        Learning Scheduling Algorithms for Data Processing Clusters
        H Mao, M Schwarzkopf, SB Venkatakrishnan, M Alizadeh
        https://arxiv.org/pdf/1810.01963.pdf
    """
    def __init__(self):
        # TODO: check if spark exists, download, build
        self.setup_space()
        # random seed
        self.seed(config.seed)

    def observe(self):
        return self.obs

    def step(self, action):

        assert self.action_space.contains(action)

        # put action into the queue for server to return to spark
        self.action_queue.put(action)

        # get next observation
        self.obs = self.obs_queue.get()

        return self.observe(), reward, done, None

    def reset(self, max_time=np.inf):
        # reset observation and action space
        self.setup_space()
        self.obs = None

        # restart spark and scheduling module
        park_path = park.__path__[0]
        os.system("ps aux | grep -ie spark-tpch | awk '{print $2}' | xargs kill -9")
        os.system(park_path + '/envs/spark/spark/sbin/stop-master.sh')
        os.system(park_path + '/envs/spark/spark/sbin/stop-slaves.sh')
        os.system(park_path + '/envs/spark/spark/sbin/stop-shuffle-service.sh')
        os.system(park_path + '/envs/spark/spark/sbin/start-master.sh')
        os.system(park_path + '/envs/spark/spark/sbin/start-slave.sh')
        os.system(park_path + '/envs/spark/spark/sbin/start-shuffle-service.sh')

        # start the server
        self.obs_queue = mp.Queue(1)
        self.act_queue = mp.Queue(1)
        server_process = SchedulingServer(self.obs_queue, self.act_queue)
        server_process.start()

        # start submitting jobs
        # TODO: sample a random set of jobs
        submit_script = 'python3 ' + park_path + '/envs/spark/submit_tpch.py'

        # get first observation
        self.obs = self.obs_queue.get()

        return self.observe()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.graph = DirectedGraph()
        # self.obs_node_low = np.array([0] * 6)
        # self.obs_node_high = np.array([config.exec_cap, 1, config.exec_cap, 1000, 100000, 1])
        # self.obs_edge_low = self.obs_edge_high = np.array([])  # features on nodes only
        # self.observation_space = spaces.Graph(
        #     node_feature_space=spaces.MultiBox(
        #         low=self.obs_node_low,
        #         high=self.obs_node_high,
        #         dtype=np.float32),
        #     edge_feature_space=spaces.MultiBox(
        #         low=self.obs_edge_low,
        #         high=self.obs_edge_high,
        #         dtype=np.float32))
        self.action_space = spaces.NodeInGraph(self.graph)


class SchedulingServer(mp.Process):
    def __init__(self):
        mp.Process.__init__(self, obs_queue, act_queue)
        self.obs_queue = obs_queue
        self.act_queue = act_queue
        self.dag_db = DAGsDatabase()
        self.exec_tracker = ExecutorMap()
        self.env = Environment(dag_db)
        self.exit = mp.Event()

        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.ipc_msg = ipc_msg_pb2.IPCMessage()
        self.ipc_reply = ipc_msg_pb2.IPCReply()

    def run(self):
        while not self.exit.is_set():
            msg = socket.recv()
            ipc_msg.ParseFromString(msg)

            if ipc_msg.msg_type == 'register':
                dag_db.add_new_app(ipc_msg.app_name, ipc_msg.app_id)
                env.add_job_dag(ipc_msg.app_id)
                ipc_reply.msg = \
                    "external scheduler register app " + str(ipc_msg.app_name)

            elif ipc_msg.msg_type == 'bind':
                env.bind_exec_id(ipc_msg.app_id, ipc_msg.exec_id, ipc_msg.track_id)
                ipc_reply.msg = \
                    "external scheduler bind app_id " + \
                    str(ipc_msg.app_id) + " exec_id " + \
                    str(ipc_msg.exec_id) + " on track_id " + \
                    str(ipc_msg.track_id)

            elif ipc_msg.msg_type == 'inform':
                env.complete_tasks(
                    ipc_msg.app_id, ipc_msg.stage_id, ipc_msg.num_tasks_left)
                ipc_reply.msg = \
                    "external scheduler updated app_id " + \
                    str(ipc_msg.app_id) + \
                    " stage_id " + \
                    str(ipc_msg.stage_id) + \
                    " with " + str(ipc_msg.num_tasks_left) + " tasks left"

            elif ipc_msg.msg_type == 'update':
                frontier_nodes_changed = \
                    env.complete_stage(ipc_msg.app_id, ipc_msg.stage_id)

                ipc_reply.msg = \
                    "external scheduler updated app_id " + \
                    str(ipc_msg.app_id) + \
                    " stage_id " + \
                    str(ipc_msg.stage_id)

            elif ipc_msg.msg_type == 'tracking':
                # master asks which app it should assign the executor to
                ipc_reply.app_id, ipc_reply.num_executors_to_take = \
                    exec_tracker.pop_executor_flow(ipc_msg.num_available_executors)
                ipc_reply.msg = \
                    "external scheduler moves " + \
                    str(ipc_reply.num_executors_to_take) + \
                    " executor to app " + ipc_reply.app_id

            elif ipc_msg.msg_type == 'consult':

                # convert ipc_msg.app_id and ipc_msg.stage_id to corresponding
                # executors in virtual environment and then inovke the
                # scheduling agent

                # 1. put the latest observation in the obs queue
                obs_queue.put([env.job_dags, env.executors, executor])

                # 2. get the next action from the agent
                node = act_queue.get()
                if node is None:
                    # no-action was made
                    ipc_reply.app_id = 'void'
                    ipc_reply.stage_id = -1
                else:
                    ipc_reply.app_id, ipc_reply.stage_id = self.spark_inverse_node_map[node]
                    if node.idx not in node.job_dag.frontier_nodes:
                        # move (or stay) the executor to the job only
                        ipc_reply.stage_id = -1

                if ipc_msg.app_id != 'void' and \
                   ipc_reply.app_id != 'void' and \
                   ipc_msg.app_id != ipc_reply.app_id:
                    # executor needs to move to another job, keep track of it
                    exec_tracker.add_executor_flow(ipc_reply.app_id, 1)

                ipc_reply.msg = \
                    "external scheduler return app_id " + str(ipc_reply.app_id) + \
                    " stage_id " + str(ipc_reply.stage_id) + \
                    " for exec_id " + str(ipc_msg.exec_id)

            elif ipc_msg.msg_type == 'deregister':
                env.remove_job_dag(ipc_msg.app_id)
                dag_db.remove_app(ipc_msg.app_id)
                exec_tracker.remove_app(ipc_msg.app_id)
                ipc_reply.msg = \
                    "external scheduler deregister app " + ipc_msg.app_id

            print("time:", datetime.now())
            print("msg_type:", ipc_msg.msg_type)
            print("app_name:", ipc_msg.app_name)
            print("app_id:", ipc_msg.app_id)
            print("stage_id:", ipc_msg.stage_id)
            print("executor_id:", ipc_msg.exec_id)
            print("track_id:", ipc_msg.track_id)
            print("num_available_executors:", ipc_msg.num_available_executors)
            print("num_tasks_left", ipc_msg.num_tasks_left)
            print("reply_msg:", ipc_reply.msg)
            print("")
            sys.stdout.flush()

            socket.send(ipc_reply.SerializeToString())

    def shutdown(self):
        self.exit.set()


