from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.utils.directed_graph import DirectedGraph
import subprocess as sp
import time
import zmq
import os
import signal
import json

class QueryOptEnv(core.Env):
    """
    TODO: describe
    """
    def __init__(self):
        print("Query Opt Env being initialized!!!")
        # FIXME: do we need to even call this here?
        # self._setup_space()
        # self.graph = DirectedGraph()

        # start calcite + java server
        print("port = ", config.qopt_port)
        self._start_java_server()

        # initialize the ZeroMQ based communication system with the backend
        # Want to find a port number to talk on
        # port = get_port()
        context = zmq.Context()
        #  Socket to talk to server
        print("Going to connect to calcite server")
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://localhost:" + str(config.qopt_port))
        self.reward_normalization = config.qopt_reward_normalization

        # just don't use clipping
        # self.reward_damping = config.qopt_reward_damping
        # self.clip_min_max = config.qopt_clip_min_max

        # TODO: describe spaces
        self.graph = None
        self.action_space = None

        self.query_set = self._send("getCurQuerySet")
        print("query set: " + self.query_set)
        self.attr_count = int(self._send("getAttrCount"))
        print("attr count: ", self.attr_count)

        # FIXME: these variable don't neccessarily belong here / should be
        # cleaned up
        # TODO: figure this out using the protocol too. Or set it on the java
        # side using some protocol.
        self.only_final_reward = config.qopt_only_final_reward

        # will store min_reward / max_reward for each unique query
        # will map query: (min_reward, max_reward)
        self.reward_mapper = {}
        # these values will get updated in reset.
        self.min_reward = None
        self.max_reward = None

        # setup space with the new graph
        self._setup_space()

    def reset(self):
        print("reset")
        query = self._send("reset")
        # TODO: set up min-max etc.
        print("query is: ", query)
        self.graph = self._observe()
        self.action_space.update_graph(self.graph)

        # return the first observation
        return self.graph

    def step(self, action):
        '''
        TODO: describe elements.
        '''
        print("step!")
        print("action: ", action)
        assert self.action_space.contains(action)

        return None, None, None, None

    def seed(self, seed):
        print("seed! not implemented yet")

    def clean(self):
        # FIXME: cleaner way to do this
        os.killpg(os.getpid(), signal.SIGTERM)
        print("killed the java server")

    def _observe(self):
        '''
        TODO: more details
        gets the current query graph from calcite, and updates self.graph and
        the spaces accordingly.
        '''
        # get first observation
        vertexes = self._send("getQueryGraph")
        # FIXME: hack
        vertexes = vertexes.replace("null,", "")
        vertexes = eval(vertexes)
        edges = self._send("")
        edges = eval(edges)

        graph = DirectedGraph()
        # now, let's fill it up
        nodes = {}
        # TODO: adding other attributes to the featurization scheme.

        if config.qopt_only_attr_features:
            for v in vertexes:
                graph.update_nodes({v["id"] : v["visibleAttributes"]})

            for e in edges:
                graph.update_edges({tuple(e["factors"]) : e["joinAttributes"]})
        else:
            assert False, "no other featurization scheme supported right now"
        assert self.observation_space.contains(graph)
        return graph

    def _setup_space(self):
        print('setup space!')
        # TODO: set the features
        node_feature_space = spaces.PowerSet(set(range(self.attr_count)))
        edge_feature_space = spaces.PowerSet(set(range(self.attr_count)))
        self.observation_space = spaces.Graph(node_feature_space,
                edge_feature_space)
        # we will be updating the action space as the graph evolves
        self.action_space = spaces.EdgeInGraph()

    def _start_java_server(self):
        JAVA_EXEC_FORMAT = 'mvn -e exec:java -Dexec.mainClass=Main \
        -Dexec.args="-query {query} -port {port} -train {train} -onlyFinalReward \
        {final_reward} -lopt {lopt} -exhaustive {exh} -leftDeep {ld} -python 1 \
        -verbose {verbose} -costModel {cm} -executeOnDB {execute} -dataset {ds}"'
        # FIXME: setting the java directory relative to the directory we are
        # executing it from?
        cmd = JAVA_EXEC_FORMAT.format(
                query = config.qopt_query, port =
                str(config.qopt_port),
                train=config.qopt_train,
                final_reward=config.qopt_only_final_reward,
                lopt=config.qopt_lopt,
                exh=config.qopt_exh, ld = config.qopt_left_deep,
                verbose=config.qopt_verbose,
                cm = config.qopt_cost_model, execute = config.qopt_execute_on_db, ds =
                config.qopt_dataset)
        print("cmd is: ", cmd)
        # FIXME: hardcoded cwd, shell=False.
        self.java_process = sp.Popen(cmd, shell=True, cwd="/home/pari/query-optimizer/")
        print("started java server!")
        # FIXME: prob not required
        time.sleep(5)

    def _send(self, msg):
        """
        """
        self.socket.send_string(msg)
        ret = self.socket.recv()
        ret = ret.decode("utf8")
        return ret
