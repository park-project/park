import park
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
        self._install_dependencies()
        # start calcite + java server
        logger.info("port = " + str(config.qopt_port))
        self._start_java_server()

        # initialize the ZeroMQ based communication system with the backend
        # Want to find a port number to talk on
        # port = get_port()
        context = zmq.Context()
        #  Socket to talk to server
        logger.info("Going to connect to calcite server")
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://localhost:" + str(config.qopt_port))
        self.reward_normalization = config.qopt_reward_normalization

        # just don't use clipping
        # self.reward_damping = config.qopt_reward_damping
        # self.clip_min_max = config.qopt_clip_min_max

        # TODO: describe spaces
        self.graph = None
        self.action_space = None
        # here, we specify the edge to choose next to calcite using the
        # position in the edge array
        # this will be updated every time we use _observe
        self._edge_pos_map = None

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

    def _install_dependencies(self):
        """
         - Install backend database (e.g., postgres) (if needed)
         - download + load dataset (imdb)
            - for postgres, can just use ryan's script
            - TODO: handling different databases properly
         - start database server
            - TODO: options for postgres data dir. Default to a directory in
              park's installation directory right now
         - clone OR pull latest version from query-optimizer github repo
            - TODO: auth?
            - set path appropriately to run java stuff

         FIXME:
            - instead of using queries from github, should download them
              separately. Ideally, can just use original JOB queries
            -
        """
        logger.info("installing dependencies for query optimizer")
        base_dir = park.__path__[0]

        # Check if postgres is installed:
        # which postgres

        # start postgres at given data directory

        # check if imdb db already exists:
        # psql -lqt | cut -d \| -f 1 | grep "imdb"


    def reset(self):
        query = self._send("reset")
        # TODO: set up min-max etc.
        self._observe()

        # return the first observation
        return self.graph

    def step(self, action):
        '''
        TODO: describe elements.
        '''
        assert self.action_space.contains(action)
        assert action in self._edge_pos_map
        action_index = self._edge_pos_map[action]

        # will make multiple zeromq calls to specify each part of the step
        # operation.
        self._send("step")
        self._send(str(action_index))
        # at this point, the calcite server would have specified the next edge
        # to be chosen in the query graph. Then, it will continue executing,
        # and block when the reward has been set

        # ask for reward, and updated observation
        self._observe()

        reward = float(self._send("getReward"))
        # TODO: add normalization
        # reward = self.normalize_reward(reward)
        done = int(self._send("isDone"))
        return self.graph, reward, done, None

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
        the action space accordingly.
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
            self._edge_pos_map = {}
            for i, e in enumerate(edges):
                graph.update_edges({tuple(e["factors"]) : e["joinAttributes"]})
                self._edge_pos_map[tuple(e["factors"])] = i
        else:
            assert False, "no other featurization scheme supported right now"
        assert self.observation_space.contains(graph)
        self.graph = graph

        # update the possible actions we have - should be at least one fewer
        # action since we just removed an edge
        self.action_space.update_graph(self.graph)

    def _setup_space(self):
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
