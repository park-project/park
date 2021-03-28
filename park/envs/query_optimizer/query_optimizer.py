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
import numpy as np
import json
import psutil
import pdb
import signal
import glob
from sklearn.model_selection import train_test_split
import wget
import psycopg2
# from utils.utils import *
# import utils
from .qopt_utils import *
from .pg_cost import *
import getpass
import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import pickle
# from collections import defaultdict
# try:
    # multiprocessing.set_start_method('forkserver')
# except:
    # pass

class QueryOptError(Exception):
    pass

class QueryOptEnv(core.Env):
    """
    TODO: describe state, including the features for nodes and edges.
    """
    def __init__(self):
        print("in init!")
        self.base_dir = None    # will be set by _install_dependencies
        # start calcite + java server
        self.use_java_backend = config.qopt_use_java
        self.opt_cache_fn = "/tmp/opt_cache.pkl"
        if os.path.isfile(self.opt_cache_fn):
            with open(self.opt_cache_fn, 'rb') as handle:
                self.opt_cache = pickle.load(handle)
        else:
            self.opt_cache = {}
            self.opt_cache[0] = {}
            self.opt_cache[1] = {}
            self.opt_cache[0]["costs"] = {}
            self.opt_cache[0]["explains"] = {}
            self.opt_cache[0]["sqls"] = {}

            self.opt_cache[1]["costs"] = {}
            self.opt_cache[1]["explains"] = {}
            self.opt_cache[1]["sqls"] = {}

        if self.use_java_backend:
            # original port:
            self.port = find_available_port(config.qopt_port)
            # this is only needed in the java backend case
            signal.signal(signal.SIGINT, self._handle_exit_signal)
            signal.signal(signal.SIGTERM, self._handle_exit_signal)
            self._start_java_server()
            self._install_dependencies()
            print("start java server succeeded")


            nIOthreads = 2                          # ____POLICY: set 2+: { 0: non-blocking, 1: blocking, 2: ...,  }
            context = zmq.Context(nIOthreads)      # ____POLICY: set several IO-datapumps

            self.socket  = context.socket(zmq.PAIR)
            self.socket.setsockopt( zmq.LINGER,      0 )  # ____POLICY: set upon instantiations
            self.socket.setsockopt( zmq.AFFINITY,    1 )  # ____POLICY: map upon IO-type thread
            self.socket.setsockopt(zmq.RCVTIMEO, 6000000)
            self.socket.connect("tcp://localhost:" + str(self.port))
            print("self.socket.connect succeeded")
            self.reward_normalization = config.qopt_reward_normalization

            # self.poller = zmq.Poller()
            # self.poller.register(self.socket, zmq.POLLIN) # POLLIN for recv

            # TODO: describe spaces
            self.graph = None
            self.action_space = None
            # here, we specify the edge to choose next to calcite using the
            # position in the edge array
            # this will be updated every time we use _observe
            self._edge_pos_map = None

            # will store _min_reward / _max_reward for each unique query
            # will map query: (_min_reward, _max_reward)
            self.reward_mapper = {}
            # these values will get updated in reset.
            self._min_reward = None
            self._max_reward = None

            # self.query_set = self._send("getCurQuerySet")
            self.attr_count = int(self._send("getAttrCount"))
            print("attr count: ", self.attr_count)

            self.current_query = None

            # setup space with the new graph
            self._setup_space()
            print("setup space done!")

            # more experimental stuff

            # original graph used to denote state
            # self.orig_graph = None
            # if config.qopt_viz:
                # self.viz_ep = 0
                # self.viz_output_dir = "./visualization/"
                # self.viz_pdf = PdfPages(self.viz_output_dir + "test.pdf")

            self.queries_initialized = False

    def _compute_join_order_loss_pg(self, sqls, true_cardinalities,
            est_cardinalities, num_processes, use_indexes, pool):

        est_costs = []
        opt_costs = []
        est_explains = []
        opt_explains = []
        est_sqls = []
        opt_sqls = []

        if use_indexes:
            use_indexes = 1
        else:
            use_indexes = 0
        opt_costs_cache = self.opt_cache[use_indexes]["costs"]
        opt_sqls_cache = self.opt_cache[use_indexes]["sqls"]
        opt_explains_cache = self.opt_cache[use_indexes]["explains"]

        par_args = []
        for i, sql in enumerate(sqls):
            sql_key = deterministic_hash(sql)
            if sql_key in opt_costs_cache:
                # already know for the true cardinalities case
                par_args.append((sql, true_cardinalities[i],
                        est_cardinalities[i], opt_costs_cache[sql_key],
                        opt_explains_cache[sql_key], opt_sqls_cache[sql_key],
                        use_indexes))
            else:
                par_args.append((sql, true_cardinalities[i],
                        est_cardinalities[i], None,
                        None, None, use_indexes))

        if pool is None:
            num_processes = max(1, num_processes)
            with Pool(processes=num_processes) as pool:
                costs = pool.starmap(compute_join_order_loss_pg_single, par_args)
        else:
            costs = pool.starmap(compute_join_order_loss_pg_single, par_args)

        # single threaded case for debugging
        # costs = []
        # for i, sql in enumerate(sqls):
            # costs.append(compute_join_order_loss_pg_single(sql,
                # true_cardinalities[i], est_cardinalities[i],
                # None, None, None, use_indexes))

        new_seen = False
        for i, (est, opt, est_explain, opt_explain, est_sql, opt_sql) \
                    in enumerate(costs):
            sql_key = deterministic_hash(sqls[i])
            est_costs.append(est)
            opt_costs.append(opt)
            est_explains.append(est_explain)
            opt_explains.append(opt_explain)
            est_sqls.append(est_sql)
            opt_sqls.append(opt_sql)

            if sql_key not in opt_costs_cache:
                opt_costs_cache[sql_key] = opt
                opt_explains_cache[sql_key] = opt_explain
                opt_sqls_cache[sql_key] = opt_sql
                new_seen = True

        if new_seen:
            # FIXME: DRY
            with open(self.opt_cache_fn, 'wb') as handle:
                pickle.dump(self.opt_cache, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
            # with open(self.opt_costs_fn, 'wb') as handle:
                # pickle.dump(self.opt_costs, handle,
                        # protocol=pickle.HIGHEST_PROTOCOL)
            # with open(self.opt_explains_fn, 'wb') as handle:
                # pickle.dump(self.opt_explains, handle,
                        # protocol=pickle.HIGHEST_PROTOCOL)
            # with open(self.opt_sqls_fn, 'wb') as handle:
                # pickle.dump(self.opt_sqls, handle,
                        # protocol=pickle.HIGHEST_PROTOCOL)

        return np.array(est_costs), np.array(opt_costs), est_explains, \
    opt_explains, est_sqls, opt_sqls

    def compute_join_order_loss(self, sqls, true_cardinalities,
            est_cardinalities, baseline_join_alg, use_indexes,
            num_processes=8, postgres=True, pool=None):
        '''
        @query_dict: [sqls]
        @true_cardinalities / est_cardinalities: [{}]
                dictionary, specifying cardinality of each subquery
                key: sorted list of [table_1_key, table_2_key, ...table_n_key]
                val: cardinality (double)
                In order to handle aliases (this information is lost when
                processing in calcite), each table_key is table name +
                first predicate (if no predicate present on that table, then just "")
            FIXME: this does not handle many edge cases, and we need a better
            way to deal with aliases.

        @ret:
            TODO
        '''
        start = time.time()
        assert isinstance(sqls, list)
        assert isinstance(true_cardinalities, list)
        assert isinstance(est_cardinalities, list)
        assert len(sqls) == len(true_cardinalities) == len(est_cardinalities)

        if postgres:
            return self._compute_join_order_loss_pg(sqls,
                    true_cardinalities, est_cardinalities, num_processes,
                    use_indexes, pool)
        else:
            assert False

        self.initialize_queries(query_dict)
        self.initialize_cardinalities(est_cardinalities)
        self._send("startTestCardinalities")
        # java will compute optimal orderings for the estimated cardinalities

        self.initialize_cardinalities(true_cardinalities)
        est_costs = json.loads(self._send("getEstCardinalityCosts"))
        opt_costs = json.loads(self._send("getOptCardinalityCosts"))

        # sanity check
        for k in query_dict:
            try:
                assert k in est_costs
                assert k in opt_costs
            except:
                print("key not found in costs returned from java")
                print(k)
                pdb.set_trace()

        return est_costs, opt_costs, None, None

    def initialize_cardinalities(self, cardinalities):
        '''
        @cardinalities: dict
            key: query name
            val: dict
                key: [table names] (sorted alphabetically)
                val: cardinality (after the relevant predicates in the given
                                  query are applied)
        '''
        # TODO: add error checking to ensure these cover the queries
        self._send("setCardinalities")
        self._send(json.dumps(cardinalities))

    def initialize_queries(self, queries, mode="train"):
        '''
        @queries: dict
            key : query name
            val: sql query
        TODO: need to also provide the user option to specify the DB they want
        to connect to along with the queries.
        '''
        self._send("setQueries")
        self._send(mode)
        self._send(json.dumps(queries))
        self.queries_initialized = True

    def _initialize_default_queries(self):
        '''
        loads the queries from the join order benchmark, and initializes them.
        '''
        self.base_dir = park.__path__[0]
        # if JOB doesn't exist, download it
        job_dir = self.base_dir + "/join-order-benchmark"
        if not os.path.exists(job_dir):
            # now need to install the join-order-benchmark as well
            JOB_REPO = "https://github.com/gregrahn/join-order-benchmark.git"
            cmd = "git clone " + JOB_REPO
            p = sp.Popen(cmd, shell=True, cwd=self.base_dir)
            p.wait()
            print("downloaded join order benchmark queries")

        # read in queries and initialize
        all_queries = []
        fns = sorted(glob.iglob(job_dir + "/*.sql"))
        for fn in fns:
            if "fk" in fn or "schem" in fn:
                continue
            with open(fn, "r") as f:
                # FIXME: this is just required because the cardinalities json
                # files had keys like this. Can change those, so only query'
                # file name is the key
                qname = "join-order-benchmark/" + os.path.basename(fn)
                all_queries.append((qname, f.read()))

        # by splitting first based on random_state, we ensure that if only a
        # subset of the queries are being run (using the qopt_query argument),
        # then the train queries / test queries would still be the same
        train , test = train_test_split(all_queries,test_size=config.qopt_test_size,
                            random_state=config.qopt_test_seed)
        valid_queries = config.qopt_query.split(",")
        trainq = {}
        testq = {}
        SKIP_LIST = []
        for t in train:
            if t[0] in SKIP_LIST:
                continue
            if len(valid_queries) > 0:
                for vq in valid_queries:
                    if vq in t[0]:
                        trainq[t[0]] = t[1]
            else:
                trainq[t[0]] = t[1]

        for t in test:
            if t[0] in SKIP_LIST:
                continue
            if len(valid_queries) > 0:
                for vq in valid_queries:
                    if vq in t[0]:
                        testq[t[0]] = t[1]
            else:
                testq[t[0]] = t[1]

        if len(testq) == 0:
            # for debugging
            testq = trainq

        self.initialize_queries(trainq, mode="train")
        self.initialize_queries(testq, mode="test")

    def _install_if_needed(self, name):
        '''
        checks if the program called `name` is installed on the local system,
        and prints an error and cleans up if it isn't.
        '''
        # Check if docker is installed.
        cmd_string = "which {}".format(name)
        # which_pg_output = sp.check_output(cmd_string.split())
        FNULL = open(config.qopt_log_file, 'w')
        process = sp.Popen(cmd_string.split(), shell=False, stdout=FNULL,
                stderr=FNULL)
        FNULL.close()
        ret_code = process.wait()
        if ret_code != 0:
            # TODO: different installations based on OS, so this should be
            # user's responsibility
            print("{} is not installed. Please install docker before \
                    proceeding".format(name))
            self.clean()

    def _install_dependencies(self):
        """
         - clone OR pull latest version from query-optimizer github repo
            - TODO: auth?
            - set path appropriately to run java stuff
         - use docker to install, and start postgres
        """
        # logger.info("installing dependencies for query optimizer")
        self.base_dir = park.__path__[0]
        self._install_if_needed("docker")
        self._install_if_needed("mvn")
        self._install_if_needed("java")

        # set up the query_optimizer repo
        try:
            qopt_path = os.environ["QUERY_OPT_PATH"]
        except:
            # if it has not been set, then set it based on the base dir
            qopt_path = self.base_dir + "/query-optimizer"
            # if this doesn't exist, then git clone this
            if not os.path.exists(qopt_path):
                print("going to clone query-optimizer library")
                cmd = "git clone https://github.com/parimarjan/query-optimizer.git"
                p = sp.Popen(cmd, shell=True,
                    cwd=self.base_dir)
                p.wait()
                print("cloned query-optimizer library")
        # print("query optimizer path is: ", qopt_path)

        # TODO: if psql -d imdb already set up locally, then do not use docker
        # to set up postgres. Is this really useful, or should we just assume
        # docker is always the way to go?

        conn_failed = False
        try:
            os_user = getpass.getuser()
            conn = psycopg2.connect(host="localhost",port=5432,dbname="imdb",
                    user="root",password="password")
            # if os_user == "ubuntu":
                # conn = psycopg2.connect(port=5432,dbname="imdb",
                        # user=os_user,password="")
            # else:
                # conn = psycopg2.connect(host="localhost",port=5432,dbname="imdb",
                        # user="pari",password="")
        except Exception as e:
            import traceback
            print("caught exception!")
            print(e)
            traceback.print_exc(e)
            conn_failed = True
            pdb.set_trace()
        if not conn_failed:
            return

        # TODO: print plenty of warning messages: going to start docker,
        # docker's directory should have enough space - /var/lib/docker OR
        # change it manually following instructions at

        docker_dir = qopt_path + "/docker"
        docker_img_name = "pg"
        container_name = "docker-pg"
        # docker build
        docker_bld = "sudo docker build -t {} . ".format(docker_img_name)
        p = sp.Popen(docker_bld, shell=True, cwd=docker_dir)
        p.wait()
        print("building docker image {} successful".format(docker_img_name))
        time.sleep(2)
        # start / or create new docker container
        # Note: we need to start docker in a privileged mode so we can clear
        # cache later on.
        local_port = 5432
        docker_run = "sudo docker run --name {} -p \
        {}:5432 --privileged -d {}".format(container_name, local_port,docker_img_name)
        docker_start_cmd = "sudo docker start docker-pg || " + docker_run
        p = sp.Popen(docker_start_cmd, shell=True, cwd=docker_dir)
        p.wait()
        print("starting docker container {} successful".format(container_name))
        time.sleep(2)

        check_container_cmd = "sudo docker ps | grep {}".format(container_name)
        process = sp.Popen(check_container_cmd, shell=True)
        ret_code = process.wait()
        if ret_code != 0:
            print("something bad happened when we tried to start docker container")
            print("got ret code: ", ret_code)
            self.clean()

        time.sleep(2)
        # need to ensure that we psql has started in the container. If this is
        # the first time it is starting, then pg_restore could take a while.
        while True:
            try:
                conn = psycopg2.connect(host="localhost",port=local_port,dbname="imdb",
        user="imdb",password="imdb")
                conn.close();
                break
            except psycopg2.OperationalError as ex:
                print("Connection failed: {0}".format(ex))
                print("""If this is the first time you are starting the
                        container, then pg_restore is probably taking its time.
                        Be patient. Will keep checking for psql to be alive.
                        Can take upto few minutes.
                        """)
                time.sleep(30)

        self.container_name = container_name

    def reset(self):
        '''
        '''
        if not self.queries_initialized:
            self._initialize_default_queries()

        self._send("reset")
        query = self._send("curQuery")
        if self.reward_normalization == "min_max":
            if query in self.reward_mapper:
                self._min_reward = self.reward_mapper[query][0]
                self._max_reward = self.reward_mapper[query][1]
            else:
                # FIXME: dumb hack so self.step does right thing when executing
                # random episode.
                self._min_reward = None
                self._max_reward = None
                self._min_reward, self._max_reward = self._run_random_episode()
                self.reward_mapper[query] = (self._min_reward, self._max_reward)
                # FIXME: dumb hack
                return self.reset()

        self.current_query = query
        self._observe()
        if config.qopt_viz:
            self.orig_graph = self.graph

        # clear cache if needed
        return self.graph

    def step(self, action):
        '''
        @action: edge as represented in networkX e.g., (v1,v2)
        '''
        # also, consider the reverse action (v1,v2) or (v2,v1) should mean the
        # same
        rev_action = (action[1], action[0])
        assert self.action_space.contains(action) or \
                    self.action_space.contains(rev_action)
        assert action in self._edge_pos_map or rev_action \
                in self._edge_pos_map
        if rev_action in self._edge_pos_map:
            action = rev_action

        action_index = self._edge_pos_map[action]

        # will make multiple zeromq calls to specify each part of the step
        # operation.
        self._send("step")
        self._send(str(action_index))
        # at this point, the calcite server would have specified the next edge
        # to be chosen in the query graph. Then, it will continue executing,
        # and block when the reward has been set

        self._observe()
        orig_reward = float(self._send("getReward"))
        reward = self._normalize_reward(orig_reward)
        if self.reward_normalization == "min_max":
            if not (self._min_reward is None or self._max_reward is None):
                stored_rewards = self.reward_mapper[self.current_query]
                if orig_reward < self._min_reward:
                    self.reward_mapper[self.current_query] = (orig_reward,
                            stored_rewards[1])
                elif orig_reward > self._max_reward:
                    self.reward_mapper[self.current_query] = (stored_rewards[0],
                                                    orig_reward)

        done = int(self._send("isDone"))
        info = None

        # print("action: {}, reward: {}, done: {}, info: {}".format(action,
            # reward, done, info))
        # pdb.set_trace()
        if done:
            # print("DONE!!!")
            info = self._send("getQueryInfo")
            info = json.loads(info)
            # print(info["queryName"])
            # print(info["costs"])
            # min_cost = min([v for k,v in info["costs"].items()])
            # for k,v in info["costs"].items():
                # print(k, v, (v / min_cost))
            # pdb.set_trace()
            # output episode based plots / viz
            if config.qopt_viz:
                plot_join_order(info, self.viz_pdf, self.viz_ep)
                self.viz_ep += 1

        # TODO: need to make this more intuitive
        if config.qopt_final_reward:
            # so that this is done ONLY sometimes, i.e., if we set park to
            # noExec mode, then there will be no final reward
            if done:
                if "RL" in info["dbmsAllRuntimes"]:
                    reward = -info["dbmsAllRuntimes"]["RL"][-1]

            # give no intermediate reward ONLY if final reward is ON.
            elif config.qopt_no_intermediate_reward:
                reward = 0.00
        return self.graph, reward, done, info

    def seed(self, seed):
        print("seed! not implemented yet")

    def _handle_exit_signal(self, signum, frame):
        self.clean()
        raise QueryOptError('From park, received signal ' + str(signum))

    def clean(self):
        '''
        kills the java server started by subprocess, and all it's children.
        '''
        if config.qopt_viz:
            self.viz_pdf.close()
        if self.use_java_backend:
            os.killpg(os.getpgid(self.java_process.pid), signal.SIGTERM)
            print("killed the java server")

    def set(self, attr, val):
        '''
        TODO: explain and reuse
        '''
        if attr == "execOnDB":
            if val:
                self._send("execOnDB")
            else:
                self._send("noExecOnDB")
        else:
            assert False

    def get_optimized_plans(self, name):
        # ignore response
        self._send(b"getOptPlan")
        resp = self._send(name)
        return resp

    def get_optimized_costs(self, name):
        self._send("getJoinsCost")
        resp = float(self._send(name))
        return resp

    def get_current_query(self):
        return self.current_query

    def get_current_query_name(self):
        return self._send("getCurrentQueryName")

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
            # print("vertexes")
            for v in vertexes:
                # print(v["id"], v["visibleAttributes"])
                graph.update_nodes({v["id"] : v["visibleAttributes"]})
            self._edge_pos_map = {}
            # print("edges: ")
            for i, e in enumerate(edges):
                # print(e["factors"], e["joinAttributes"])
                graph.update_edges({tuple(e["factors"]) : e["joinAttributes"]})
                self._edge_pos_map[tuple(e["factors"])] = i
        else:
            assert False, "no other featurization scheme supported right now"
        # assert self.observation_space.contains(graph)
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
        print("_start_java_server")
        JAVA_EXEC_FORMAT = 'mvn -e exec:java -Dexec.mainClass=Main \
        -Dexec.args="-port {port} -train {train} \
        -lopt {lopt} -exhaustive {exh} -leftDeep {ld} -python 1 \
        -verbose {verbose} -costModel {cm} -dataset {ds} \
        -execOnDB {execOnDB} -clearCache {clearCache} \
        -recomputeFixedPlanners {recompute} -numExecutionReps {reps} \
        -maxExecutionTime {max_exec} -useIndexNestedLJ {nlj} \
        -scanCostFactor {scanCostFactor} -getSqlToExecute {getSql} \
        -testCardinalities {testCardinalities}"'
        # FIXME: setting the java directory relative to the directory we are
        # executing it from?
        # if config.qopt_test_cardinalities:
            # print("WARNING: test cardinalities mode on")

        cmd = JAVA_EXEC_FORMAT.format(
                query = config.qopt_query,
                port  = str(self.port),
                train = config.qopt_train,
                lopt = config.qopt_lopt,
                exh = config.qopt_exh,
                ld = config.qopt_left_deep,
                verbose = config.qopt_verbose,
                cm = config.qopt_cost_model,
                ds = config.qopt_dataset,
                execOnDB = config.qopt_final_reward,
                clearCache = config.qopt_clear_cache,
                reps       = config.qopt_num_execution_reps,
                max_exec   = config.qopt_max_execution_time,
                recompute = config.qopt_recompute_fixed_planners,
                nlj       = config.qopt_use_index_nested_lj,
                scanCostFactor = config.qopt_scan_cost_factor,
                getSql = config.qopt_get_sql,
                testCardinalities = config.qopt_test_cardinalities)
        try:
            qopt_path = os.environ["QUERY_OPT_PATH"]
        except:
            # if it has not been set, then set it based on the base dir
            qopt_path = self.base_dir + "/query-optimizer"

        # FIXME: hardcoded cwd, shell=False.
        # Important to use preexec_fn=os.setsid, as this puts the java process
        # and all it's children into a new groupid, which can be killed in
        # clean without shutting down the current python process

        # FIXME: update this.
        # if not os.path.exists(qopt_path + "/pg.json"):
            # wget.download(
                # url="https://parimarjan.github.io/dbs/pg.json",
                # out=qopt_path + "/pg.json"
            # )

        # FIXME: always assume it is compiled
        # if not config.qopt_java_output:
            # FNULL = open(config.qopt_log_file, 'w')
            # compile_pr = sp.Popen("mvn package", shell=True,
                    # cwd=qopt_path, stdout=FNULL, stderr=FNULL,
                    # preexec_fn=os.setsid)
            # FNULL.close()
        # else:
            # compile_pr = sp.Popen("mvn package", shell=True,
                    # cwd=qopt_path,
                    # preexec_fn=os.setsid)
        # compile_pr.wait()

        if not config.qopt_java_output:
            FNULL = open(config.qopt_log_file, 'w')
            self.java_process = sp.Popen(cmd, shell=True,
                    cwd=qopt_path, stdout=FNULL, stderr=FNULL,
                    preexec_fn=os.setsid)
        else:
            self.java_process = sp.Popen(cmd, shell=True,
                    cwd=qopt_path, preexec_fn=os.setsid)

        print("started java server!!!")
        # FIXME: prob not required
        time.sleep(0.5)

    def _send(self, msg):
        """
        """
        start = time.time()
        self.socket.send_string(msg)
        # ret = self.socket.recv()
        while True:
            try:
                ret = self.socket.recv()
                if ret is not None:
                    break
            except zmq.error.Again as e:
                # print(e)
                # print("waited forever for response from java")
                # pdb.set_trace()
                raise QueryOptError("fucking zmq error: Again")

        # finally:
            # self.socket.close()
            # self.context.term()

        # while True:
            # print("going to try again")
            # if time.time() - start > 2.00:
                # print("no reply for over 2 seconds")
                # pdb.set_trace()
            # ret = self.socket.recv(flags = zmq.NOBLOCK)
            # if ret is not None:
                # break
            # else:
                # time.sleep(0.00001)

            # evts = self.poller.poll(10000) # wait *up to* one second for a message to
            # if len(evts) > 0:
                # print("msg arrivwd. receiving")
                # ret = self.socket.recv(zmq.NOBLOCK)
            # else:
                # print("error: message timeout")
                # # retry?
                # print(msg)
                # pdb.set_trace()

        ret = ret.decode("utf8")
        return ret

    def _run_random_episode(self):
        '''
        runs a random episode, and returns the minimum / and maximum reward
        seen during this episode.
        '''
        done = False
        min_reward = 10000000
        max_reward = -10000000
        self._observe()
        while not done:
            act = self.action_space.sample()
            _, reward, done, _ = self.step(act)
            if reward < min_reward:
                min_reward = reward
            if reward > max_reward:
                max_reward = reward
        return min_reward, max_reward

    def _normalize_reward(self, reward):
        '''
        '''
        # first check if it is one of the cases in which we should just return
        # the reward without any normalization
        if self.reward_normalization == "min_max":
            if self._min_reward is None or self._max_reward is None:
                return reward
            reward = (reward-self._min_reward) / \
                    float((self._max_reward-self._min_reward))
            reward = max(0, reward)
            # reward = np.interp(reward, [0,1])
            # print("reward after normalization: ", reward)

        elif self.reward_normalization == "scale_down":
            reward = reward / 10e30

        elif self.reward_normalization == "":
            return reward
        else:
            assert False

        return reward

