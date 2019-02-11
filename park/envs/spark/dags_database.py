import copy
import numpy as np
from sim.param import *
from sim.spark_env.wall_time import WallTime
from sim.spark_env.job_dag import merge_job_dags
from sim.spark_env.job_generator import JobLoader


class DAGsDatabase(object):
    """
    Map spark application id to known DAG
    """
    def __init__(self):
        # dummy wall_time
        self.wall_time = WallTime()
        # stores app_name -> job_dag (can be multiple DAGs)
        self.apps_store = {}
        # stores app_name -> {node.idx -> stage_id} map
        self.stage_store = {}

        # dynamically bind app_id -> job_dag
        self.apps_map = {}
        # dynamically bind app_id -> {node.idx -> stage_id}
        self.stage_map = {}
        # dynamically bind app_id -> app_name (query)
        self.queries_map = {}

        # initialize dags_store
        for query_size in args.tpch_size:
            for query_idx in xrange(1, args.tpch_num + 1):

                query = args.tpch_prefix + query_size + '-' + str(query_idx)

                job_dag = JobLoader(args.tpch_folder, query, self.wall_time)

                self.apps_store[args.tpch_prefix + query_size + \
                    '-' + str(query_idx)] = job_dag

                # load stage_id -> node_idx map
                stage_id_to_node_idx_map = \
                    np.load(args.tpch_folder + query_size + '/' + \
                        'stage_id_to_node_idx_map_' + \
                        str(query_idx) + '.npy').item()

                # build up the new map based on merged job_dag
                node_idx_to_stage_id_map = \
                    {v: k for k, v in stage_id_to_node_idx_map.iteritems()}

                # store the {node.idx -> stage_id} map
                self.stage_store[
                    args.tpch_prefix + query_size + '-' + str(query_idx)] = \
                    node_idx_to_stage_id_map

    def add_new_app(self, app_name, app_id):
        job_dag = None
        stage_map = None

        if app_name in self.apps_store:
            job_dag = copy.deepcopy(self.apps_store[app_name])
            stage_map = self.stage_store[app_name]

        self.apps_map[app_id] = job_dag
        self.stage_map[app_id] = stage_map
        self.queries_map[app_id] = app_name

    def remove_app(self, app_id):
        if app_id in self.apps_map:
            del self.apps_map[app_id]
            del self.stage_map[app_id]
            del self.queries_map[app_id]
