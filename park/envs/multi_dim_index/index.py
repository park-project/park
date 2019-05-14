from park.core import Env, Space
from config import Action, ActionSpace, DataObs, QueryObs, Query
from gen_osm_queries import QueryGen
import random
from park.spaces.tuple_space import Tuple
import params
import spaces
import config
import wget
import subprocess
from timeit import default_timer as timer


class MultiDimIndexEnv(Env):
    metadata = {'env.name': 'multi_dim_index'}
    # Rewards are reported as throughput (queries per second)
    reward_range = (0, 1e6)
    action_space = ActionSpace()
    observation_space = Tuple(DataObsSpace, QueryObsSpace)

    def __init__(self):
        datafile = params.DATASET_PATH
        if not os.path.exists(datafile):
            print('Downloading dataset...')
            wget.download(params.DATA_DOWNLOAD_URL, out=datafile)
        binary = params.BINARY_PATH
        if not os.path.exists(binary):
            print('Downloading binary...')
            wget.download(params.BINARY_DOWNLOAD_URL, out=binary)
        
        self.step_count = 0

        sys.stdout.write('Initializing OSM Query generator...')
        start = timer()
        self.query_generator = QueryGen(datafile)
        end = timer()
        print(end-start, 's')

    def parse_cmd_output(self, output):
        lines = output.split('\n')
        times = []
        for line in lines:
            if line.startswith('Query'):
                time = int(line.split(':')[1])
                times.append(time)
        return times

    def step(self, action):
        assert self.action_space.contains(action)
        layout_filename = "mdi_layout.dat"
        action.to_file(layout_filename)
        
        sys.stdout.write('Generating next query workload...')
        start = timer()
        new_queries = []
        for _ in range(params.QUERIES_PER_STEP):
            q = self.query_generator.random_query()
            new_queries.append(q)
        query_filename = 'queries.bin'
        np.array(new_queries).tofile(query_filename)
        end = timer()
        print(end-start, 's')

        sys.stdout.write('Running range query workload...')
        start = timer()
        cmd = "%s --dataset=%s --workload=%s --projector=%s" \
                (params.BINARY_PATH, params.DATASET_PATH, query_filename,
                        layout_filename)
        outfile = 'cmd_output.txt'
        done = subprocess.run(cmd, capture_output=True, encoding='utf-8')
        if done.returncode != 0:
            raise Exception('Query binary did not finish successfully')
        times = []
        with open(outfile, 'w') as out:
            out.write(done.stdout)
            times = parse_cmd_output(done.stdout)
        if len(times) != len(new_queries):
            raise Exception('Results from binary are incomplete')
        end = timer()
        print(end-start, 's')

        reward = np.mean(times)
        obs = QueryObs(new_queries)
        self.step_count += 1

        # The query times are given as information.
        return obs, reward, self.step_count >= params.STEPS_PER_EPOCH, {"times": times}

    def reset(self):
        # Restart the query generator with a new random configuration.
        self.query_generator = QueryGen(datafile)

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed + 5)



