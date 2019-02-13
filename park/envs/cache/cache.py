import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.envs.cache.trace_loader import load_traces


class TraceSrc(object):
    def __init__(self, trace, cache_size):
        # load trace into dataframe
        self.trace = load_traces(trace, cache_size)
        self.n_request = len(self.trace)
        self.cache_size = cache_size
        self.min_values = self.trace.min(axis=0)
        self.max_values = self.trace.max(axis=0)
        # min cache remaining size == 0
        self.min_values[3] = 0
        # max cost == rm all from cache + add all cache size = 2*cache size
        self.max_values[4] = max(self.trace[3])*2
        # max last visit time == max time
        self.max_values[5] = max(self.trace[0])
        # rm limitation of req time and id, not relevant
        del self.min_values[0]
        del self.min_values[1]
        del self.max_values[0]
        del self.max_values[1]
        self.req = 0
    
    def reset(self):
        self.idx = 0 if self.n_request == len(self.trace) \
                     else np.random.randint(low=0, high=len(self.trace.index)-self.n_request-2)
        self.req = 0

    def step(self):    
        obs = self.trace.iloc[self.idx].values
        self.idx += 1
        self.req += 1
        done = self.req >= self.n_request
        return obs, done


class CacheSim(object) :
    def __init__(self, n_request, cache_size, action_space, observation_space):
        # invariant
        self.n_request  = n_request
        self.cache_size = cache_size
        self.action_space = action_space
        self.observation_space = observation_space
        # vary every step
        self.obj              = [[]] * self.n_request
        self.req              = 0
        self.actions          = np.zeros(self.n_request)
        self.cache            = defaultdict(list) # requested items with caching
        self.non_cache        = defaultdict(list) # requested items without caching
        self.cache_remain     = np.zeros(self.n_request)
        self.cache_remain_prev= np.zeros(self.n_request)
        self.last_req_time    = np.zeros(self.n_request)
        self.count_ohr        = np.zeros(self.n_request)
        self.ohr              = np.zeros(self.n_request)
        self.count_bhr        = np.zeros(self.n_request)
        self.size_all         = np.zeros(self.n_request)
        self.bhr              = np.zeros(self.n_request)
        self.cost             = np.zeros(self.n_request)
    
    def reset(self):
        self.obj = [[]] * self.n_request
        self.req = 0
        self.actions.fill(0)
        self.cache = defaultdict(list)
        self.non_cache = defaultdict(list)
        self.cache_remain.fill(self.cache_size)
        self.cache_remain_prev.fill(self.cache_size)
        self.last_req_time.fill(0)
        self.count_ohr.fill(0)
        self.ohr.fill(0)
        self.count_bhr.fill(0)
        self.bhr.fill(0)
        self.size_all.fill(0)
        self.cost.fill(0)
    
    def step(self, action, obj):
        # use previous value to initialize
        self.cache_remain_prev[self.req] = self.cache_size if self.req == 0 \
            else self.cache_remain[self.req-1]
        self.cache_remain[self.req] = self.cache_size if self.req == 0 \
            else self.cache_remain[self.req-1]
        self.count_ohr[self.req] = 0.0 if self.req == 0 else self.count_ohr[self.req-1]
        self.count_bhr[self.req] = 0.0 if self.req == 0 else self.count_bhr[self.req-1]
        self.size_all[self.req] = 0.0 if self.req == 0 else self.size_all[self.req-1]
        self.cost[self.req] = 0.0

        # self.obj[self.req][0]: timestamp
        # self.obj[self.req][1]: obj_id
        # self.obj[self.req][2]: obj_size
        # cache: {'obj_id': [obj_size, last_visit_time]}
        self.obj[self.req] = obj

        self.actions[self.req] = action

        try:
            self.last_req_time[self.req] = self.req - self.cache[self.obj[self.req][1]][1]
        except:
            try:
                self.last_req_time[self.req] = self.req - self.non_cache[self.obj[self.req][1]][1]
            except:
                self.last_req_time[self.req] = self.req

        # create the current state for cache simulator
        online_obj = [self.obj[self.req][2],
                      self.cache_remain[self.req],
                      self.cost[self.req - 1],
                      self.last_req_time[self.req]]

        # simulation
        if self.obj[self.req][2] >= self.cache_size:
            hit = 0
            # record the request
            try:
                self.non_cache[self.obj[self.req][1]][1] = self.req
            except:
                self.non_cache[self.obj[self.req][1]] = [self.obj[self.req][2], self.req] 

        else:
            # accept request
            if self.actions[self.req] == 1:
                try: 
                    # find the object in the cache, no cost, OHR and BHR ++
                    self.cache[self.obj[self.req][1]][1] = self.req
                    self.count_ohr[self.req] += 1
                    self.count_bhr[self.req] += self.obj[self.req][2]
                    self.size_all[self.req] += self.obj[self.req][2]
                    self.cost[self.req] = 0
                    hit = 1

                except IndexError:
                    # can't find the object in the cache, add the object into cache after replacement (if needed), cost ++
                    while self.cache_remain[self.req] < self.obj[self.req][2]:
                        # remove the largest object in the cache, can be changed to other replacement policies
                        rm_id = max([[k, v] for k, v in self.cache.items()], key=lambda i: i[1])[0]
                        self.cache_remain[self.req] += self.cache[rm_id][0]
                        self.cost[self.req] += self.cache[rm_id][0]
                        del self.cache[rm_id] 
                    # add into cache {obj_id: [size, last_time]}
                    self.cache[self.obj[self.req][1]] = [self.obj[self.req][2], self.req] 
                    self.cache_remain[self.req] -= self.obj[self.req][2]
                    self.size_all[self.req] += self.obj[self.req][2]
                    # cost value is based on size, can be changed
                    self.cost[self.req] += self.obj[self.req][2]
                    hit = 0

            # reject request
            else:
                self.size_all[self.req] += self.obj[self.req][2]
                hit = 0
                # record the request
                try:
                    self.non_cache[self.obj[self.req][1]][1] = self.req
                except:
                    self.non_cache[self.obj[self.req][1]] = [self.obj[self.req][2], self.req] 

        self.ohr[self.req] = self.count_ohr[self.req] / (self.req + 1)
        self.bhr[self.req] = self.count_bhr[self.req] / self.size_all[self.req]
        reward = hit * self.cost[self.req]

        self.req += 1
        
        online_obj = np.array(online_obj)
        assert self.observation_space.contains(online_obj)
        
        return online_obj, reward, {'BHR': self.bhr[self.req], \
                                    'OHR': self.ohr[self.req], \
                                    'Cost': self.cost[self.req] \
                                   }

        # record internal state in a dataframe
        def to_df(self): 
            cols = ['obj_size', 'action', 'cache_remain', 'last_req_time', 'cost', 'ohr', 'bhr']
            df = pd.DataFrame({
                'obj_size': self.obj,
                'action': self.actions, 
                'cache_remain': self.cache_remain_prev, 
                'last_req_time': self.last_req_time,
                'cost': self.cost, 
                'ohr': self.ohr,
                'bhr': self.bhr
            }, columns=cols)
            return df


class CacheEnv(core.Env):
    """
    Cache description.

    * STATE *
        The state is represented as a vector:
        [request object size, 
         cache remaining size, 
         cost of previous request, 
         time of last request to the same object]

    * ACTIONS *
        Whether the cache accept the incoming request, represented as an
        integer in [0, 1].

    * REWARD *
        BHR (byte-hit-ratio)
    
    * REFERENCE *    
    """
    def __init__(self):
        self.seed(config.seed)
        self.cache_size = config.cache_size
        # load trace files from config.cache_trace, attach initial online feature values
        self.src = TraceSrc(trace=config.cache_trace, cache_size=self.cache_size)        
        # set up the observation and action space
        self.action_space = spaces.Discrete(2)
        self.observation_space= spaces.Box(self.src.min_values, \
                                           self.src.max_values, \
                                           dtype=np.float32)
        
        # cache simulator
        self.sim = CacheSim(n_request=self.src.n_request, \
                            cache_size=self.cache_size, \
                            action_space=self.action_space, \
                            observation_space=self.observation_space)
        
        # reset environment (generate new jobs)
        self.reset()

    def reset(self):
        self.src.reset()
        self.sim.reset()
        return self.src.step()[0]
    
    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def step(self, action):
        # 0 <= action < num_servers
        assert self.action_space.contains(action)
        
        observation, done = self.src.step()
        online_obj, reward, info = self.sim.step(action, observation)
        return online_obj, reward, done, info
    
    def render(self, mode='human', close=False):
        pass
