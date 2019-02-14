import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import heapq

from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.envs.cache.trace_loader import load_traces


class TraceSrc(object):
    def __init__(self, trace, cache_size):
        self.trace = trace
        self.cache_size = cache_size
        self.load_trace = load_traces(self.trace, self.cache_size)
        self.n_request = len(self.load_trace)
        self.cache_size = cache_size
        self.min_values = np.asarray([1, 0, 0])
        self.max_values = np.asarray([self.cache_size, self.cache_size, max(self.load_trace[0])])
        self.req = 0
    
    def reset(self):
        self.load_trace = load_traces(self.trace, self.cache_size)
        self.n_request = len(self.load_trace)
        self.min_values = np.asarray([1, 0, 0])
        self.max_values = np.asarray([self.cache_size, self.cache_size, max(self.load_trace[0])])
        self.idx = 0
        self.req = 0

    def step(self):    
        obs = self.load_trace.iloc[self.idx].values
        self.idx += 1
        self.req += 1
        done = self.req >= self.n_request
        return obs, done


class CacheSim(object) :
    def __init__(self, cache_size, policy, action_space, state_space):
        # invariant
        self.cache_size = cache_size
        self.policy = policy
        self.action_space = action_space
        self.state_space = state_space
        self.req = 0
        self.non_cache = defaultdict(list) # requested items without caching
        self.cache = defaultdict(list) # requested items with caching
        self.cache_pq_size = []
        self.cache_pq_time = []
        self.cache_remain = self.cache_size
        self.last_req_time_dict = {}
        self.count_ohr = 0
        self.count_bhr = 0
        self.size_all = 0
    
    def reset(self):
        self.req = 0
        self.non_cache = defaultdict(list)
        self.cache = defaultdict(list)
        self.cache_pq_size = []
        self.cache_pq_time = []
        self.cache_remain = self.cache_size
        self.last_req_time_dict = {}
        self.count_ohr = 0
        self.count_bhr = 0
        self.size_all = 0
    
    def step(self, action, obj):
        req = self.req
        cache_size_online_remain = self.cache_remain
        discard_obj_if_admit = []
        
        obj_time, obj_id, obj_size = obj[0], obj[1], obj[2]
        try:
            self.last_req_time_dict[obj_id] = req - self.cache[obj[1]][1]
        except:
            try:
                self.last_req_time_dict[obj_id] = req - self.non_cache[obj[1]][1]
            except:
                self.last_req_time_dict[obj_id] = req
                
        # create the current state for cache simulator
        # cache: {'obj_id': [obj_size, last_visit_time]}
        online_obj = [obj_size, cache_size_online_remain, self.last_req_time_dict[obj_id]]
        cost = 0
        
        # simulation
        if obj_size >= self.cache_size:
            hit = 0
            # record the request
            try:
                self.non_cache[obj_id][1] = req
            except:
                self.non_cache[obj_id] = [obj_size, req]

        else:
            # accept request
            if action == 1:
                try: 
                    # find the object in the cache, no cost, OHR and BHR ++
                    self.cache[obj_id][1] = req
                    self.count_bhr += obj_size
                    self.count_ohr += 1
                    self.size_all += obj_size
                    cost = 0
                    hit = 1

                except IndexError:
                    # can't find the object in the cache, add the object into cache after replacement, cost ++
                    while cache_size_online_remain < obj_size:
                        if self.policy == 'size':
                            # remove the largest object in the cache
                            rm_id = self.cache_pq_size[0][1]
                            cache_size_online_remain += self.cache_pq_size[0][0]
                            cost += self.cache_pq_size[0][0]
                        elif self.policy == 'time':
                            # remove the oldest object in the cache
                            rm_id = self.cache_pq_time[0][1]
                            cache_size_online_remain += self.cache_pq_time[0][0]
                            cost += self.cache_pq_time[0][0]
                        discard_obj_if_admit.append(rm_id)
                        heapq.heappop(self.cache_pq_size)
                        heapq.heappop(self.cache_pq_time)
                        del self.cache[rm_id]
                            
                    # add into cache
                    self.cache[obj_id] = [obj_size, req] 
                    heapq.heappush(self.cache_pq_size, (obj_size, obj_id))
                    heapq.heappush(self.cache_pq_time, (req, obj_id))
                    cache_size_online_remain -= obj_size
                    self.size_all += obj_size
                    # cost value is based on size, can be changed
                    cost += obj_size
                    hit = 0

            # reject request
            else:
                self.size_all += obj_size
                hit = 0
                # record the request to non_cache
                try:
                    self.non_cache[obj_id][1] = req
                except:
                    self.non_cache[obj_id] = [obj_size, req]

        bhr = self.count_bhr / self.size_all
        ohr = self.count_ohr / (req + 1)
        reward = hit * cost

        self.req += 1
        self.cache_remain = cache_size_online_remain
        
        assert self.state_space.contains(np.array(online_obj))
        
        return online_obj, reward, {'bhr': bhr, \
                                    'ohr': ohr, \
                                    'cost': cost, \
                                    'discard_obj_if_admit': discard_obj_if_admit, \
                                    'cache_status': self.cache
                                   }


class CacheEnv(core.Env):
    """
    Cache description.

    * STATE *
        The state is represented as a vector:
        [request object size, 
         cache remaining size, 
         time of last request to the same object]

    * ACTIONS *
        Whether the cache accept the incoming request, represented as an
        integer in [0, 1].

    * REWARD *
        Cost of previous step (object size) * hit
    
    * REFERENCE *    
    """
    
    def __init__(self):
        self.seed(config.seed)
        self.cache_size = config.cache_size
        self.policy = config.cache_replace_policy
        
        # load trace, attach initial online feature values
        self.src = TraceSrc(trace=config.cache_trace, cache_size=self.cache_size)

        # set up the state and action space
        self.action_space = spaces.Discrete(2)
        self.state_space= spaces.Box(self.src.min_values, \
                                     self.src.max_values, \
                                     dtype=np.float32)
        
        # cache simulator
        self.sim = CacheSim(cache_size=self.cache_size, \
                            policy=self.policy, \
                            action_space=self.action_space, \
                            state_space=self.state_space)
        
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
        state, done = self.src.step()
        online_obj, reward, info = self.sim.step(action, state)
        return online_obj, reward, done, info
    
    def render(self, mode='human', close=False):
        pass
