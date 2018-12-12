import numpy as np

from park import core, spaces
from park.param import config
from park.utils import seeding
from park.envs.load_balance.job import Job
from park.envs.load_balance.job_generator import generate_jobs
from park.envs.load_balance.server import Server
from park.envs.load_balance.timeline import Timeline
from park.envs.load_balance.wall_time import WallTime


class LoadBalanceEnv(core.Env):
    """
    Balance the load among n (default 10) heterogeneous servers
    to minimize  average job processing time (queuing delay).
    Jobs arrive according to a Poisson process and the job size
    distributes according to a Pareto distribution.

    * STATE *
        Current Load (total work waiting in the queue +
        remaining work currently being processed (if any) among n
        servers) and the incoming job size.
        The state is represented as a vector:
        [load_server_1, load_server_2, ..., load_server_n, job_size]

    * ACTIONS *
        Which server to assign the incoming job, represented as an
        integer in [0, n-1].

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
        Figure 1a, Section 6.2 and Appendix J
        Variance Reduction for Reinforcement Learning in Input-Driven Environments. 
        H Mao, SB Venkatakrishnan, M Schwarzkopf, M Alizadeh.
        https://openreview.net/forum?id=Hyg1G2AqtQ
    
        Certain optimality properties of the first-come first-served discipline for g/g/s queues.
        DJ Daley.
        Stochastic Processes and their Applications, 25:301–308, 1987.
    """
    def __init__(self):
        # random seed
        self.seed(config.seed)
        # global timer
        self.wall_time = WallTime()
        # uses priority queue
        self.timeline = Timeline()
        # total number of streaming jobs (can be very large)
        self.num_stream_jobs = config.num_stream_jobs
        # servers
        self.servers = self.initialize_servers(config.service_rates)
        # current incoming job to schedule
        self.incoming_job = None
        # finished jobs (for logging at the end)
        self.finished_jobs = []

    def generate_jobs(self):
        all_t, all_size = generate_jobs(self.num_stream_jobs, self.np_random)
        for t, size in zip(all_t, all_size):
            self.timeline.push(t, size)

    def initialize(self):
        assert self.wall_time.curr_time == 0
        new_time, obj = self.timeline.pop()
        self.wall_time.update(new_time)
        assert isinstance(obj, int)  # a job arrival event
        size = obj
        self.incoming_job = Job(size, self.wall_time.curr_time)

    def initialize_servers(self, service_rates):
        servers = []
        for server_id in range(config.num_servers)
            server = Server(server_id, service_rates[server_id], self.wall_time)
            servers.append(server)
        return servers

    def observe(self):
        return self.servers, self.incoming_job

    def reset(self):
        for server in self.servers:
            server.reset()
        self.wall_time.reset()
        self.timeline.reset()
        self.generate_jobs()
        self.incoming_job = None
        self.finished_jobs = []
        # initialize environment (jump to first job arrival event)
        self.initialize()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def step(self, action):

        # schedule job to server
        self.servers[action].schedule(self.incoming_job)
        running_job = self.servers[action].process()
        if running_job is not None:
            self.timeline.push(running_job.finish_time, running_job)

        # erase incoming job
        self.incoming_job = None

        # set to compute reward from this time point
        reward = 0

        while len(self.timeline) > 0:

            new_time, obj = self.timeline.pop()

            # update reward
            num_active_jobs = sum(len(w.queue) for w in self.servers)
            for server in self.servers:
                if server.curr_job is not None:
                    assert server.curr_job.finish_time >= \
                           self.wall_time.curr_time  # curr job should be valid
                    num_active_jobs += 1
            reward -= (new_time - self.wall_time.curr_time) * num_active_jobs

            # tick time
            self.wall_time.update(new_time)

            if isinstance(obj, int):  # new job arrives
                size = obj
                self.incoming_job = Job(size, self.wall_time.curr_time)
                # break to consult agent
                break

            elif isinstance(obj, Job):  # job completion on server
                job = obj
                self.finished_jobs.append(job)
                if job.server.curr_job == job:
                    # server's current job is done
                    job.server.curr_job = None
                running_job = job.server.process()
                if running_job is not None:
                    self.timeline.push(running_job.finish_time, running_job)

            else:
                print("illegal event type")
                exit(1)

        done = ((len(self.timeline) == 0) and \
               self.incoming_job is None)

        return self.observe(), reward, done, {'curr_time': self.wall_time.curr_time}
