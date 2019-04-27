# Park
A suite of system environments wrapped by OpenAI Gym interface.

### Interface
```
import park

env = park.make('load_balance')

obs = env.reset()
done = False

while not done:
    # act = agent.get_action(obs)
    act = env.action_space.sample()
    obs, reward, done, info = env.step(act)
```

### Real system interface
```
import park
import agent_impl  # implemented by user

env = park.make('congestion_control')

# the run script will start the real system
# and periodically invoke agent.get_action
agent = Agent(env.observation_space, env.action_space)
env.run(agent)
```

The `agent_impl.py` should implement
```
class Agent(object):
    def __init__(self, state_space, action_space, *args, **kwargs):
        self.state_space = state_space
        self.action_space = action_space

    def get_action(self, obs, prev_reward, prev_done, prev_info):
        act = self.action_space.sample()
        # implement real action logic here
        return act
```

### Contributors

| Environment                     | Committers |
| -------------                   | ------------- |
| Adaptive video streaming        | Hongzi Mao, Akshay Narayan |
| Spark cluster job scheduling    | Hongzi Mao, Malte Schwarzkopf |
| SQL database query optimization | Parimarjan Negi |
| Network congestion control      | Akshay Narayan, Frank Cangialosi |
| Network active queue management | Mehrdad Khani, Songtao He |
| Circuit design                  | Hanrui Wang, Kuan Wang, Jiacheng Yang |
| Tensorflow device placement     | Ravichandra Addanki |
| CDN memory caching              | Haonan Wang, Wei-Hung Weng |
| Account region assignment       | Ryan Marcus |
| Server load balancing           | Hongzi Mao |
| Switch scheduling               | Ravichandra Addanki, Hongzi Mao |

### Misc
Note: to use `argparse` that is compatiable with park parameters, add parameters using
```
from park.param import parser
parser.add_argument('--new_parameter')
config = parser.parse_args()
print(config.new_parameter)
```
