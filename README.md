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
env.run(agent_impl.Agent, agent_parameters)
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

Note: to use `argparse` that is compatiable with park parameters, add parameters using
```
from park.param import parser
parser.add_argument('--new_parameter')
config, _ = parser.parse_known_args()
print(config.new_parameter)
```
