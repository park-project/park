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

Note: to use `argparse` that is compatiable with park parameters, add parameters using
```
from park.param import config, parser
parser.add_argument('--new_parameter')
print(config.new_parameter)
```
