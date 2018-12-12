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
