import park


def run_env_with_random_agent(env_name, seed):
    env = park.make(env_name)
    env.seed(seed)

    obs = env.reset()
    done = False

    while not done:
        act = env.action_space.sample()
        obs, reward, done, info = env.step(act)
