from park.envs.query_optimizer.query_optimizer import QueryOptEnv

print("qopt next")
env = QueryOptEnv()

env.reset()
env._send("test!")

