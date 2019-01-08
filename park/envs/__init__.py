# Format follows OpenAI gym https://gym.openai.com

from park.envs.registration import register, make


register(
    env_id='load_balance',
    entry_point='park.envs.load_balance:LoadBalanceEnv',
)

register(
    env_id='abr_sim',
    entry_point='park.envs.abr_sim:ABRSimEnv',
)

register(
    env_id='aqm',
    entry_point='park.envs.aqm:AQMEnv',
)

register(
    env_id='spark_sim',
    entry_point='park.envs.spark_sim:SparkSimEnv',
)

register(
    env_id='query_optimizer',
    entry_point='park.envs.query_optimizer:QueryOptEnv',
)

register(
    env_id='abr',
    entry_point='park.envs.abr:ABREnv',
)

register(
    env_id='cache',
    entry_point='park.envs.cache:CacheEnv',
)