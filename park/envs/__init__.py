# Format follows OpenAI gym https://gym.openai.com

from park.envs.registration import register, make


register(
    env_id='load_balance',
    entry_point='park.envs.load_balance:LoadBalanceEnv',
)

register(
    env_id='abr',
    entry_point='park.envs.abr:ABREnv',
)

register(
    env_id='aqm',
    entry_point='park.envs.aqm:AQMEnv',
)