from .multi_car_racing import MultiCarRacing

from gym.envs.registration import register

register(
    id='MultiCarRacing-v0',
    entry_point='gym_multi_car_racing:MultiCarRacing',
    max_episode_steps=1000,
    reward_threshold=900
)
