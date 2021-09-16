import gym
import gym_multi_car_racing

from stable_baselines3 import DQN

env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=990, log_interval=4)
model.save("dqn_singleCar")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_singleCar")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()