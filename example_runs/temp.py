import gym
import gym_multi_car_racing

env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)

obs = env.reset()
done = False
total_reward = 0

while not done:
  # The actions have to be of the format (num_agents,3)
  # The action format for each car is as in the CarRacing-v0 environment.
  action = env.action_space.sample()

  # Similarly, the structure of this is the same as in CarRacing-v0 with an
  # additional dimension for the different agents, i.e.
  # obs is of shape (num_agents, 96, 96, 3)
  # reward is of shape (num_agents,)
  # done is a bool and info is not used (an empty dict).
  obs, reward, done, info = env.step(action)
  feats = env.get_feat(car_id=0) # feature is a vector of 32x1, go to get_feat for more details
  print(feats.shape)

  total_reward += reward
  env.render()

print("individual scores:", total_reward)