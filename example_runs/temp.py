'''
This is an example script to run the multi-agent car racing environment
but with a single car. Actions are discretized and states are features
obtained from the game
'''
import gym
import gym_multi_car_racing
import random

env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)

obs = env.reset()
done = False
total_reward = 0

while not done:
  # actions are discretized as follows:
  # 1-) left and little gas,  2-) little gas,       3-) right and little gas
  # 4-) left and gas,         5-) gas,              6-) right and gas
  # 7-) left and brake,       8-) brake,            9-) right and brake
  # 10-) left,                11-) nothing,         12-) right
  action = random.randint(0, env.action_space.n - 1) # taking random action

  # state is the feature vectors from the game
  # reward is the step reward at the current time step
  # done is a bool and info is not used (an empty dict).
  state, reward, done, info = env.step(action)

  # features can be obtained for each car as follows
  feats = env.get_feat(car_id=0) 
  print(feats.shape)

  total_reward += reward
  env.render()

print("individual scores:", total_reward)