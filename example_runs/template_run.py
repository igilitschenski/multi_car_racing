'''
This is an example script to run the multi-agent car racing environment
but with a single car. Actions are discretized and states are features
obtained from the game
'''
import os
import sys
import gym
import random
import numpy as np
import gym_multi_car_racing


# env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
#         use_random_direction=True, backwards_flag=True, h_ratio=0.25,
#         use_ego_color=False)

SCALE = 6.0  # Track scale
PLAYFIELD = 2000 / SCALE  # Game over boundary
ACTIONS = [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0.2)]
action_description = ['left', 'right', 'accaleration', 'brake']
VELOCITY_REWARD_PROPORTIONAL = 0.0
VELOCITY_REWARD_LOW = -10.0
ANGLE_DIFF_REWARD = 0.0
ON_GRASS_REWARD = -1.0
BACKWARD_REWARD = 0.0

def action_space_setup(env):
    env.cont_action_space = ACTIONS
    env.action_space = gym.spaces.Discrete(len(env.cont_action_space))

def get_reward(env, action):
    step_reward = np.zeros(env.num_agents)
    for car_id, car in enumerate(env.cars):  # First step without action, called from reset()

        velocity = abs(env.all_feats[car_id, 33])
        step_reward[car_id] += (VELOCITY_REWARD_LOW if velocity < 2.0 else velocity * VELOCITY_REWARD_PROPORTIONAL)  # normalize the velocity later
        step_reward[car_id] += abs(env.all_feats[car_id, 3]) * ANGLE_DIFF_REWARD  # normalize angle diff later

        step_reward[car_id] += float(env.all_feats[car_id, 4]) * ON_GRASS_REWARD  # driving on grass
        step_reward[car_id] += float(env.all_feats[car_id, 5]) * BACKWARD_REWARD  # driving backward
    return step_reward

def episode_end(env, car_id):
    done = False
    x, y = env.cars[car_id].hull.position
    if (abs(x) > PLAYFIELD or abs(y) > PLAYFIELD) or (env.driving_backward[car_id]):
        done = True
    return done


env = gym.make("MultiCarRacing-v0", num_agents=1,
                direction='CCW', use_random_direction=True,
                backwards_flag=True, h_ratio=0.25,
                use_ego_color=False, setup_action_space_func=action_space_setup,
                get_reward_func=get_reward, episode_end_func=episode_end,
                observation_type='frames')

obs = env.reset()
done = False
total_reward = 0
counter = 0

while not done:

  action_idx = random.randint(0, len(ACTIONS) - 1) # taking random action
  # print('Chosen action: {}'.format(action_description[action_idx]))
  state, reward, done, info = env.step(ACTIONS[action_idx])

  # features can be obtained for each car as follows
  feats = env.all_feats[0]
  if counter % 10 == 0:
    print('\n\nFeature space: {}'.format(feats.shape))
    print('Car Position: {} and {}'.format(feats[0], feats[1]))
    print('Car Angle: {:.3f} pi'.format(feats[2]/np.pi))
    print('Angle Diff: {:.3f} pi'.format(feats[3]/np.pi))
    print('On grass: {}'.format(feats[4]))
    print('Driving backward: {}'.format(feats[5]))

    print('Distance to the left: {}'.format(feats[45]))
    print('Distance to the right: {}'.format(feats[46]))
    print('Velocity: {}'.format(feats[47]))
    
    # 6-36 are the tile information
    for tile_ahead_idx in range(3):
      feat_idx_this = tile_ahead_idx*3+6
      print('{}-th tile ahead, distance: {}, {} and angle diff: {:.3f} pi'.format(tile_ahead_idx+1,
                         feats[feat_idx_this], feats[feat_idx_this+1], feats[feat_idx_this+2]/np.pi))

  total_reward += reward
  counter = counter + 1
  env.render()

print("individual scores:", total_reward)