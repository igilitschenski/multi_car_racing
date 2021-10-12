# Code modified from 
# https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN/blob/master/play_car_racing_by_the_model.py

import gym
import gym_multi_car_racing
from collections import deque
from DuelingDDQN import DuelingDDQN
import numpy as np

num_episodes = 1
train_model = '.\save\\trial_39.h5'
stack_size = 4


if __name__ == '__main__':

    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                   use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                   use_ego_color=False)
    agent = DuelingDDQN(feature_size=env.num_feats*stack_size)
    agent.load_weights(train_model)

    for episode in range(num_episodes):
        env.reset()
        init_state = env.get_feat(car_id=0)

        total_reward = 0
        state_stack_queue = deque([init_state]*stack_size, maxlen=stack_size)
        time_frame_counter = 1
        while True:
            env.render()
            current_state_stack = np.transpose(np.array(state_stack_queue),(1,0))
            action_index = agent.act(current_state_stack, eps=0)
            next_state, reward, done, info = env.step(action_index)

            total_reward += reward

            state_stack_queue.append(np.squeeze(next_state)) # squeeze because env.step return an array of shape (#cars, state_dim) but for now stack is only for single car meaning input must have shape of (state_dim,)
            next_state_stack = np.transpose(np.array(state_stack_queue),(1,0)) # transform deque to array and shift stack dim to the end

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(episode+1, num_episodes, time_frame_counter, float(total_reward)))
                break
            time_frame_counter += 1

    env.close()