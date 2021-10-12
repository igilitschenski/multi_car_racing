# Code modified from 
# https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN/blob/master/CarRacingDQNAgent.py

import gym
import gym_multi_car_racing
import numpy as np
from DuelingDDQN import DuelingDDQN
from collections import deque

# Set parameters
max_training_episodes = 40
stack_size = 4
batch_size = 64
save_frequency = 5
update_frequency = 4
memory_size = 5000
gamma = 0.95  # discount rate
epsilon = 1.0   # Initial exploration rate
epsilon_min = 0.1
epsilon_decay = 0.999
learning_rate = 0.001
skip_frames = 2 
tolerance_steps = 100
max_negative_reward_steps = 25
load_weights = False

if __name__ == '__main__':

    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                   use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                   use_ego_color=False)

    action_space = env.cont_action_space

    agent = DuelingDDQN(feature_size=env.num_feats*stack_size, action_size=12,learning_rate = learning_rate, memory_size = memory_size)
   
    if load_weights:
        agent.load_weights('.\save\\trial_7.h5') 
    counter = 0
    for episode in range(max_training_episodes):
        done = False
        env.reset()
        init_state = env.get_feat(car_id=0)
        total_reward = 0
        negative_reward_counter = 0
        state_stack_queue = deque([init_state]*stack_size, maxlen=stack_size)
        time_frame_counter = 1
        
        while not done:
            env.render()
            current_state_stack = np.transpose(np.array(state_stack_queue),(1,0))   # transform deque to array and shift stack dim to the end
            action_index = agent.act(current_state_stack, epsilon)
            action = action_space[action_index]
            reward = 0
            for _ in range(skip_frames+1):
                next_state, r, done, info = env.step(action_index)
                reward += r
                if done:
                    break

            # If continually getting negative reward 'max_negative_reward_steps' times after the 'tolerance_steps', terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > tolerance_steps and reward < 0 else 0

            # Extra bonus for the model if it uses full gas
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward

            state_stack_queue.append(np.squeeze(next_state)) # squeeze because env.step return an array of shape (#cars, state_dim) but for now stack is only for single car meaning input must have shape of (state_dim,)
            next_state_stack = np.transpose(np.array(state_stack_queue),(1,0)) # transform deque to array and shift stack dim to the end

            agent.memorize(current_state_stack, action_index, reward, next_state_stack, done)

            if done or negative_reward_counter >= max_negative_reward_steps or total_reward < 0:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Epsilon: {:.2}'.format(episode, max_training_episodes, time_frame_counter, float(total_reward), float(epsilon)))
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size, gamma)
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
                    
            time_frame_counter += 1
            print('Episode: {}/{}, Steps: {}, Total Rewards: {}, Epsilon: {}'.format(episode, max_training_episodes, time_frame_counter, float(total_reward), float(epsilon)))
            #if counter % 50 == 0:
                #agent.save_weights('.\save\\trial_{}.h5'.format(episode),overwrite=True)
            counter += 1
            
        if episode % update_frequency == 0:
            agent.update_target_model()
            print('Target model updated\n')

        if episode % save_frequency == 0:
            agent.save_weights('.\save\\trial_{}.h5'.format(episode), overwrite=True)

    env.close()
