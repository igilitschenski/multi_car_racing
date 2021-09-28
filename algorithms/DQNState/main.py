import random

import gym_multi_car_racing
import gym
import DQNAgent
import numpy as np
from numpy import array
import math
import scipy.special as sci
import os
import tensorflow as tf

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# a = [None]*5
# b = len(a)
#
# def dene(name):
#     env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
#                    use_random_direction=True, backwards_flag=True, h_ratio=0.25,
#                    use_ego_color=False)
#
#     obs = env.reset()
#     done = False
#     total_reward = 0
#     iter = 0
#     while not done:
#         # The actions have to be of the format (num_agents,3)
#         # The action format for each car is as in the CarRacing-v0 environment.
#         action = [0,1,0]
#
#         # Similarly, the structure of this is the same as in CarRacing-v0 with an
#         # additional dimension for the different agents, i.e.
#         # obs is of shape (num_agents, 96, 96, 3)
#         # reward is of shape (num_agents,)
#         # done is a bool and info is not used (an empty dict).
#         obs, reward, done, info = env.step(action)
#         total_reward += reward
#         print(str(iter)+" "+str(reward))
#         env.render()
#         iter +=1
#
#     print("individual scores:", total_reward)


#action: (Steering Wheel, Gas, Break); (-1to1, 0to1, 0to1)
# maintain, accelerate, hard-accelerate, decelerate, hard-decelerate, turn-left, turn-right
action_space = [(0,0,0), (0,0.4,0), (0,0.75,0), (0,0,0.4), (0,0,0.75), (1,0,0), (-1,0,0)]

# maintain, accelerate, hard-accelerate, decelerate, hard-decelerate,
# turn-left, turn-left-acc, turn-left-dec, turn-right, turn-right-acc, turn-right-dec
action_space_complex = [(0,0,0), (0,0.5,0), (0,1,0), (0,0,0.5), (0,0,1), (1,0,0),
                        (-1,0,0), (-1,0.5,0), (-1,0,0.5), (1,0,0), (1,0.5,0), (1,0,0.5)]

def convert_action(index, complex=False):
    if complex:
        return action_space_complex[index]
    return action_space[index]

def train(complex):

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    training = True

    if training:
        timestep = 1
        ignore_stopping = True
        state_size = 48
        action_size = len(action_space) if not complex else len(action_space_complex)
        num_state_resets = 100
        num_episodes = 100
        runtime = 5000
        avg_reward = 0
        finpoint = 0
        pol = 0
        target_up = 20
        level_trained = 1
        agent = DQNAgent.DQNAgent(state_size, action_size)
        #agent.load('weight4.h5')
        #agent.T=1
        done = False
        batch_size = 500

        for i in range(0, num_state_resets):
            numcars = 1
            if i == (num_state_resets - 1):
                finpoint = 1
            print('State:' + str(i) + ' Training with car number: ' + str(numcars))
            reward = 0
            counter = 0  # Current step in the episode
            action = 0
            collusion = []
            collusion_count = 0
            env = 0
            for j in range(0, num_episodes):
                print(100 * i + j)
                runsteps = int(runtime/timestep)
                count = 0
                collusion_count = 0
                del env
                env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                               use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                               use_ego_color=False)
                obs = env.reset()
                reward = 0
                sum = 0
                counter = 0
                actionindex = 0
                for step in range(0, runsteps):
                    print(str(step)+" "+str(actionindex) + " " + str(reward))
                    if step < 50:
                        actionindex = random.randint(0, 4)
                        action = convert_action(actionindex, complex)
                        features, reward, done, info = env.step(action)
                        env.render()
                    else:
                        currentstate = np.reshape(env.get_feat(0), [1, 1, state_size])
                        actionindex = agent.act(currentstate)
                        action = convert_action(actionindex, complex)
                        features, reward, done, info = env.step(action)
                        nextstate = np.reshape(env.get_feat(0), [1, 1, state_size])
                        env.render()

                        sum += reward
                        counter += 1

                        if step>50:
                            agent.remember(currentstate, actionindex, reward, nextstate, done)
                        avg_reward += (reward - avg_reward) / float((step + 1 + j * runsteps + i * num_episodes * runsteps))
                        # print('avg_reward: ' + str(avg_reward) + ' ', end=' ')

                        if len(agent.memory) > 5 * batch_size:
                            agent.replay(batch_size)

                        if done:
                            print('collision at step ' + str(step) + ' ', end=' ')
                            collusion_count += 1

                        if done:
                            break

                # print('', end='\n')
                env.close()
                if j == (num_episodes - 1):
                    check = True;

                if j % target_up == 0:
                    agent.update_target_model()

                avg = sum / counter
                # print(avg, end='\n')
                collusion.append(collusion_count)
                file = open('data/reward.dat', 'a')
                file.write('' + str((j + i * num_episodes)) + '\t' + str(avg_reward) + '\n')
                file.close()
                file = open('data/step_reward.dat', 'a')
                file.write('' + str((j + i * num_episodes)) + '\t' + str(avg) + '\n')
                file.close()
                file = open('data/collusion.dat', 'a')
                file.write(str(collusion_count) + '\n')
                file.close()

            if agent.T > 1:
                agent.T = agent.T * 0.9
                # agent.T = agent.T*0.984474
            fname = 'weight' + str(i) + '.h5'
            agent.save(fname)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train(False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
