# This code is originated from https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN
import os
import sys
import argparse
import gym
import gym_multi_car_racing
import fDDQNAgent
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from fDDQNAgent import CarRacingDQNAgent

# Set parameters
RENDER                        = True
STARTING_EPISODE              = 101
ENDING_EPISODE                = 601
SKIP_FRAMES                   = 0
TRAINING_BATCH_SIZE           = 64
SAVE_TRAINING_FREQUENCY       = 50
UPDATE_TARGET_MODEL_FREQUENCY = 5

VELOCITY_REWARD_PROPORTIONAL = 0.05
VELOCITY_REWARD_LOW = -0.1
ANGLE_DIFF_REWARD = 0.0
ON_GRASS_REWARD = -1
BACKWARD_REWARD = 0
def get_reward(env, action):
    step_reward = np.zeros(env.num_agents)
    for car_id, car in enumerate(env.cars):  # First step without action, called from reset()

        velocity = abs(env.all_feats[car_id, 47])
        step_reward[car_id] += (VELOCITY_REWARD_LOW if velocity < 2.0 else velocity * VELOCITY_REWARD_PROPORTIONAL)  # normalize the velocity later
        step_reward[car_id] += abs(env.all_feats[car_id, 3]) * ANGLE_DIFF_REWARD  # normalize angle diff later

        step_reward[car_id] += float(env.all_feats[car_id, 4]) * ON_GRASS_REWARD  # driving on grass
        step_reward[car_id] += float(env.all_feats[car_id, 5]) * BACKWARD_REWARD  # driving backward
        step_reward[car_id] -= 0.1
    return step_reward

if __name__ == '__main__':

    main_path = sys.path[0]
    date_now = datetime.now()
    date_str = date_now.strftime("%d_%m_%H_%M")
    save_folder = os.path.join(main_path, 'save', date_str)

    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    #parser.add_argument('-m', '--model', default='\\Users\\Parsa\\Desktop\\Formula1\\Group\\multi_car_racing\\algorithms\\dueling\\Feature\save\\09_10_21_15\\episode_100.h5')

    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    args = parser.parse_args()

    #env = gym.make('CarRacing-v0') # original environment
    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                   use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                   use_ego_color=False, get_reward_func=get_reward)
    agent = CarRacingDQNAgent(epsilon=args.epsilon)
    if args.model:
        agent.load(args.model)

    total_reward_list = []
    real_reward_list = []   
    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        print('Episode #{}'.format(e))
        #init_state = env.reset()
        env.reset()
        init_state = env.get_feat(car_id=0)
        init_state = agent.process_state_feats(init_state)

        total_reward = 0
        real_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        
        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = agent.generate_state_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            # print('Action to apply: {}'.format(action))
            reward = 0
            for _ in range(SKIP_FRAMES+1):
                next_state, r, done, info = env.step(action)
                reward += r
                real_reward += r
                if done:
                    break

            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            #Extra bonus for the model if it uses full gas
            if action[1] == 1 and action[2] == 0:
               reward *= 1.5

            total_reward += reward

            
            next_state = env.get_feat(car_id=0)

            next_state = agent.process_state_feats(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = agent.generate_state_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)
            
            if done or negative_reward_counter >= 25 or total_reward < -5:
                print('Episode: {}/{}, Time Frames: {}, Rewards: {:.2}, Rewards(adjusted): {:.2}, Epsilon: {:.2}'.format(
                                e, ENDING_EPISODE, time_frame_counter, float(real_reward), float(total_reward), float(agent.epsilon)))
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            save_full_dir = '{}/episode_{}.h5'.format(save_folder, e)
            agent.save(save_full_dir)

        total_reward_list.append(total_reward)
        real_reward_list.append(real_reward)

        env.close()

    total_reward_arr = np.asarray(total_reward_list)
    real_reward_arr = np.asarray(real_reward_list)

    # Plotting the rewards
    fig_id = plt.figure(figsize=(12,9))
    ax = fig_id.add_subplot(211)
    ax.plot(total_reward_arr)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward (adjusted)')

    ax = fig_id.add_subplot(212)
    ax.plot(real_reward_arr)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')

    plot_full_dir = '{}/rewards.png'.format(save_folder)
    plt.savefig(plot_full_dir)


    
    

