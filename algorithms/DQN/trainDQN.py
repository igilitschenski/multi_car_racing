# This code is originated from https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN
import os
import sys
import argparse
import gym
import DQNAgent
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from DQNAgent import CarRacingDQNAgent

RENDER                        = True
STARTING_EPISODE              = 1
ENDING_EPISODE                = 501 # 1000
SKIP_FRAMES                   = 2
TRAINING_BATCH_SIZE           = 64
SAVE_TRAINING_FREQUENCY       = 50
UPDATE_TARGET_MODEL_FREQUENCY = 5

if __name__ == '__main__':
    main_path = sys.path[0]
    date_now = datetime.now()
    date_str = date_now.strftime("%d_%m_%H_%M")
    save_folder = os.path.join(main_path, 'save', date_str)

    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    args = parser.parse_args()

    env = gym.make('CarRacing-v0') # original environment
    agent = CarRacingDQNAgent(epsilon=args.epsilon)
    if args.model:
        agent.load(args.model)

    total_reward_list = []
    real_reward_list = []
    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):
        print('Episode #{}'.format(e))
        init_state = env.reset()
        init_state = agent.process_state_image(init_state)

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

            # Extra bonus for the model if it uses full gas
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward

            next_state = agent.process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = agent.generate_state_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
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