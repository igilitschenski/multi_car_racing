import os
import sys
import gym
from collections import deque
from DQNAgent import CarRacingDQNAgent

if __name__ == '__main__':
    main_path = sys.path[0]
    date_str = '20_09_17_47'
    model_name = 'episode_500.h5'
    model_dir = os.path.join(main_path, 'save', date_str, model_name)

    play_episodes = 1

    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNAgent(epsilon=0) # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(model_dir)

    for e in range(play_episodes):
        init_state = env.reset()
        init_state = agent.process_state_image(init_state)

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        
        while True:
            env.render()

            current_state_frame_stack = agent.generate_state_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            next_state, reward, done, info = env.step(action)

            total_reward += reward

            next_state = agent.process_state_image(next_state)
            state_frame_stack_queue.append(next_state)

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e+1, 
                                    play_episodes, time_frame_counter, float(total_reward)))
                break
            time_frame_counter += 1