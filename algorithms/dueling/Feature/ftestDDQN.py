import os
import sys
import gym
import gym_multi_car_racing
from collections import deque
from matplotlib import animation
import matplotlib.pyplot as plt
from fDDQNAgent import CarRacingDQNAgent
import numpy as np

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=2)
    writergif = animation.PillowWriter(fps=120) 

    # plt.show()
    anim.save(os.path.join(path,filename), writer=writergif)

    return -1


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
    save_or_plot = 'save'
    main_path = sys.path[0]
    date_str = '09_10_21_15'
    model_name = 'episode_600.h5'
    model_dir = os.path.join(main_path, 'save', date_str, model_name)

    play_episodes = 10

    #env = gym.make('CarRacing-v0')
    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                   use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                   use_ego_color=False, get_reward_func=get_reward)
    agent = CarRacingDQNAgent(epsilon=0) # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(model_dir)

    for e in range(play_episodes):
        frames = []
        env.reset()
        init_state = env.get_feat(car_id=0)
        init_state = agent.process_state_feats(init_state)

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        
        while True:
            if save_or_plot == 'plot':
                env.render()
            else:
                framesThis = env.render(mode="rgb_array")
                framesThis = np.squeeze(framesThis)
                frames.append(framesThis)

            current_state_frame_stack = agent.generate_state_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            next_state, reward, done, info = env.step(action)


            total_reward += reward
            next_state = env.get_feat(car_id=0)

            next_state = agent.process_state_feats(next_state)
            state_frame_stack_queue.append(next_state)

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e+1, 
                                    play_episodes, time_frame_counter, float(total_reward)))
                break
            time_frame_counter += 1
        
        if save_or_plot == 'save':
            save_frames_as_gif(frames, path=os.path.join(main_path, 'save', date_str), 
                                            filename='gym_animation_{}.gif'.format(e+1))
