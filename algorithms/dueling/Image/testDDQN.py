import os
import sys
import gym
from collections import deque
from matplotlib import animation
import matplotlib.pyplot as plt
from DDQNAgent import CarRacingDQNAgent

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

if __name__ == '__main__':
    save_or_plot = 'save'
    main_path = sys.path[0]
    date_str = '26_09_15_27'
    model_name = 'episode_500.h5'
    model_dir = os.path.join(main_path, 'save', date_str, model_name)

    play_episodes = 10

    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNAgent(epsilon=0) # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(model_dir)

    for e in range(play_episodes):
        frames = []
        init_state = env.reset()
        init_state = agent.process_state_image(init_state)

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        
        while True:
            if save_or_plot == 'plot':
                env.render()
            else:
                frames.append(env.render(mode="rgb_array"))

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
        
        if save_or_plot == 'save':
            save_frames_as_gif(frames, path=os.path.join(main_path, 'save', date_str), 
                                            filename='gym_animation_{}.gif'.format(e+1))
