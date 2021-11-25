import gym_multi_car_racing
from gym_multi_car_racing import MultiCarRacing
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

class TesterAgent:
    def __init__(self):
        pass

    def state_to_action(self, state):
        raise NotImplementedError

    @staticmethod
    def setup_action_space(env):
        raise NotImplementedError

    @staticmethod
    def get_observation_type():
        raise NotImplementedError


class ModelTester:
    def __init__(self,
                 agents,  # This should be inherited from TesterAgent and already initialized with your agent
                 num_test_episodes=10,
                 render=False,
                 save_gif=False,
                 save_gif_idx=0,
                 ):
        self.agents = agents
        self.setup_action_space_func = agents[0].setup_action_space
        self.observation_type = agents[0].get_observation_type()
        self.num_test_episodes = num_test_episodes
        self.env = None
        self.render = render
        self.save_gif = save_gif
        self.gif_frames = []
        self.save_gif_idx = save_gif_idx
        self.num_cars = len(agents)

    def _create_env(self):
        env = gym.make("MultiCarRacing-v0",
                       num_agents=self.num_cars,
                       direction='CCW',
                       use_random_direction=True,
                       backwards_flag=True,
                       h_ratio=0.25,
                       use_ego_color=False,
                       setup_action_space_func=self.setup_action_space_func,
                       episode_end_func=lambda x, y: False,
                       observation_type=self.observation_type,
                       seed=42)
        return env

    def evaluate(self):
        self.env = self._create_env()
        score_arr = []
        for n_epi in range(self.num_test_episodes):
            print('Episode {}/{}'.format(n_epi+1, self.num_test_episodes))
            done = False
            s = self.env.reset()
            while not done:
                action_vec = []
                for i in range(self.num_cars):
                    action_vec.append(self.agents[i].state_to_action(s))
                action_vec = np.stack(action_vec, axis=0)
                s, r, done, info = self.env.step(action_vec)
                if self.render:
                    self.env.render()
                if self.save_gif and n_epi == self.save_gif_idx:
                    self.gif_frames.append(self.env.render(mode="rgb_array"))
            if self.save_gif and n_epi == self.save_gif_idx:
                _save_frames_as_gif(self.gif_frames)
            score = info['total_score']
            score_arr.append(score)
            print('Final score of episode: ', score)
        self.env.close()
        score_arr = np.array(score_arr)
        eval_data = {
            'scores': score_arr,
            'avg_score': np.mean(score_arr, axis=0),
            'std_score': np.std(score_arr, axis=0),
            'num_episodes': self.num_test_episodes
        }
        return eval_data

def _save_frames_as_gif(frames, path='./', filename='gym_animation'):
    num_cars = frames[0].shape[0]
    for i in range(num_cars):
        frame = [f[i] for f in frames]
        plt.figure(figsize=(frame[0].shape[1] / 72.0, frame[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frame[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frame[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frame), interval=2.0)
        writergif = animation.PillowWriter(fps=60)

        # plt.show()
        fname = filename + '_' + str(i) + '.gif'
        anim.save(os.path.join(path, fname), writer=writergif)
    return -1