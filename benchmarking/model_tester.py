import gym_multi_car_racing
from gym_multi_car_racing import MultiCarRacing
import gym
import numpy as np


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
                 agent,  # This should be inherited from TesterAgent and already initialized with your agent
                 num_test_episodes=10,
                 render=False,
                 ):
        self.agent = agent
        self.setup_action_space_func = agent.setup_action_space
        self.observation_type = agent.get_observation_type()
        self.num_test_episodes = num_test_episodes
        self.env = None
        self.render = render

    def _create_env(self):
        env = gym.make("MultiCarRacing-v0",
                       num_agents=1,
                       direction='CCW',
                       use_random_direction=True,
                       backwards_flag=True,
                       h_ratio=0.25,
                       use_ego_color=False,
                       setup_action_space_func=self.setup_action_space_func,
                       episode_end_func=lambda x, y: False,
                       observation_type=self.observation_type)
        return env

    def evaluate(self):
        self.env = self._create_env()
        score_arr = []
        for n_epi in range(self.num_test_episodes):
            print('Episode {}/{}'.format(n_epi+1, self.num_test_episodes))
            done = False
            s = self.env.reset()
            while not done:
                action_vec = self.agent.state_to_action(s)
                s, r, done, info = self.env.step(action_vec)
                score = info['total_score']
                score_arr.append(score)
                if self.render:
                    self.env.render()
            print('Final score of episode: ', score)
        self.env.close()
        score_arr = np.array(score_arr)
        eval_data = {
            'scores': score_arr,
            'avg_score': np.mean(score_arr),
            'std_score': np.std(score_arr),
            'num_episodes': self.num_test_episodes
        }
        return eval_data