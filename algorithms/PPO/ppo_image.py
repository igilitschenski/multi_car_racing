import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.dirname(sys.path[0]))

project_dir = os.path.dirname(os.path.dirname(sys.path[0]))

sys.path.append(project_dir)

sys.path.append(os.path.join(project_dir, 'gym_multi_car_racing'))


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym_multi_car_racing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
sys.path.insert(0, '../../benchmarking')
from model_tester import TesterAgent




#Hyperparameters
learning_rate = 0.0001 #0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.05
K_epoch       = 1 #5#3
T_horizon     = 5 #100 #100 #20

num_frames = 5
beta = 0.1
SCALE = 10.0  # Track scale
PLAYFIELD = 2000 / SCALE  # Game over boundary
max_temp = 1.0
ACTIONS = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Action Space Structure
        (-1, 1, 0), (0, 1, 0), (1, 1, 0),  # (Steering Wheel, Gas, Break)
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Range        -1~1       0~1   0~1
        (-1, 0, 0), (0, 0, 0), (1, 0, 0)
        ]
VELOCITY_REWARD_PROPORTIONAL = 1.0
VELOCITY_REWARD_LOW = -50.0
ANGLE_DIFF_REWARD = -20.0
ON_GRASS_REWARD = -5.0
BACKWARD_REWARD = -1.0

n_epi_train   = 500
n_epi_test = 10

def to_grayscale(img):
    return np.dot(img, [0.299, 0.587, 0.144])


def zero_center(img):
    return (img - 127.0) / 256.0


def crop(img, bottom=12, left=6, right=6):
    height, width = img.shape
    return img[0: height - bottom, left: width - right]


def preprocess_state(s, car_id):
    s = s[car_id, ...]
    s = to_grayscale(s)
    s = zero_center(s)
    s = crop(s)
    return torch.from_numpy(s.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # N, C, H, W tensor


def add_frame(s_new, s_stack):
    s_stack = torch.roll(s_stack, shifts=-1, dims=1)
    s_stack[:, -1, :, :] = s_new
    return s_stack

def action_space_setup(env):
    env.cont_action_space = ACTIONS
    env.action_space = gym.spaces.Discrete(len(env.cont_action_space))

def get_reward(env, action):
    step_reward = np.zeros(env.num_agents)
    for car_id, car in enumerate(env.cars):  # First step without action, called from reset()

        velocity = abs(env.all_feats[car_id, 47])
        step_reward[car_id] += (VELOCITY_REWARD_LOW if velocity < 2.0 else velocity * VELOCITY_REWARD_PROPORTIONAL)  # normalize the velocity later
        step_reward[car_id] += abs(env.all_feats[car_id, 3]) * ANGLE_DIFF_REWARD  # normalize angle diff later

        step_reward[car_id] += float(env.all_feats[car_id, 4]) * ON_GRASS_REWARD  # driving on grass
        step_reward[car_id] += float(env.all_feats[car_id, 5]) * BACKWARD_REWARD  # driving backward
    return step_reward

def episode_end(env, car_id):
    done = False
    x, y = env.cars[car_id].hull.position
    if (abs(x) > PLAYFIELD or abs(y) > PLAYFIELD) or (env.driving_backward[car_id]):
        done = True
    return done


class PPO(nn.Module):
    def __init__(self, num_of_inputs):
        super(PPO, self).__init__()
        self.data = []
        self.conv1 = nn.Conv2d(num_of_inputs, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.linear1 = nn.Linear(32*9*9, 256)
        self.policy = nn.Linear(256, 12)
        self.value = nn.Linear(256, 1)
        self.num_of_actions = 12
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #samples actions from policy given states
    def pi(self, x, t=1.0):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        flattened = torch.flatten(conv2_out, start_dim=1)  # N x 9*9*32
        linear1_out = self.linear1(flattened)
        policy_output = self.policy(linear1_out)
        prob = F.softmax(policy_output / t, dim=-1)
        return prob
    
    # value estimates
    def v(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        flattened = torch.flatten(conv2_out, start_dim=1)  # N x 9*9*32
        linear1_out = self.linear1(flattened)
        v = self.value(linear1_out).squeeze(-1)
        return v

   
def train(self):
    env = gym.make("MultiCarRacing-v0", 
                    num_agents = 1, 
                    verbose = 0,
                    direction='CCW',
                    use_random_direction = True, 
                    backwards_flag = True, 
                    h_ratio = 0.25,
                    use_ego_color = False, 
                    setup_action_space_func = action_space_setup,
                    get_reward_func = get_reward,
                    #episode_end_func=episode_end,
                    observation_type='frames')
    
    #model_path = "./state_dict_model.pt"
    #model.load_state_dict(torch.load("./state_dict_model_final.pt"))
    

    print_interval = 1
    save_interval = 10

    scores = np.zeros(n_epi_train)
   
    
    avg_loss_episode = np.zeros(n_epi_train)
    avg_entropy_loss_episode = np.zeros(n_epi_train)
    avg_value_loss_episode = np.zeros(n_epi_train)
    avg_policy_loss_episode = np.zeros(n_epi_train)


    for n_epi in range(n_epi_train):
        print("The episode is", n_epi)
        done = False
        score = 0.0
        negative_reward_counter = 0
        slow_velocity_counter = 0
        s = env.reset()
        # Initialize state with repeated first frame
        car_id = 0
        s = preprocess_state(s, car_id)
        s_prime = s
        s = s.repeat(1, num_frames, 1, 1)

        sprime_frames = s

        temperature = np.exp(-(n_epi - n_epi_train) / n_epi_train * np.log(max_temp))
        
        action_list = []


        while not done:
            s_lst, a_lst, r_lst, sprime_lst, prob_lst, done_lst = [], [], [],[], [], []
            loss_lst, entropy_loss_lst, value_loss_lst, policy_loss_lst = [], [], [], []
            for t in range(T_horizon):
                s = add_frame(s_prime, s)
                prob = model.pi(s, t=temperature)
                m = Categorical(prob)
                a = m.sample().item()
                action_vec = np.array(ACTIONS[a])
                s_prime, r, done, total_score = env.step(action_vec)
                r = r[0]
                r /= 100
                s_prime = preprocess_state(s_prime, car_id)

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r.astype(np.float32))
                prob_lst.append(prob[0,a].item())
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                action_list.append(a)
                sprime_frames = add_frame(s_prime, sprime_frames)
                sprime_lst.append(sprime_frames)
                #score += r
                env.render()

                negative_reward_counter = negative_reward_counter + 1 if r < 0 else 0

                if done or negative_reward_counter >= 25:
                    if negative_reward_counter >= 25:
                        print ("negative_reward_counter >= 25, inside")
                    print("done early! breaking out")
                    break

            
            s_batch, sprime_batch, a_batch = torch.cat(s_lst, dim=0), torch.cat(sprime_lst, dim=0), torch.tensor(a_lst).squeeze(-1)
            
            sprime_values = self.v(sprime_batch)
            if torch.isnan(sprime_values).any():
                print("sprime_values has a nan")
            if torch.max(sprime_values) > 1e10:
                print("sprime has some large values")

            td_target = torch.tensor(r_lst) + gamma * sprime_values* torch.tensor(done_lst).squeeze(-1)
            
            if torch.isnan(td_target).any():
                print("td_target has a nan")
            
            s_values = self.v(s_batch)
            if torch.isnan(s_values).any():
                print("s_values has a nan")
            if torch.max(s_values) > 1e10:
                print("s_values has some large values")
            delta = td_target - s_values
            #delta = td_target - model(s_batch, t=temperature)[2]
            if torch.isnan(delta).any():
                print("td_target has a nan")
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).squeeze()

            pi = self.pi(s_batch, t=temperature)
            if torch.isnan(pi).any():
                print("pi has a nan")
            #pi = model(s_batch, t=temperature)[0]
            pi_a = pi.gather(1, a_batch.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(torch.log(pi_a) - torch.log(torch.tensor(prob_lst)))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage

            #Build loss function
            entropy_loss = -(pi * torch.log(pi + 1e-20)).sum(dim=1).mean()
            policy_loss = torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(self.v(s_batch), td_target.detach())
            loss = -2.0 * policy_loss + \
                2.0 * value_loss + \
                -beta* entropy_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            loss_lst.append(loss.detach().numpy().item())
            entropy_loss_lst.append(entropy_loss.detach().numpy().item())
            value_loss_lst.append(value_loss.detach().numpy().item())
            policy_loss_lst.append(policy_loss.detach().numpy().item())

            if negative_reward_counter >= 75:
                print ("negative_reward_counter >= 75. end episode!")
                break
            
        scores[n_epi] = total_score['total_score'][0]

        avg_loss_episode[n_epi] = np.mean(loss_lst)
        avg_entropy_loss_episode[n_epi] = np.mean(entropy_loss_lst)
        avg_value_loss_episode[n_epi] = np.mean(value_loss_lst)
        avg_policy_loss_episode[n_epi] = np.mean(policy_loss_lst)


        if n_epi%save_interval == 0 and n_epi !=0:
            print("saving model!")
            filepath = "./PPO_results/state_dict_model_{}.pt".format(n_epi)
            torch.save(model.state_dict(), filepath)
            plot_losses(scores,avg_loss_episode, avg_entropy_loss_episode, avg_value_loss_episode,avg_policy_loss_episode, n_epi+1)

        if n_epi%print_interval == 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, total_score['total_score'][0]))
            #print(action_list)
            #print(prob)
    
    env.close()

    torch.save(model.state_dict(), "./PPO_results/state_dict_model_final.pt")
    return scores, action_list, avg_entropy_loss_episode, avg_loss_episode, avg_value_loss_episode, avg_policy_loss_episode



def plot_losses(scores,avg_loss_episode, avg_entropy_loss_episode, avg_value_loss_episode,avg_policy_loss_episode, epi_number  ):
    plt.figure(0)
    plt.plot(np.arange(epi_number), scores[0:epi_number])
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.savefig('./PPO_results/scores_{}.png'.format(epi_number-1))

    plt.figure(1)
    plt.plot(np.arange(epi_number), avg_loss_episode[0:epi_number])
    plt.xlabel("Episodes")
    plt.ylabel("Avg. total loss")
    plt.savefig('./PPO_results/total_loss_{}.png'.format(epi_number-1))

    plt.figure(2)
    plt.plot(np.arange(epi_number), avg_entropy_loss_episode[0:epi_number])
    plt.xlabel("Episodes")
    plt.ylabel("Avg. entropy loss")
    plt.savefig('./PPO_results/entropy_{}.png'.format(epi_number-1))


    plt.figure(3)
    plt.plot(np.arange(epi_number), avg_value_loss_episode[0:epi_number])
    plt.xlabel("Episodes")
    plt.ylabel("Avg. value loss")
    plt.savefig('./PPO_results/value_{}.png'.format(epi_number-1))

    plt.figure(4)
    plt.plot(np.arange(epi_number), avg_policy_loss_episode[0:epi_number])
    plt.xlabel("Episodes")
    plt.ylabel("Avg. policy loss")
    plt.savefig('./PPO_results/policy_{}.png'.format(epi_number-1))

    plt.close('all')


def evaluate(self):
    num_test_episodes = 10
    env = gym.make("MultiCarRacing-v0", 
                    num_agents = 1, 
                    verbose = 0,
                    direction='CCW',
                    use_random_direction = True, 
                    backwards_flag = True, 
                    h_ratio = 0.25,
                    use_ego_color = False, 
                    setup_action_space_func = action_space_setup,
                    episode_end_func=lambda x, y: False,
                    observation_type='frames', 
                    seed=42)
    score_arr = []
    for n_epi in range(num_test_episodes):
        print('Episode {}/{}'.format(n_epi+1, num_test_episodes))
        done = False
        s = env.reset()
        s = preprocess_state(s, 0)
        s = s.repeat(1, num_frames, 1, 1)
        while not done:
            prob = model.pi(s, t=1.0)
            m = Categorical(prob)
            a = m.sample().item()
            action_vec = np.array(ACTIONS[a])
            s_prime, r, done, info = env.step(action_vec)
            #score = info['total_score']
            #score_arr.append(score)
            s_prime = preprocess_state(s_prime, 0)
            s = add_frame(s_prime, s)
            
            env.render()
    env.close()

     

class PPOTesterAgent(TesterAgent):
    def __init__(self,
                 model_path='./saved_models/PPO/state_dict_model_220.pt',
                 car_id=0,
                 num_frames=5,
                 actions=ACTIONS,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.car_id = car_id
        self.frame_buffer = None
        self.num_frames = num_frames
        self.actions = actions
        self.agent = self._load_model(model_path)

    def _load_model(self, model_path):
        agent = PPO(self.num_frames)
        agent.load_state_dict(torch.load(model_path))
        agent.eval()
        return agent

    def _update_frame_buffer(self, new_frame):
        if self.frame_buffer is None:
            self.frame_buffer = new_frame.repeat(1, self.num_frames, 1, 1)
        else:
            self.frame_buffer = add_frame(new_frame, self.frame_buffer)

    def state_to_action(self, s):
        """
        This function should take the most recent state and return the
        action vector in the exact same form it is needed for the environment.
        If you are using frame buffer see example in _update_frame_buffer
        how to take care of that.
        """
        s = preprocess_state(s, self.car_id)
        self._update_frame_buffer(s)
        prob = self.agent.pi(self.frame_buffer)
        a = Categorical(prob).sample().item()
        action_vec = np.array(self.actions[a])
        return action_vec

    @staticmethod
    def setup_action_space(env):
        """
        This should be the same action space setup function that you used for training.
        Make sure that the actions set here are the same as the ones used to train the model.
        """
        env.cont_action_space = ACTIONS
        env.action_space = gym.spaces.Discrete(len(env.cont_action_space))

    @staticmethod
    def get_observation_type():
        """
        Simply return 'frames' or 'features'
        """
        return 'frames'


if __name__ == '__main__':
    model = PPO(num_frames)
    model.load_state_dict(torch.load('saved_models/PPO/state_dict_model_220.pt'))
    evaluate(model)

