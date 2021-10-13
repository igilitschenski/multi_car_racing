# Source: https://github.com/seungeunrho/minimalRL/blob/master/ppo.py
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
#from matplotlib.ticker import MaxNLocator


#Hyperparameters
learning_rate = 0.0001 #0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 1 #5#3
T_horizon     = 20 #100 #100 #20

num_frames = 5
beta = 0.001
SCALE = 10.0  # Track scale
PLAYFIELD = 2000 / SCALE  # Game over boundary
max_temp = 5.0
ACTIONS = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Action Space Structure
        (-1, 1, 0), (0, 1, 0), (1, 1, 0),  # (Steering Wheel, Gas, Break)
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Range        -1~1       0~1   0~1
        (-1, 0, 0), (0, 0, 0), (1, 0, 0)
        ]
VELOCITY_REWARD_PROPORTIONAL = 0.0
VELOCITY_REWARD_LOW = -10.0
ANGLE_DIFF_REWARD = 0.0
ON_GRASS_REWARD = -1.0
BACKWARD_REWARD = 0.0

n_epi_train   = 800
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

    # def forward(self, x, t=1.0):
    #     conv1_out = F.relu(self.conv1(x))

    #     if torch.isnan(conv1_out).any():
    #             print("conv1_out has a nan")

    #     conv2_out = F.relu(self.conv2(conv1_out))

    #     flattened = torch.flatten(conv2_out, start_dim=1)  # N x 9*9*32
    #     linear1_out = self.linear1(flattened)

    #     policy_output = self.policy(linear1_out)
    #     value_output = self.value(linear1_out).squeeze(-1)

    #     probs = F.softmax(policy_output / t, dim=-1)
    #     log_probs = F.log_softmax(policy_output, dim=-1)
    #     return probs, log_probs, value_output
      

def train(self):
    env = gym.make("MultiCarRacing-v0", 
                    num_agents = 1, 
                    verbose = 1,
                    direction='CCW',
                    use_random_direction = True, 
                    backwards_flag = True, 
                    h_ratio = 0.25,
                    use_ego_color = False, 
                    setup_action_space_func = action_space_setup,
                    get_reward_func = get_reward,
                   episode_end_func=episode_end,
                   observation_type='frames')
    
    #model = PPO(num_frames)
    model_path = "./state_dict_model.pt"
    model.load_state_dict(torch.load(model_path))
    
    scores = []
   
    print_interval = 5
    save_interval = 50

    for n_epi in range(n_epi_train):
        done = False
        score = 0.0
        negative_reward_counter = 0;
        slow_velocity_counter = 0;
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
            for t in range(T_horizon):
                s = add_frame(s_prime, s)
                #prob = model(s, temperature)[0]
                prob = model.pi(s, t=temperature)
                #prob = model.pi(torch.from_numpy(s).float(), t=temperature)
                m = Categorical(prob)
                a = m.sample().item()
                #action = env.cont_action_space[a]
                action_vec = np.array(ACTIONS[a])
                s_prime, r, done, _ = env.step(action_vec)
                r = r[0]
                print(r)
                r /= 100
                s_prime = preprocess_state(s_prime, car_id)

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r.astype(np.float32))
                prob_lst.append(prob[0,a].item())
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                env.render()
                
                #model.put_data((s, a, r, s_prime, prob[0,a].item(), done))
                
                action_list.append(a)
                #s = add_frame(s_prime, s)
                sprime_frames = add_frame(s_prime, sprime_frames)
                sprime_lst.append(sprime_frames)
                score += r
                if done:
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
            loss = -1.0 * policy_loss + \
                0.5 * value_loss + \
                -beta* entropy_loss

            #entropy = -torch.sum(pi * torch.log(pi + 1e-20), dim=1, keepdim=True)
            #t2 = F.smooth_l1_loss(self.v(s) , td_target.detach())
            #loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach()) - entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            #model.train_net(temperature, s=s_lst, a=a_lst, r=r_lst, s_prime=sprime_lst, done_mask=done_lst, prob_a=prob_lst)

            if n_epi%save_interval == 0:
                torch.save(model.state_dict(), model_path)


        scores.append(score)
        #print(action_list)

        #if n_epi%save_interval == 0:
            #torch.save(model.state_dict(), model_path)

        if n_epi%print_interval == 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score))
            print(action_list)
            print(prob)
    
    env.close()

    torch.save(model.state_dict(), model_path)
    return scores, action_list


def test():
    
    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                   use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                   use_ego_color=False)
    model = PPO()
    # Load saved model
    model_path = "./saved_models/PPO/state_dict_model.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    score = 0.0
    print_interval = 1
    scores = []

    for n_epi in range(n_epi_test):
        score = 0.0
        done = False
        env.reset()
        s = env.get_feat(car_id=0)
        action_list = []
        while not done:
            
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                action = env.cont_action_space[a]
                action_list.append(a)
                _, r, done, _ = env.step(action)
                s_prime = env.get_feat(car_id=0)
                model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime
                score += r.item()
                env.render()

        scores.append(score)
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, score))
        print(action_list)
        #if n_epi%print_interval==0 and n_epi!=0:
            #print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            #score = 0.0
        
    env.close()
    return scores, action_list

if __name__ == '__main__':
    model = PPO(num_frames)
    scores, action_list = train(model)
    #print(scores)
    print(action_list)
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()