# Source: https://github.com/seungeunrho/minimalRL/blob/master/ppo.py
import os, sys

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
learning_rate = 0.0005 #0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 5#3
T_horizon     = 20#100 #100 #20
max_temp = 6.0

# Reward weights
velocity_reward = 5

# Penalty weights
grass_penalty = -100
backwards_penalty = -1.5
angle_diff_penalty = -100


actions =  [
                (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Action Space Structure
                (-1, 1, 0), (0, 1, 0), (1, 1, 0),  # (Steering Wheel, Gas, Break)
                (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Range        -1~1       0~1   0~1
                (-1, 0, 0), (0, 0, 0), (1, 0, 0)
            ]

n_epi_train   = 800
n_epi_test = 10


VELOCITY_REWARD_PROPORTIONAL = 5.0
VELOCITY_REWARD_LOW = -1.0

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(48,256)
        self.fc_pi = nn.Linear(256,12)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #samples actions from policy given states
    def pi(self, x, softmax_dim = 0, t=1.0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x/t, dim=softmax_dim)
        return prob
    
    # value estimates
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r.astype(np.float32))
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            entropy = -torch.sum(pi * torch.log(pi + 1e-20), dim=1, keepdim=True)
            t2 = F.smooth_l1_loss(self.v(s) , td_target.detach())
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach()) - entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
    

def get_reward(env, action):
    step_reward = np.zeros(env.num_agents)
    for car_id, car in enumerate(env.cars):  # First step without action, called from reset()

        # Rewards
        step_reward[car_id] += abs(env.all_feats[car_id, 47]) * velocity_reward * (1-float(env.all_feats[car_id, 4])) #normalize the velocity later



        #velocity = abs(env.all_feats[car_id, 47])
        #step_reward[car_id] += (VELOCITY_REWARD_LOW if velocity < 1.0 else velocity * VELOCITY_REWARD_PROPORTIONAL)  # normalize the velocity later
       

        if action is not None and action[1] == 1:
            step_reward[car_id] *= 2

        # Penalties
        step_reward[car_id] += abs(env.all_feats[car_id, 3]) * angle_diff_penalty  # normalize angle diff later
        step_reward[car_id] += float(env.all_feats[car_id, 4]) * grass_penalty  # driving on grass
        step_reward[car_id] += float(env.all_feats[car_id, 5]) * backwards_penalty  # driving backward
    return step_reward


def action_space_setup(env):
    env.cont_action_space = actions
    env.action_space = gym.spaces.Discrete(len(env.cont_action_space))


def train():
    env = gym.make("MultiCarRacing-v0", 
                    num_agents = 1, 
                    verbose = 1,
                    direction='CCW',
                    use_random_direction = True, 
                    backwards_flag = True, 
                    h_ratio = 0.25,
                    use_ego_color = False, 
                    setup_action_space_func = action_space_setup,
                    get_reward_func = get_reward)
    
    model = PPO()
    model_path = "./saved_models/PPO/state_dict_model.pt"
    model.load_state_dict(torch.load(model_path))

    scores = []
   
    print_interval = 5
    save_interval = 50

    for n_epi in range(n_epi_train):
        done = False
        score = 0.0
        negative_reward_counter = 0;
        slow_velocity_counter = 0;
        env.reset()
        s = env.get_feat(car_id=0)
        temperature = np.exp(-(n_epi - n_epi_train) / n_epi_train * np.log(max_temp))
        
        action_list = []
        while not done:
            
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float(), t=temperature)
                m = Categorical(prob)
                a = m.sample().item()
                action = env.cont_action_space[a]
                _, r, done, _ = env.step(action)
                
                s_prime = env.get_feat(car_id=0)
                #model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime
                
                action_list.append(a)

                reward = r.item()/100.0
                # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
                negative_reward_counter = negative_reward_counter + 1 if reward < 0 else 0

                
                score += reward
                env.render()
                if done or negative_reward_counter >=80:
                    if negative_reward_counter >=80:
                        print("negative_reward_counter is over 80")
                        print(reward)
                    done = True
                    break

            model.train_net()

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
    scores, action_list = train()
    #print(scores)
    print(action_list)
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()