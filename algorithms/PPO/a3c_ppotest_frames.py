# Code modified from https://github.com/seungeunrho/minimalRL/blob/master/a3c.py
import os, sys

from numpy.core.fromnumeric import squeeze

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
import torch.multiprocessing as mp
import time, sys
import gym_multi_car_racing
from gym_multi_car_racing import MultiCarRacing
import numpy as np
import matplotlib.pyplot as plt



# Hyperparameters
n_train_processes = 1
learning_rate = 0.0001
beta = 0.001
lmbda         = 0.95
eps_clip      = 0.1

update_interval = 5
gamma = 0.99
max_train_ep = 700
max_test_ep = 800
num_frames = 5
max_temp = 5.0



ACTIONS = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Action Space Structure
        (-1, 1, 0), (0, 1, 0), (1, 1, 0),  # (Steering Wheel, Gas, Break)
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Range        -1~1       0~1   0~1
        (-1, 0, 0), (0, 0, 0), (1, 0, 0)
        ]
num_of_actions =  len(ACTIONS)

VELOCITY_REWARD_PROPORTIONAL = 1.0
VELOCITY_REWARD_LOW = -5.0
ANGLE_DIFF_REWARD = 0.0
ON_GRASS_REWARD = -1.0
BACKWARD_REWARD = 0.0


def to_grayscale(img):
    return np.dot(img, [0.299, 0.587, 0.144])


def zero_center(img):
    return (img - 127.0) / 256.0


def crop(img, bottom=12, left=6, right=6):
    height, width = img.shape
    return img[0: height - bottom, left: width - right]


class PPO(nn.Module):
    def __init__(self, num_of_inputs, num_of_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(num_of_inputs, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.linear1 = nn.Linear(32*9*9, 256)
        self.policy = nn.Linear(256, num_of_actions)
        self.value = nn.Linear(256, 1)
        #self.num_of_actions = num_of_actions
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, t=1.0):
        x = x.to(torch.float32)
        conv1_out = F.relu(self.conv1(x))

        if torch.isnan(conv1_out).any():
                print("conv1_out has a nan")


        conv2_out = F.relu(self.conv2(conv1_out))

        flattened = torch.flatten(conv2_out, start_dim=1)  # N x 9*9*32
        linear1_out = self.linear1(flattened)

        policy_output = self.policy(linear1_out)
        value_output = self.value(linear1_out).squeeze(-1)

        probs = F.softmax(policy_output  / t, dim=-1)
        log_probs = F.log_softmax(policy_output, dim=-1)
        return probs, log_probs, value_output


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

        velocity = abs(env.all_feats[car_id, 33])
        step_reward[car_id] += (VELOCITY_REWARD_LOW if velocity < 2.0 else velocity * VELOCITY_REWARD_PROPORTIONAL)  # normalize the velocity later
        step_reward[car_id] += abs(env.all_feats[car_id, 3]) * ANGLE_DIFF_REWARD  # normalize angle diff later

        step_reward[car_id] += float(env.all_feats[car_id, 4]) * ON_GRASS_REWARD  # driving on grass
        step_reward[car_id] += float(env.all_feats[car_id, 5]) * BACKWARD_REWARD  # driving backward
    return step_reward

def train():
    car_id = 0  # Change this for multi-agent caseß
    model = PPO(num_frames, num_of_actions)
    #model_path = "./saved_models/PPO/state_dict_model.pt"

    #model.load_state_dict(global_model.state_dict())

    env = gym.make("MultiCarRacing-v0",
                   num_agents=1,
                   direction='CCW',
                   use_random_direction=True,
                   backwards_flag=True,
                   h_ratio=0.25,
                   use_ego_color=False,
                   setup_action_space_func=action_space_setup,
                   get_reward_func=get_reward,
                   observation_type='frames')

    for n_epi in range(max_train_ep):
        done = False

        # Initialize state with repeated first frame
        s = env.reset()
        s = preprocess_state(s, car_id)
        s_prime = s

        s = s.repeat(1, num_frames, 1, 1)

        # Initialize sprime_frames with repeated first frame
        sprime_frames = s

        temperature = np.exp(-(n_epi - max_train_ep) / max_train_ep * np.log(max_temp))
        #print('Temperature: ', temperature)
        while not done:
            s_lst, a_lst, r_lst, sprime_lst, prob_lst, done_lst = [], [], [], [], [], []
            for t in range(update_interval):
                s = add_frame(s_prime, s)
                prob = model(s, t=temperature)[0]
                m = Categorical(prob)
                a = m.sample().item()
                print(ACTIONS[a])
                action_vec = np.array(ACTIONS[a])
                s_prime, r, done, _ = env.step(action_vec)
                r = r[0]
                r /= 100.
                s_prime = preprocess_state(s_prime, car_id)

                #splot = s.squeeze(0)
                #plt.imshow(splot[4,:,:])
                #plt.show()

                # plt.imshow(s_prime.squeeze(0).squeeze(0))
                # plt.show()


                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r.astype(np.float32))
                prob_lst.append(prob[0,a].item())
                done_mask = 0 if done else 1
                done_lst.append([done_mask])

                sprime_frames = add_frame(s_prime, sprime_frames)
                sprime_lst.append(sprime_frames)
                if done:
                    break

            s_final = s
            R = 0.0 if done else model(s_final.float(), t=temperature)[2].item()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                td_target_lst.append(R)
            td_target_lst.reverse()

            s_batch, a_batch,  r_batch, sprime_batch= torch.cat(s_lst, dim=0), torch.tensor(a_lst).squeeze(-1), \
                torch.tensor(r_lst), torch.cat(sprime_lst, dim=0)
                
            td_target = torch.tensor(td_target_lst).squeeze(-1).float()
            advantage = td_target - model(s_batch.float(), t=temperature)[2]


            # td_target = r_batch + gamma * local_model(sprime_batch.float(), t=temperature)[2] * torch.tensor(done_lst).squeeze(-1)
            # if torch.isnan(td_target).any():
            #     print("td_target has a nan")
            # delta = td_target - local_model(s_batch, t=temperature)[2]
            # delta = delta.detach().numpy()
            # advantage_lst = []
            # advantage = 0.0
            # for delta_t in delta[::-1]:
            #     advantage = gamma * lmbda * advantage + delta_t
            #     advantage_lst.append([advantage])
            # advantage_lst.reverse()
            # advantage = torch.tensor(advantage_lst, dtype=torch.float).squeeze(-1)
            
                       
            

            pi = model(s_batch.float(), t=temperature)[0]
            pi_a = pi.gather(1, a_batch.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(torch.log(pi_a) - torch.log(torch.tensor(prob_lst)))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage


            entropy_loss = -(pi * torch.log(pi + 1e-20)).sum(dim=1).mean().type(torch.float32)
            print (entropy_loss.requires_grad)
            policy_loss = torch.min(surr1, surr2).mean().type(torch.float32)
            #value_loss = F.smooth_l1_loss(local_model(s_batch.float(), t=temperature)[2], td_target.detach()).type(torch.float32)
            loss = -1.0 * policy_loss -beta * entropy_loss
                #0.5 * value_loss + \
                #-beta * entropy_loss
            #print('Probs:{}, E:{}'.format(pi.detach(), entropy_loss.detach()))

            self.optimizer.zero_grad()
            loss.backward()
            # for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                
            #     global_param._grad = local_param.grad
            #     print(torch.norm(local_param.grad))

            self.optimizer.step()
        print('Rank {}, episode {}, loss {}'.format(rank, n_epi, loss.mean().detach()))
    env.close()
    print("Training process {} reached maximum episode.".format(rank))


def test(global_model):
    env = gym.make("MultiCarRacing-v0",
                   num_agents=1,
                   direction='CCW',
                   use_random_direction=True,
                   backwards_flag=True,
                   h_ratio=0.25,
                   use_ego_color=False,
                   setup_action_space_func=action_space_setup,
                   get_reward_func=get_reward,
                   observation_type='frames'
                   )
    score = 0.0
    print_interval = 10
    car_id = 0
    best_score = -100000000.0
    for n_epi in range(max_test_ep):
        done = False
        s = env.reset()
        s = preprocess_state(s, car_id)
        s = s.repeat(1, num_frames, 1, 1)
        while not done:
            prob = global_model(s)[0]
            a = Categorical(prob).sample().item()
            action_vec = np.array(ACTIONS[a])
            s_prime, r, done, _ = env.step(action_vec)
            r = r[0]
            s_prime = preprocess_state(s_prime, car_id)
            s = add_frame(s_prime, s)
            score += r
            #env.render()
           # print('On grass: {}, driving bckwd: {}, velocity: {}, angle diff: {}, done: {}'.format(env.all_feats[0, 4], env.all_feats[car_id, 5], env.all_feats[0, 33], env.all_feats[car_id, 3], env.all_feats[0, 32]))

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            print("# of episode :{}, avg score : {}".format(n_epi, avg_score))
            if avg_score > best_score:
                best_score = avg_score
                print('Saving best model.')
                torch.save(global_model.state_dict(), '../ppo_agent.pth')
            score = 0.0
            time.sleep(1)
    env.close()



if __name__ == '__main__':
    #global_model = PPO(num_frames, len(ACTIONS))
    #global_model.share_memory()
    train()


    # p = mp.Process(target=train, args=(global_model, rank,))
    # p.start()


    # processes = []
    # for rank in range(n_train_processes + 1):  # + 1 for test process
    #     if rank == 0:
    #         p = mp.Process(target=test, args=(global_model,))
    #     else:
    #         p = mp.Process(target=train, args=(global_model, rank,))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
