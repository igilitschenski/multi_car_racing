# Code modified from https://github.com/seungeunrho/minimalRL/blob/master/a3c.py
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


# Hyperparameters
n_train_processes = 6
learning_rate = 0.0001
beta = 0.001
update_interval = 5
gamma = 0.99
max_train_ep = 700
max_test_ep = 800
num_frames = 5
max_temp = 5.0


def to_grayscale(img):
    return np.dot(img, [0.299, 0.587, 0.144])


def zero_center(img):
    return (img - 127.0) / 256.0


def crop(img, bottom=12, left=6, right=6):
    height, width = img.shape
    return img[0: height - bottom, left: width - right]


class ActorCritic(nn.Module):
    def __init__(self, num_of_inputs, num_of_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(num_of_inputs, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.linear1 = nn.Linear(32*9*9, 256)
        self.policy = nn.Linear(256, num_of_actions)
        self.value = nn.Linear(256, 1)
        self.num_of_actions = num_of_actions

    def forward(self, x, t=1.0):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))

        flattened = torch.flatten(conv2_out, start_dim=1)  # N x 9*9*32
        linear1_out = self.linear1(flattened)

        policy_output = self.policy(linear1_out)
        value_output = self.value(linear1_out).squeeze(-1)

        probs = F.softmax(policy_output / t, dim=-1)
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


def train(global_model, rank):
    car_id = 0  # Change this for multi-agent caseß
    local_model = ActorCritic(num_frames, global_model.num_of_actions)
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                   use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                   use_ego_color=False)

    for n_epi in range(max_train_ep):
        done = False

        # Initialize state with repeated first frame
        s = env.reset()
        s = preprocess_state(s, car_id)
        s = s.repeat(1, num_frames, 1, 1)
        temperature = np.exp(-(n_epi - max_train_ep) / max_train_ep * np.log(max_temp))
        print('Temperature: ', temperature)
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                prob = local_model(s, t=temperature)[0]
                m = Categorical(prob)
                a = m.sample().item()
                # if np.random.uniform(0.0, 1.0) < 0.01:
                #     a = np.random.randint(0, global_model.num_of_actions)
                s_prime, r, done, _ = env.step(a)
                r /= 100.
                s_prime = preprocess_state(s_prime, car_id)

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r.astype(np.float32))

                s = add_frame(s_prime, s)
                if done:
                    break

            s_final = s
            R = 0.0 if done else local_model(s_final, t=temperature)[2].item()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                td_target_lst.append(R)
            td_target_lst.reverse()

            s_batch, a_batch, td_target = torch.cat(s_lst, dim=0), torch.tensor(a_lst).squeeze(-1), \
                torch.tensor(td_target_lst).squeeze(-1)
            advantage = td_target - local_model(s_batch, t=temperature)[2]

            pi = local_model(s_batch, t=temperature)[0]
            pi_a = pi.gather(1, a_batch.unsqueeze(-1)).squeeze(-1)
            entropy_loss = -(pi * torch.log(pi + 1e-20)).sum(dim=1).mean()
            policy_loss = (-torch.log(pi_a) * advantage.detach()).mean()
            value_loss = F.smooth_l1_loss(local_model(s_batch, t=temperature)[2], td_target.detach())
            loss = 1.0 * policy_loss + \
                0.5 * value_loss + \
                -beta * entropy_loss
            print('Probs:{}, E:{}'.format(pi.detach(), entropy_loss.detach()))

            optimizer.zero_grad()
            loss.backward()
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
        print('Rank {}, episode {}, loss {}'.format(rank, n_epi, loss.mean().detach()))
    env.close()
    print("Training process {} reached maximum episode.".format(rank))


def test(global_model):
    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                   use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                   use_ego_color=False)
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
            s_prime, r, done, _ = env.step(a)
            s_prime = preprocess_state(s_prime, car_id)
            s = add_frame(s_prime, s)
            score += r
            env.render()
           # print('On grass: {}, driving bckwd: {}, velocity: {}, angle diff: {}, done: {}'.format(env.all_feats[0, 4], env.all_feats[car_id, 5], env.all_feats[0, 33], env.all_feats[car_id, 3], env.all_feats[0, 32]))

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            print("# of episode :{}, avg score : {}".format(n_epi, avg_score))
            if avg_score > best_score:
                best_score = avg_score
                print('Saving best model.')
                torch.save(global_model.state_dict(), '../a3c_agent.pth')
            score = 0.0
            time.sleep(1)
    env.close()



if __name__ == '__main__':
    global_model = ActorCritic(num_frames, MultiCarRacing.num_of_actions)
    global_model.share_memory()

    processes = []
    for rank in range(n_train_processes + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(global_model,))
        else:
            p = mp.Process(target=train, args=(global_model, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
