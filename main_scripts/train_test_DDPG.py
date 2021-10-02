import numpy as np
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from distutils.util import strtobool
main_proj_dir = os.path.dirname(sys.path[0])
sys.path.append(os.path.join(main_proj_dir, 'gym_multi_car_racing'))
sys.path.append(os.path.join(main_proj_dir, 'algorithms', 'DDPG'))
sys.path.append(os.path.join(main_proj_dir, 'tools'))
import argparse
from aux_functions import *
from config_default import *
from DDPG import *
from multi_car_racing import *


if __name__=="__main__":

    main_directory = main_proj_dir
    print('main directory to save the results: {}'.format(main_directory))
    def str2bool(v):
        return bool(strtobool(v)) 

    parser = argparse.ArgumentParser(description='Training the agent on single agent car racing')

    parser.add_argument('--num_cars', type=int, default=1, help='Number of cars to train the agent (only 1 is supported rn)')
    parser.add_argument('--render', type=str2bool, nargs='?', const=True, default=True, help='Whether to display the game on the screen during training')
    parser.add_argument('--num_episodes', type=int, default=4000, help='Number of episodes to play the game')
    parser.add_argument('--max_episode_length', type=int, default=2500, help='Max number of timesteps for a single episode')
    parser.add_argument('--skip_frames', type=int, default=1, help='Number of frames to execute a single action taken by actor network')
    parser.add_argument('--max_neg_steps', type=int, default=200, help='Maximum number of consecutive negative steps w/ nonpositive reward to terminate the episode')

    parser.add_argument('--noise_type', type=str, default='ou', help='Type of noise to be added to the action (ou or normal)')
    parser.add_argument('--noise_std', type=str, default='0.1,0.8', help='Comma separated noise std for OU noise, check the length matches the action dimension')
    parser.add_argument('--noise_scale', type=float, default=1, help='Noise scale for normal noise')
    parser.add_argument('--action_dim', type=int, default=2, help='Action dimension (2, 3, 4)')
    parser.add_argument('--tau', type=float, default=0.005, help='1 - polyak average weight to update target network weights')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for Q-value calculation')
    parser.add_argument('--max_memory_size', type=int, default=50000, help='Max size of replay buffer')
    parser.add_argument('--min_memory_size', type=int, default=2000, help='Max size of replay buffer')
    parser.add_argument('--action_divide_factor', type=int, default=4, help='Divide the action generated to ease the training')
    
    parser.add_argument('--init_lr_actor', type=float, default=0.00001, help='Learning rate for actor model')
    parser.add_argument('--init_lr_critic', type=float, default=0.002, help='Learning rate for critic model')
    parser.add_argument('--reg_weight_actor', type=float, default=0, help='Regularization weight for actor model')
    parser.add_argument('--reg_weight_critic', type=float, default=0, help='Regularization weight for critic model')
    parser.add_argument('--reg_type_actor', type=str, default='l2', help='Regularization type for actor model (l2, l1)')
    parser.add_argument('--reg_type_critic', type=str, default='l2', help='Regularization weight for critic model (l2, l1)')
    parser.add_argument('--decay_lr_steps', type=int, default=10, help='Exponential decay steps for lrs')
    parser.add_argument('--decay_lr_rate', type=float, default=0.99, help='Exponential decay rate for lrs')
    parser.add_argument('--decay_noise_steps', type=int, default=10, help='Exponential decay steps for noise')
    parser.add_argument('--decay_noise_rate', type=float, default=0.99, help='Exponential decay rate for noise')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training the networks')
    parser.add_argument('--max_to_keep', type=int, default=20, help='Maximum number of checkpoints to save')

    parser.add_argument('--train_or_test', type=str, default='train', help='Whether to train or load a model')
    parser.add_argument('--test_eps', type=int, default=1, help='Number of episodes to test the model')
    parser.add_argument('--ckpt_load', type=int, default=168, help='Number of checkpoint to load')
    parser.add_argument('--create_gif', type=str2bool, nargs='?', const=True, default=True, help='Whether to create gif during testing')

    # get the arguments
    args = parser.parse_args()
    num_cars = args.num_cars
    render = args.render
    num_episodes = args.num_episodes
    max_episode_length = args.max_episode_length
    skip_frames = args.skip_frames
    max_neg_steps = args.max_neg_steps

    noise_type = args.noise_type
    noise_std = np.array([float(i) for i in args.noise_std.split(',')])
    noise_scale = args.noise_scale
    action_dim = args.action_dim 
    tau = args.tau
    gamma = args.gamma
    max_memory_size = args.max_memory_size
    min_memory_size = args.min_memory_size
    action_divide_factor = args.action_divide_factor
    batch_size = args.batch_size

    init_lr_actor = args.init_lr_actor
    init_lr_critic = args.init_lr_critic
    reg_weight_actor = args.reg_weight_actor
    reg_weight_critic = args.reg_weight_critic
    reg_type_actor = args.reg_type_actor
    reg_type_critic = args.reg_type_critic
    decay_lr_steps = args.decay_lr_steps
    decay_lr_rate = args.decay_lr_rate
    decay_noise_steps = args.decay_noise_steps
    decay_noise_rate =args.decay_noise_rate
    max_to_keep = args.max_to_keep

    train_or_test = args.train_or_test
    ckpt_load = args.ckpt_load
    test_eps = args.test_eps

    # create the config to overwrite to default one
    config_write = dict(noise_type=noise_type, noise_std=noise_std, noise_scale=noise_scale, action_dim=action_dim, tau=tau, gamma=gamma, max_memory_size=max_memory_size, 
                        init_lr_actor=init_lr_actor, init_lr_critic=init_lr_critic, rw_actor=reg_weight_actor, rw_critic=reg_weight_critic, 
                        decay_lr_steps=decay_lr_steps, decay_lr_rate=decay_lr_rate, decay_noise_steps=decay_noise_steps, decay_noise_rate=decay_noise_rate,
                        rt_actor=reg_type_actor, rt_critic=reg_type_critic, batch_size=batch_size, max_to_keep=max_to_keep, directory_to_save=main_directory,
                        train_or_test=train_or_test, ckpt_load=ckpt_load)

    # create the default config and overwrite
    config = set_config_default_DDPG()
    config = add_settings_to_config(config, config_write)
    print('Settings used to start the training \n {}'.format(config))

    # create the game environment and reset it
    # gym.logger.set_level(40)
    env = MultiCarRacing(num_cars)
    env.reset()

    # create the learning agent
    STATE_W = 96   # less than Atari 160x192
    STATE_H = 96
    state_shape = (STATE_W, STATE_H, 3)
    agent = DDPG(config, state_shape, env)
    
    if train_or_test == 'train':

        # create tensorflow writer
        agent.create_writer()

        # Fill the memory
        for st in range(min_memory_size):
            print('Filling memory, step: {}'.format(st))
            # render if requested
            if render:
                env.render()

            # get state and action
            state = env.render('state_pixels')
            action, action_before = agent.get_action(state)
            action = action / action_divide_factor
            print('action: {}, action_before: {}'.format(action, action_before))
            # execute action on the environment
            reward = 0
            for i in range(skip_frames):
                next_state, r, terminal, _ = env.step(action)
                reward += r
            
            # append the experience to the memory (append the action (generated by the actor + noise) )
            agent.memory.append_memory(state, action_before, reward, next_state, terminal)

            # reset the env if terminal state
            if terminal:
                env.reset()

        # average reward initialization
        avg_reward = tf.Variable(0, dtype=tf.float32)
        cum_reward = tf.Variable(0, dtype=tf.float32)

        # start running episodes
        start_ep = ckpt_load if ckpt_load is not None else 0
        for ep_num in range(start_ep, num_episodes):

            # reset the environment, save models, update and reset noise, update learning rate
            env.reset()
            agent.save_models()
            agent.reset_update_noise(ep_num=ep_num)
            agent.update_network_lrs(ep_num=ep_num)

            # set episode variables to write tensorboard
            ep_reward = tf.Variable(0, dtype=tf.float32)
            ep_length = 0
            neg_reward_count = 0
            ep_done = False

            print('Episode {} starts...'.format(ep_num+1))
            while not ep_done:

                print('Episode {}, timestep {}/{}'.format(ep_num+1, ep_length, max_episode_length))

                # render the env if requested
                if render:
                    env.render()
                
                # get the action from last state
                state = env.render('state_pixels')

                # get the action from actor (divide by 4 to train easier)
                action, action_before = agent.get_action(state)
                action = action / action_divide_factor
                
                # obtain the next state
                reward = 0
                for _ in range(skip_frames):
                    next_state, r, terminal, _ = env.step(action)
                    reward += r

                print('Episode {}, step {}, action: {}, before clipping {}'.format(ep_num+1, ep_length, action, action_before))
            
                # if agent gets negative reward for max_neg_steps steps, episode ends
                if reward <= 0:
                    neg_reward_count += 1
                else:
                    neg_reward_count = 0

                # increase ep length and ep reward
                ep_length += 1
                ep_reward.assign_add(reward[0])

                if terminal or ep_length == max_episode_length or neg_reward_count == max_neg_steps:
                    ep_done = True
                    agent.memory.append_memory(state, action_before, 0., next_state, True)
                else:
                    agent.memory.append_memory(state, action_before, reward, next_state, terminal)

                # train the networks
                agent.train_step()
            
            env.close()
            # get the avg and cumulative reward
            cum_reward.assign_add(ep_reward)
            print('Episode reward: {}'.format(ep_reward.numpy()))

            avg_reward = cum_reward.numpy() / (ep_length + 1)
            print('Average reward: {}'.format(avg_reward))

            # write avg and episode reward to tensorboard
            agent.def_summary(avg_reward, ep_reward, ep_num, 'train')

    else:
        frames = []
        env.reset()

        if render:
            env.render()

        agent.create_writer()

        # average reward initialization
        avg_reward = tf.Variable(0, dtype=tf.float32)
        cum_reward = tf.Variable(0, dtype=tf.float32)

        for ep_num in range(test_eps):
            print('Test episodes: {}/{}'.format(ep_num+1, test_eps))
            ep_reward = tf.Variable(0, dtype=tf.float32)
            ep_done = False
            ep_length = 0

            while not ep_done:
                # get the video frame and append to frames
                frames.append(env.render('rgb_array'))

                # render the env if requested
                if render:
                    env.render()

                # get the state and execute on the environment
                state = env.render('state_pixels')
                action, action_before = agent.get_action(state)
                action = action / action_divide_factor

                reward = 0
                for _ in range(skip_frames):
                    next_state, r, terminal, _ = env.step(action)
                reward += r

                ep_length += 1
                ep_reward.assign_add(reward[0])

                if terminal or ep_length == max_episode_length:
                    ep_done = True

            # get the avg and cumulative reward
            cum_reward.assign_add(ep_reward)
            print('Episode reward: {}'.format(ep_reward.numpy()))

            avg_reward = cum_reward.numpy() / (ep_num + 1)
            print('Average reward: {}'.format(avg_reward))

            # write avg and episode reward to tensorboard
            agent.def_summary(avg_reward, ep_reward, ep_num, 'train')

            if ep_num != test_eps - 1: 
                print('Starting next_episode...')
            else:
                print('Creating the gif for the episode...')
        
        # create the gif out of the episode
        create_gif(frames, interval=100, dpi=80, save_path=os.path.join(main_directory, 'gifs', 'DDPG.gif'))

