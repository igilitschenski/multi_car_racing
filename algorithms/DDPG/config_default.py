import numpy as np


def add_settings_to_config(config, new_config):
    for key in new_config:
        if key in config:
            config[key] = new_config[key]
    return config


def set_config_default_DDPG():

    config = {}

    config['noise_type'] = 'ou'
    config['noise_std'] = np.array([0.1, 0.8], dtype=np.float32)
    config['noise_mean'] = np.zeros((2,), dtype=np.float32)
    config['noise_scale'] = 1
    config['action_dim'] = 2

    config['tau'] = 0.005
    config['gamma'] = 0.99

    config['init_lr_actor'] = 0.00001
    config['init_lr_critic'] = 0.002

    config['rw_actor'] = 0
    config['rw_critic'] = 0
    config['rt_actor'] = 'l2'
    config['rt_critic'] = 'l2'
    
    config['decay_lr_steps'] = 10
    config['decay_lr_rate'] = 0.99

    config['decay_noise_steps'] = 10
    config['decay_noise_rate'] = 0.99

    config['batch_size'] = 64
    config['max_memory_size'] = 50000

    config['max_to_keep'] = 20
    config['train_or_test'] = 'train'
    config['ckpt_load'] = 150

    config['directory_to_save'] = '/Users/erayerturk/Documents/USC/Courses/CSCI527/multi_car_racing'
    
    return config