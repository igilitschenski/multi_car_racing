import numpy as np
import matplotlib.pyplot as plt
import os, sys, random 
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import MultivariateNormalTriL
from aux_functions import *
from multi_car_racing import *
from memory_noise_models import *


class DDPG(object):

    def __init__(self, config, state_shape, env):

        self.config = config
        self.state_shape = state_shape
        self.env = env
        if len(self.state_shape) != 3 or not isinstance(self.state_shape, tuple):
            assert 1==0, 'State shape must be tuple and of length 3, check if both correct!'

        self.noise_type = self.config['noise_type'].lower()
        self.noise_std = np.array(self.config['noise_std'], dtype=np.float32).reshape(-1, )
        self.noise_mean = np.array(self.config['noise_mean'], dtype=np.float32).reshape(-1, )
        self.noise_scale = self.config['noise_scale']
        self.action_dim = self.config['action_dim']
        self.tau = self.config['tau']
        self.gamma = self.config['gamma']
        self.max_memory_size = self.config['max_memory_size']
        self.batch_size = self.config['batch_size']

        self.max_to_keep = self.config['max_to_keep']
        self.model_save_suffices = ['saved_models/DDPG/actor', 'saved_models/DDPG/critic', 'saved_models/DDPG/target_actor', 'saved_models/DDPG/target_critic']
        self.model_save_paths = dict(actor_path='{}/{}'.format(self.config['directory_to_save'], self.model_save_suffices[0]), critic_path='{}/{}'.format(self.config['directory_to_save'], self.model_save_suffices[1]),
                                     target_actor_path='{}/{}'.format(self.config['directory_to_save'], self.model_save_suffices[2]), target_critic_path='{}/{}'.format(self.config['directory_to_save'], self.model_save_suffices[3]))

        for _, model_save_path in self.model_save_paths.items():
            if not os.path.isdir(model_save_path):
                os.makedirs(model_save_path)

        # setup the model parameters
        self._set_network_params()
        # set action noise 
        self._set_noise_dist()
        # create/load main and target models
        self._set_or_init_networks()
        # create checkpoints
        self._set_checkpoints()

        # experience replay buffer 
        self.memory  = ReplayBuffer(self.max_memory_size, self.state_shape, self.action_dim)
    

    def _set_or_init_networks(self):

        # create main and target models
        self.actor_model = self._build_actor_network(name='actor')
        self.target_actor_model = self._build_actor_network(name='target_actor')

        self.critic_model = self._build_critic_network(name='critic')
        self.target_critic_model = self._build_critic_network(name='target_critic')

        if self.config['train_or_test'] == 'train' and self.config['ckpt_load'] <= 0:
            # initialize the target and actual models in the same way
            self.target_actor_model.set_weights(self.actor_model.get_weights())
            self.target_critic_model.set_weights(self.critic_model.get_weights())

        else: 
            # create the temp checkpoints to load the weights
            checkpoint_actor = tf.train.Checkpoint(model=self.actor_model)
            checkpoint_target_actor = tf.train.Checkpoint(model=self.target_actor_model)

            checkpoint_critic = tf.train.Checkpoint(model=self.critic_model)
            checkpoint_target_critic = tf.train.Checkpoint(model=self.target_critic_model)

            checkpoint_actor.restore(self.model_save_paths['actor_path'] + '/ckpt-{}'.format(self.config['ckpt_load']))
            checkpoint_target_actor.restore(self.model_save_paths['target_actor_path'] + '/ckpt-{}'.format(self.config['ckpt_load']))
            checkpoint_critic.restore(self.model_save_paths['critic_path'] + '/ckpt-{}'.format(self.config['ckpt_load']))
            checkpoint_target_critic.restore(self.model_save_paths['target_critic_path'] + '/ckpt-{}'.format(self.config['ckpt_load']))


    def _set_network_params(self):
        # set learning rates 
        self.init_lr_actor = self.config['init_lr_actor']
        self.init_lr_critic = self.config['init_lr_critic']
        
        # set regularizations
        self.rw_actor = self.config['rw_actor']
        self.rw_critic = self.config['rw_critic']
        self.rt_actor = self.config['rt_actor']
        self.rt_critic = self.config['rt_critic']

        if self.rt_actor.lower() == 'l2':
            self.rf_actor = tf.keras.regularizers.l2(self.rw_actor)
        elif self.rt_actor.lower() == 'l1':
            self.rf_actor = tf.keras.regularizers.l1(self.rw_actor)
        else:
            assert 1==0, 'Given regularization function is not available for actor network! (l2, l1)'

        if self.rt_critic.lower() == 'l2':
            self.rf_critic = tf.keras.regularizers.l2(self.rw_critic)
        elif self.rt_critic.lower() == 'l1':
            self.rf_critic = tf.keras.regularizers.l1(self.rw_critic)
        else:
            assert 1==0, 'Given regularization function is not available for critic network! (l2, l1)'

        # set the optimizers
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate = self.init_lr_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate = self.init_lr_critic)
        self.initializer_layer = tf.keras.initializers.GlorotNormal()

        # set exponential decay learning rate params
        self.decay_lr_steps = self.config['decay_lr_steps']
        self.decay_lr_rate = self.config['decay_lr_rate']


    def _set_noise_dist(self):
        if self.noise_type == 'ou':
            # if self.action_dim == 2:
            #     self.noise_mean = np.array([0, -0.8], dtype=np.float32)
            # else:
            # self.noise_mean = np.zeros((self.action_dim, ), dtype=np.float32)
        
            # else:
            #     self.noise_mean = 0.5 * np.ones((self.action_dim, ), dtype=np.float32)

            if self.noise_std.size != self.action_dim:
                assert 1==0, 'OU Noise std shape is not compatible with provided action dimension!'
            else:
                self.noise = NoiseGenerator(self.noise_mean, self.noise_std)
        else:
            # if self.action_dim == 2:
            # self.noise_mean = np.zeros((self.action_dim, ), dtype=np.float32)
            # else:
            #     self.noise_mean = 0.5 * np.ones((self.action_dim, ), dtype=np.float32)

            if self.noise_std.size != self.action_dim:
                assert 1==0, 'Normal noise scale shape is not compatible with provided action dimension!'
            else:
                self.noise_cov = np.diag(self.noise_scale)
            self.noise_std = compute_cholesky_if_possible(self.cov_noise, type_jitter='add')
            self.noise = MultivariateNormalTriL(self.noise_mean, self.noise_std)
        
        # set exponential decay scaling params
        self.decay_noise_steps = self.config['decay_noise_steps']
        self.decay_noise_rate = self.config['decay_noise_rate']


    def _set_checkpoints(self):
        
        # set checkpoints
        self.checkpoint_actor = tf.train.Checkpoint(model=self.actor_model)
        self.manager_actor = tf.train.CheckpointManager(self.checkpoint_actor, self.model_save_paths['actor_path'], max_to_keep = self.max_to_keep)

        self.checkpoint_critic = tf.train.Checkpoint(model=self.critic_model)
        self.manager_critic = tf.train.CheckpointManager(self.checkpoint_critic, self.model_save_paths['critic_path'], max_to_keep = self.max_to_keep)

        self.checkpoint_target_actor = tf.train.Checkpoint(model=self.target_actor_model)
        self.manager_target_actor = tf.train.CheckpointManager(self.checkpoint_target_actor, self.model_save_paths['target_actor_path'], max_to_keep = self.max_to_keep)

        self.checkpoint_target_critic = tf.train.Checkpoint(model=self.target_critic_model)
        self.manager_target_critic = tf.train.CheckpointManager(self.checkpoint_target_critic, self.model_save_paths['target_critic_path'], max_to_keep = self.max_to_keep)


    def _build_actor_network(self, name='actor'):
        # state is the input to the network
        input_layer = tf.keras.Input(shape=self.state_shape)
        net = tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(4,4), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_actor)(input_layer)
        net = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(3,3), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_actor)(net)
        net = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(3,3), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_actor)(net)
        net = tf.keras.layers.Flatten()(net)

        net = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_actor)(net)
        
        # get the actions
        if self.action_dim == 2:
            act = tf.keras.layers.Dense(self.action_dim, activation='tanh', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_actor)(net)
        elif self.action_dim in [3,4]:
            act = tf.keras.layers.Dense(self.action_dim, activation='sigmoid', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_actor)(net)

        # build actor model
        actor_model = tf.keras.Model(inputs=input_layer, outputs=act, name=name)
        return actor_model


    def _build_critic_network(self, name='critic'):
        # state is the input to the network
        input_layer_state = tf.keras.Input(shape=self.state_shape)
        net_state = tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(4,4), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_critic)(input_layer_state)
        net_state = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(3,3), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_critic)(net_state)
        net_state = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(3,3), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_critic)(net_state)
        net_state = tf.keras.layers.Flatten()(net_state)

        # pass action thru network
        input_layer_action = tf.keras.Input(shape=(self.action_dim, ))
        net_action = input_layer_action

        # concatenate action and state outs, pass thru the network
        net_together = tf.keras.layers.Concatenate()([net_state, net_action])
        net_together = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_critic)(net_together)
        net_together = tf.keras.layers.Dense(32, activation='relu', kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_critic)(net_together)
        
        # get the output
        out = tf.keras.layers.Dense(1, kernel_initializer=self.initializer_layer, kernel_regularizer=self.rf_critic)(net_together)

        # build critic model
        critic_model = tf.keras.Model(inputs=[input_layer_state, input_layer_action], outputs=out, name=name)
        return critic_model
                

    def generate_noise(self): 
        if self.noise_type == 'ou':
            return self.noise.generate()
        else:
            return self.noise.sample(sample_shape=1).numpy().reshape(-1, )


    def get_action(self, state):
        # if len(state.shape) == 3:
        #     num_pix_w, num_pix_h, num_ch = state.shape
        #     state = (state.reshape(1, num_pix_w, num_pix_h, num_ch) - np.min(state)) / (np.max(state) - np.min(state))
        
        state = process_image(state)

        # get the action from actor network
        action_before = self.actor_model(np.expand_dims(state, axis=0), training=False).numpy()
        action_before = action_before[0] + self.generate_noise()
        
        if self.action_dim == 2:
            action = [action_before[0], action_before[1].clip(0, 1),  -action_before[1].clip(-1, 0)]
        elif self.action_dim == 3:
            action = [2*action_before[0] - 1, action_before[1], action_before[2]]
        elif self.action_dim == 4:
            right_left = 1 if action_before[-1] > 0.5 else -1
            action = [action_before[0], action_before[1], action_before[2]]
            action[0] *= right_left

        action = np.clip(np.array(action), a_min=self.env.action_space.low, a_max=self.env.action_space.high)
        return action, action_before


    @tf.function
    def update_target_network_params(self):
        for v, t in zip(self.critic_model.trainable_variables, self.target_critic_model.trainable_variables):
            t.assign(t * (1-self.tau) + v * self.tau)

        for v, t in zip(self.actor_model.trainable_variables, self.target_actor_model.trainable_variables):
            t.assign(t * (1-self.tau) + v * self.tau)
    

    @tf.function
    def update_network_params(self, states, actions, rewards, next_states, terminals):
        # update critic model
        with tf.GradientTape(persistent=True) as tape:
            next_actions = self.target_actor_model(next_states, training=True)
            next_q_vals = self.target_critic_model([next_states, next_actions], training=True)
            y = rewards + self.gamma * tf.multiply((1-terminals),  next_q_vals) #

            loss_critic = tf.math.reduce_mean( tf.square( self.critic_model([states, actions], training=True) - y ) ) + tf.reduce_sum(self.critic_model.losses)

        critic_gradients = tape.gradient(loss_critic, self.critic_model.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(critic_gradients, self.critic_model.trainable_variables))

        # update actor model
        with tf.GradientTape(persistent=True) as tape:
            q_vals = self.critic_model([states, self.actor_model(states, training=True)], training=True)
            loss_actor = -tf.math.reduce_mean(q_vals) + tf.reduce_sum(self.actor_model.losses)

        actor_gradients = tape.gradient(loss_actor, self.actor_model.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(actor_gradients, self.actor_model.trainable_variables))


    def train_step(self):
        # sample the batch
        states, actions, rewards, next_states, terminals = self.memory.sample(self.batch_size)
        
        num_pix_h, num_pix_w, num_ch = states.shape[-3], states.shape[-2], states.shape[-1]
        states = tf.constant(states.reshape(-1, num_pix_h, num_pix_w, num_ch), dtype=tf.float32)
        next_states = tf.constant(next_states.reshape(-1, num_pix_h, num_pix_w, num_ch), dtype=tf.float32)
        actions = tf.constant(actions, dtype=tf.float32)
        rewards = tf.constant(rewards, dtype=tf.float32)
        terminals = tf.cast(tf.constant(terminals, dtype=tf.bool), dtype=tf.float32)

        # update networks
        self.update_network_params(states, actions, rewards, next_states, terminals)
        self.update_target_network_params()
     

    def save_models(self):
        self.manager_actor.save()
        self.manager_critic.save()
        self.manager_target_actor.save()
        self.manager_target_critic.save()


    def reset_update_noise(self, ep_num):
        if self.noise_type == 'ou':
            self.noise.reset()
            self.noise_std *= decayed_learning_rate(1, ep_num, self.decay_noise_steps, self.decay_noise_rate, staircase=True)
        else:
            updated_noise_scale = decayed_learning_rate(self.noise_scale, ep_num, self.decay_noise_steps, self.decay_noise_rate, staircase=True)
            self.noise_cov = updated_noise_scale * np.eye(self.action_dim, dtype=tf.float32)
            self.noise_std = compute_cholesky_if_possible(self.cov_noise, type_jitter='add')
            self.noise = MultivariateNormalTriL(self.noise_mean, self.noise_std)


    def update_network_lrs(self, ep_num):
        self.optimizer_actor.learning_rate.assign(decayed_learning_rate(self.init_lr_actor, ep_num, self.decay_lr_steps, self.decay_lr_rate, staircase=True))
        self.optimizer_critic.learning_rate.assign(decayed_learning_rate(self.init_lr_critic, ep_num, self.decay_lr_steps, self.decay_lr_rate, staircase=True))


    def create_writer(self):
        summary_path_main = '{}/graphs'.format(self.config['directory_to_save'])
        if not os.path.isdir(summary_path_main):
            os.mkdir(summary_path_main)
        summary_path = '{}/{}'.format(summary_path_main, 'DDPG')
        if not os.path.isdir(summary_path):
            os.mkdir(summary_path)
        
        self.writer = tf.summary.create_file_writer(summary_path)


    def def_summary(self, avg_reward, step_reward, step, prefix):
        """ 
        Add training values to a TF Summary object for Tensorboard
        """
        if not hasattr(self, 'writer'):
            return 

        with self.writer.as_default(): 
            tf.summary.scalar(name=prefix + 'avg_reward', data=tf.constant(avg_reward, dtype=tf.float32), step=step)
            tf.summary.scalar(name=prefix + 'step_reward', data=tf.constant(step_reward, dtype=tf.float32), step=step)

    







