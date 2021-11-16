# This code is originated from https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN
import os, sys
import random
import numpy as np
from collections import deque

import gym
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from model_tester import TesterAgent
from gym_multi_car_racing import MultiCarRacing


# smoother 10 actions below
ACTIONS  = [(-1, 0.85, 0.15), (0, 0.85, 0.15), (1, 0.85, 0.15), (0, 1,   0), 
        (-1, 0, 0.1), (0, 0, 0.2), (1, 0, 0.1),
        (-1, 0,   0), (0, 0,   0), (1, 0,   0)]

def process_state_image(state, car_id):
    state = state[car_id, ...]
    #state = np.dot(state, [0.299, 0.587, 0.144])
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def add_frame(s_new, s_stack):
    s_stack[:,:,:-1] = s_stack[:,:,1:]
    s_stack[:,:,-1] = s_new
    return s_stack


class DQNTesterAgent(TesterAgent):
    def __init__(self,
                 model_path='algorithms\\DQN\\save\\ten_soft_action_acc_reward_true\\episode_1000.h5',
                 car_id=0,
                 num_frames=3,
                 learning_rate = 0.001,
                 actions=ACTIONS,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.learning_rate   = learning_rate
        self.car_id = car_id
        self.frame_buffer = None
        self.num_frames = num_frames
        self.actions = actions
        self.model = self.build_model()
        self.agent = self._load_model(model_path)

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', 
                                            input_shape=(96, 96, self.num_frames)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.actions), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate, epsilon=1e-7))
        return model

    def _load_model(self, model_path):
        self.model.load_weights(model_path)

    def _update_frame_buffer(self, new_frame):
        if self.frame_buffer is None:
            self.frame_buffer = np.repeat(new_frame[:,:,np.newaxis], self.num_frames, axis=2)            
        else:
            self.frame_buffer = add_frame(new_frame, self.frame_buffer)

    def state_to_action(self, state):
        """
        This function should take the most recent state and return the
        action vector in the exact same form it is needed for the environment.
        If you are using frame buffer see example in _update_frame_buffer
        how to take care of that.
        """
        state = process_state_image(state, self.car_id)
        self._update_frame_buffer(state)
        act_values = self.model.predict(np.expand_dims(self.frame_buffer, axis=0))
        action_index = np.argmax(act_values[0])
    
        return self.actions[action_index]

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