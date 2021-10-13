# This code is originated from https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN
# Also uses https://github.com/pythonlessons/Reinforcement_Learning/blob/master/03_CartPole-reinforcement-learning_Dueling_DDQN/Cartpole_Double_DDQN.py
import gym
import os, sys
import random
import numpy as np
from collections import deque
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Add
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow.keras.backend as K
from model_tester import TesterAgent
from gym_multi_car_racing import MultiCarRacing
import torch


ACTIONS  = [
            (-1, 0.6, 0.2), (0, 1, 0.2), (1, 0.6, 0.2), #           Action Space Structure
            (-1, 0.6,   0), (0, 1,   0), (1, 0.6,   0), #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
           ]

def process_state_image(state, car_id):
    state = state[car_id, ...]
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def add_frame(s_new, s_stack):
    # s_stack = torch.roll(s_stack, shifts=-1, dims=1)
    # s_stack[:, -1, :, :] = s_new
    s_stack[:,:,:-1] = s_stack[:,:,1:]
    s_stack[:,:,-1] = s_new
    return s_stack
    
class DDQNTesterAgent(TesterAgent):
    def __init__(self,
                 model_path='..\\algorithms\dueling\images\save\\26_09_15_27\episode_500.h5',
                 car_id=0,
                 num_frames=4,
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
        input = Input(shape=(96, 96, self.num_frames), name='input')
        x = Conv2D(filters=6, kernel_size=(7,7), strides=3, activation='relu')(input)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(filters=12, kernel_size=(4,4), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(len(self.actions) + 1, activation='linear')(x)
        output = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(len(self.actions),))(x)

        model = Model(inputs = input, outputs = output, name='Dueling_DDQN_model')
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
        model.summary()
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
