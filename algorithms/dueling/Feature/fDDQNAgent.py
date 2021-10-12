# This code is originated from https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN
# Also uses https://github.com/pythonlessons/Reinforcement_Learning/blob/master/03_CartPole-reinforcement-learning_Dueling_DDQN/Cartpole_Double_DDQN.py

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

class CarRacingDQNAgent:
    def __init__(
        self,
        #action_space    = [
           # (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
           # (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
           # (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
           # (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        #],
        action_space    = [
            (-1, 0.6, 0.2), (0, 1, 0.2), (1, 0.6, 0.2), #           Action Space Structure
            (-1, 0.6,   0), (0, 1,   0), (1, 0.6,   0), #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ],
        frame_stack_num = 1,
        #frame_stack_num = 1,
        memory_size     = 5000,
        gamma           = 0.95,  # discount rate
        epsilon         = 1.0,   # exploration rate
        epsilon_min     = 0.1,
        epsilon_decay   = 0.9999,
        learning_rate   = 0.001
        #learning_rate   = 0.001
    ):
        self.action_space    = action_space
        self.frame_stack_num = frame_stack_num
        self.memory          = deque(maxlen=memory_size)
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        self.model           = self.build_model()
        self.target_model    = self.build_model()
        self.update_target_model()

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        input = Input(shape=(48, self.frame_stack_num), name='input')
        x = Flatten()(input)
        x = Dense(128, activation='relu')(x) 

        # - - - Option 1 - - - 
        # value_fc = Dense(128, activation='relu')(x)
        # value = Dense(1, activation='relu')(value_fc)
        # value = Lambda(lambda v: K.expand_dims(v[:, 0], -1), output_shape=(len(self.action_space),))(value)
        # advantage_fc = Dense(128, activation='relu')(x)
        # advantage = Dense(len(self.action_space), activation='relu')(advantage_fc)
        # advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),  output_shape=(len(self.action_space),))(advantage)
        # output = Add()([value, advantage])
        # - - - - - - - - - - -
        # - - - Option 2 - - - 
        x = Dense(256, activation='relu')(x)
        x = Dense(len(self.action_space) + 1, activation='linear')(x)
        output = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(len(self.action_space),))(x)
        # - - - - - - - - - - -

        model = Model(inputs = input, outputs = output, name='Dueling_DDQN_model')
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.epsilon:
            act_values = self.model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        # - - - -  - - Option 1 (slower) - - - - - - - - 
        # train_state = []
        # train_target = []
        # for state, action_index, reward, next_state, done in minibatch:
        #     target = self.model.predict(np.expand_dims(state, axis=0))[0]
        #     if done:
        #         target[action_index] = reward
        #     else:
        #         t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
        #         target[action_index] = reward + self.gamma * np.amax(t)
        #     train_state.append(state)
        #     train_target.append(target)
        # - - - - - - - - - - - - - - - - - - - 

        # - - - - - - Option 2 (faster) - - - -
        train_state = np.zeros((batch_size,48,self.frame_stack_num))
        train_next_state = np.zeros((batch_size,48,self.frame_stack_num))
        action, reward, done = [], [], []
        for i in range(batch_size):
            train_state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            train_next_state[i] = minibatch[i][3]            
            done.append(minibatch[i][4])
        train_target = self.model.predict(train_state)
        train_target_next = self.model.predict(train_next_state)
        train_target_val = self.target_model.predict(train_next_state)
        for i in range(len(minibatch)):
            if done[i]:
                train_target[i][action[i]] = reward[i]
            else:
                a = np.argmax(train_target_next[i])
                train_target[i][action[i]] = reward[i] + self.gamma * train_target_val[i][a]
         # - - - - - - - - - - - - - - - - - - - 

        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)

    @staticmethod
    # Is it needed for feature version?
    def process_state_feats(state):
        #state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = state.astype(float)
        #state /= 255.0
        return state

    @staticmethod
    def generate_state_from_queue(deque):
        frame_stack = np.array(deque)
        # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
        if len(frame_stack.shape)>2:
            return np.transpose(frame_stack, (1, 2, 0))
        else:
            return np.transpose(frame_stack, (1, 0))
