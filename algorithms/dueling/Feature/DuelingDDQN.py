# Code modified from 
# https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN/blob/master/CarRacingDQNAgent.py
# https://github.com/pythonlessons/Reinforcement_Learning/blob/master/03_CartPole-reinforcement-learning_Dueling_DDQN/Cartpole_Double_DDQN.py

import gym
import gym_multi_car_racing
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from collections import deque

class DuelingDDQN:
    def __init__(self,feature_size=32,action_size=12,learning_rate = 0.001, memory_size=5000 ):
        self.feature_size =feature_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.memory = deque(maxlen=memory_size)
        self.update_target_model()


    def build_model(self):
        # Neural Net for Deep-Q learning Model
        input = Input((self.feature_size,),name='input')
        x = Dense(256, activation='relu')(input)
        value = Dense(1)(x)
        value = Lambda(lambda s: keras.backend.expand_dims(s[:, 0], -1), output_shape=(self.action_size,))(value)
        advantage = Dense(self.action_size)(x)
        advantage = Lambda(lambda a: a[:, :] - keras.backend.mean(a[:, :], keepdims=True), output_shape=(self.action_size,))(advantage)
        output = Add()([value, advantage])

        model = Model(inputs = input, outputs = output, name='Dueling_DDQN_model')
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action_index, reward, next_state, done):
        self.memory.append((state, action_index, reward, next_state, done))

    def act(self, state, eps):
        if np.random.rand() > eps:
            state = np.reshape(state,-1) # For 1D state features. nor need for CNN
            act_values = self.model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randrange(self.action_size)
        return action_index

    def replay(self, batch_size, gamma):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            state = np.reshape(state,-1) # For 1D state features. no need for CNN
            next_state = np.reshape(next_state,-1) # For 1D state features. no need for CNN
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if done:
                target[action_index] = reward
            else:
                target_temp = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + gamma * np.amax(target_temp)
            train_state.append(state)
            train_target.append(target)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)


    def load_weights(self, filePath):
        self.model.load_weights(filePath)
        self.update_target_model()

    def save_weights(self, filePath, overwrite=False):
        self.model.save_weights(filePath, overwrite=overwrite)






