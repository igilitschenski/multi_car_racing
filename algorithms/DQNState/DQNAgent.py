import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
import keras

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9975
        self.learning_rate = 0.0013
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.T = np.float64(50)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Flatten(input_shape=(1,self.state_size)))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        act_values = self.model.predict(state).astype('float64')
        exp_values = np.exp(act_values[0]/self.T).astype('float64')
        rand = np.random.rand()
        total = np.float64(0)
        for i in range(self.action_size):
            if np.sum(exp_values) == 0 or np.sum(exp_values) == np.inf:
                return np.argmax(act_values[0])
            total += (exp_values[i]/np.sum(exp_values))
            if (rand < total):
                return i
        return np.argmax(act_values[0])

    def getq_prob(self, state):
        #if np.random.rand() <= self.epsilon:
        #    return random.randrange(self.action_size)
        act_values = self.model.predict(state).astype('float64')

        q_val = np.exp(act_values/self.T)
        divisor = np.sum(q_val[0])
        if(divisor == 0):
            k = np.argmax(act_values[0])
            q_val[0, k] = 1
        elif(divisor == np.inf):
            for i in range(len(act_values[0])):
                q_val[0, i] = 0
            q_val[0, np.argmax(act_values[0])] = 1

        else:
            for i in range(len(act_values[0])):
                q_val[0,i] = (q_val[0,i]/divisor)
        #q_val[0] = np.nan_to_num(q_val[0])

        return q_val  # returns action

    def getq_prob_batch(self, state):
        #if np.random.rand() <= self.epsilon:
        #    return random.randrange(self.action_size)
        act_values = self.model.predict(state).astype('float64')
        q_val = np.exp(act_values/self.T)
        row_sums = q_val.sum(axis=1, keepdims=True)
        if(np.any(row_sums[:,0] == 0)):
            inds = np.asarray(np.where(row_sums[:,0] == 0))[0]
            for tval in range(inds.shape[0]):
                tmpint = inds[tval]
                k = np.argmax(act_values[tmpint])
                q_val[tmpint,k] = 1
                row_sums[tmpint,0] = 1
        elif(np.any(row_sums[:,0] == np.inf)):
            inds = np.asarray(np.where(row_sums[:, 0] == np.inf))[0]
            for tval in range(inds.shape[0]):
                tmpint = inds[tval]
                q_val[tmpint] = np.zeros((self.action_size,))
                q_val[tmpint, np.argmax(act_values[tmpint])] = 1
                row_sums[tmpint, 0] = 1
        q_val/= row_sums
        #q_val[0] = np.nan_to_num(q_val[0])

        return q_val  # returns action

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state)[0]
            if not done:
                a = np.argmax(self.target_model.predict(next_state)[0])
                t = self.target_model.predict(next_state)[0]
                #t = self.model.predict(next_state)[0]
                target[action] = reward + self.gamma * t[a]
            else:
                target[action] = reward
            X[i], Y[i] = state, target

        X = X.reshape((batch_size,1,self.state_size))
        Y = Y.reshape((batch_size,self.action_size))
        self.model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
