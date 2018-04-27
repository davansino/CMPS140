# INITIALIZATION: libraries, parameters, network...

from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves
from keras.models import load_model      # For loading models


import random     # For sampling batches from the observations
import numpy as np
import gym                                # To train our network
#env = gym.make('MountainCar-v0')          # Choose game (any in the gym should work)
env = gym.make('SpaceInvaders-v0')

import os
cwd = os.getcwd()

# Load any previous models
model = load_model(cwd + '/my_model.h5')
print ('Model loaded')

# Play!
#-----------------------------------------------------------------
while 1:
    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)
    done = False
    tot_reward = 0.0
    while not done:
        env.render()                    # Uncomment to see game running
        Q = model.predict(state)
        action = np.argmax(Q)
        observation, reward, done, info = env.step(action)
        obs = np.expand_dims(observation, axis=0)
        state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
        tot_reward += reward
    print('Game ended! Total reward: {}'.format(tot_reward))

