# INITIALIZATION: libraries, parameters, network...

import tensorflow as tf
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves
from keras.models import load_model      # For loading models
from keras.callbacks import TensorBoard

import random     # For sampling batches from the observations
import numpy as np
import gym                                # To train our network
#env = gym.make('MountainCar-v0')          # Choose game (any in the gym should work)
env = gym.make('SpaceInvaders-v0')

import os
cwd = os.getcwd()

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush() 

try:
    # Load any previous models
    model = load_model(cwd + '/my_model.h5')
    print ('Model loaded')
except:
    print ('Starting new model')
    # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
    model = Sequential()
    model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, kernel_initializer='uniform', activation='relu'))
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(env.action_space.n, kernel_initializer='uniform', activation='linear'))    # Same number of outputs as possible actions
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

log_path = './logs'
callback = TensorBoard(log_path)
callback.set_model(model)
train_names = ['train_loss']

# Parameters
D = deque()                                # Register where the actions will be stored
observetime = 5000                          # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7                              # Probability of doing a random move
gamma = 0.9                                # Discounted future reward. How much we care about steps further in time
mb_size = 50                               # Learning minibatch size
batch_num = 0
tot_reward = 0.0

# FIRST STEP: Keowing what each action does (Observing)
#-----------------------------------------------------------------
for j in range(0,10):
    observation = env.reset()                                                               # Game begins
    obs = np.expand_dims(observation, axis=0)                                               # (Formatting issues) Making the observation the first element of a batch of inputs
    state = np.stack((obs, obs), axis=1)
    done = False
    for t in range(observetime):
        env.render()                                                                        # Uncomment to see game running
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
        else:
            Q = model.predict(state)                                                        # Q-values predictions
            action = np.argmax(Q)                                                           # Move with highest Q-value is the chosen one
        observation_new, reward, done, info = env.step(action)                              # See state of the game, reward... after performing the action
        obs_new = np.expand_dims(observation_new, axis=0)                                   # (Formatting issues)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game
        D.append((state, action, reward, state_new, done))                                  # 'Remember' action and consequence
        state = state_new                                                                   # Update state
        tot_reward += reward

        if t % 100 == 0:
            print 'frames: {}'.format(t)
        if done:
            print('Game ended! Total reward: {}'.format(tot_reward))
            env.reset()                                                                     # Restart game if it's finished
            obs = np.expand_dims(observation, axis=0)                                       # (Formatting issues) Making the observation the first element of a batch of inputs
            state = np.stack((obs, obs), axis=1)
            tot_reward = 0.0

    print('Observing Finished')


    # SECOND STEP: Learning from the observations (Experience replay)
    #-----------------------------------------------------------------
    minibatch = random.sample(D, mb_size)                                                   # Sample some moves
    inputs_shape = (mb_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, env.action_space.n))

    for i in range(0, mb_size):
        print 'mb count: {}'.format(i)
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]

    # Build Bellman equation for the Q function
        inputs[i:i+1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)
        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

    # Train network to output the Q function
        logs = model.train_on_batch(inputs, targets)
        write_log(callback, train_names, logs, batch_num)
        batch_num += 1
    print('Learning Finished')
    model.save(cwd + '/my_model.h5')


# THIRD STEP: Play!
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
