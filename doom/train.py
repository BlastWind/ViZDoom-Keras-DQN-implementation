#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from tqdm import trange
import tensorflow as tf
from tqdm import trange
import vizdoom as vzd
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
import itertools as it
from collections import deque

# Q-learning settings
learning_rate = 0.0001
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (96, 128)

# exploration 
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.998

batch_size = 64

def initialize_doom(game):
    game = vzd.DoomGame()
    game.load_config("basic.cfg")
    
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    
    game.set_window_visible(False)

    return game

def preprocess(img):
    img = img[:400]
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis = 0)

    return img

class DoomAgent: 
    def __init__(self, game):
        
        shoot = [0, 0, 1]
        left = [1, 0, 0]
        right = [0, 1, 0]
        actions = [shoot, left, right]
        self.actions = actions
        self.action_space = len(self.actions)
        self.exploration_rate = EXPLORATION_MAX

        # self.action_space = action_space
        self.memory = deque(maxlen=replay_memory_size)


        # two conv layers, flat, dense to 64, then dense to 8 
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size = (8,8), strides = (4, 4), padding='valid', activation='relu', input_shape=(96, 128, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size = (4,4), strides = (2, 2), padding='valid', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size = (4,4), strides = (2, 2), padding='valid', activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    def appendToMemory(self, current_state, next_state, action, reward, isterminal):
        self.memory.append((current_state, next_state, action, reward, isterminal))

    def get_action(self, current_state):
        if np.random.rand() < self.exploration_rate: 
            # In english: we choose to choose a random action when the exploration rate is high
                return random.randrange(self.action_space)
        else: 
            # if we don't explore, we use or "exploit" our q-table
            q_values = self.model.predict(current_state)
        
            # q_values might be double or single array
            action = np.argmax(q_values)
            return action
    def train_batch(self, epoch_num):
        
        def exploration_rate(epoch_num):
            """# Define exploration rate change over time"""
            start_eps = 1.0
            end_eps = 0.1
            const_eps_epochs = 0.1 * epochs  # 10% of learning time
            eps_decay_epochs = 0.6 * epochs  # 60% of learning time
    
            if epoch_num < const_eps_epochs:
                return start_eps
            elif epoch_num < eps_decay_epochs:
                # Linear decay
                return start_eps - (epoch_num - const_eps_epochs) / \
                                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
            else:
                return end_eps

        # Get a random minibatch from the replay memory and learns from it.
        if len(self.memory) > batch_size:
            batch = self.get_sample(batch_size, self.memory)
            batch = np.array(batch)
            # to get a training example from memory
            # memory[training example index]
            # to get the current state, memory[training example index][0]
            # to get the next state, memory[training example index][1]
            # to get reward, memory[training example index][2]
            # to get terminated or not, memory[training example index][3]   
            
            # select the first item of everysingle tuple in the list
            
            ## array manipulations 
            s2 = [batch[i][1] for i in range(63)]
            
            a = np.stack(s2, axis=1)
            a = a[0]
            
            s1 = [batch[i][0] for i in range(63)]
            b = np.stack(s1, axis = 1)
            b = b[0]
            
            q2 = np.argmax(self.model.predict(a), axis=1)
            target_q = self.model.predict(b)
            #print(target_q)
            
            actions = [batch[i][2] for i in range(63)]
            reward = [batch[i][3] for i in range(63)]
            
            isterminal_ = [batch[i][4] for i in range(63)]
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
            ones = np.ones((63, 1))
            isterminal_to_integer = np.expand_dims(np.array([int(isterminal_[i]) for i in range(63)]), axis=1)
            diff = ones-isterminal_to_integer
            q2 = np.expand_dims(q2, axis=1)
            placeholder =  reward + discount_factor * np.multiply(diff, q2)
            placeholder = placeholder[:, 0]
            target_q[np.arange(target_q.shape[0]), actions] = placeholder
            self.model.train_on_batch(b, target_q)
            
            
            self.exploration_rate = exploration_rate(epoch_num)
            
            
    
    def get_sample(self, sample_size, memory):
        i = random.sample(range(0, len(self.memory)), sample_size)
        newarray = [self.memory[index] for index in i]
        return newarray
'''
Initialize Doom Environment E
Initialize replay Memory M with capacity N (= finite capacity)
Initialize the DQN weights w
for episode in max_episode:
    s = Environment state
    for steps in max_steps:
         Choose action a from state s using epsilon greedy.
         Take action a, get r (reward) and s' (next state)
         Store experience tuple <s, a, r, s'> in M
         s = s' (state = new_state)
         
         Get random minibatch of exp tuples from M
         Set Q_target = reward(s,a) +  γmaxQ(s')
         Update w =  α(Q_target - Q_value) *  ∇w Q_value
'''

def train():
    game = vzd.DoomGame()
    game = initialize_doom(game)
    
    agent = DoomAgent(game)
    
    # get frame with game.get_state().screen_buffer
    # get reward with game.make_action(actions[a], frame_repeat)
    # take action with game.make_action
    
    # probably need a lot more
    
    epoch_reward_analysis = []
    for epoch_num in range(20):
        for episode in trange(2000, leave=False):
          
            epoch_reward = 0 
            game.new_episode()
            current_statea = game.get_state().screen_buffer
            current_state = preprocess(current_statea)
            
            while not game.is_episode_finished():
               
                action = agent.get_action(current_state)
                
                reward = game.make_action(agent.actions[action], 12)
                isterminal = game.is_episode_finished()
                next_state = preprocess(game.get_state().screen_buffer) if not isterminal else np.zeros((1, 96, 128, 1))
                
                
                agent.appendToMemory(current_state, next_state, action, reward, isterminal)
        
        
                # to get a training example from memory
                # memory[training example index]
                # to get the current state, memory[training example index][0]
                # to get the next state, memory[training example index][1]
                # to get reward, memory[training example index][2]
                # to get terminated or not, memory[training example index][3]
        
                current_state = next_state
            
            score = game.get_total_reward()
            epoch_reward += score
            if episode == 99: 
                print("our total reward over this epoch was : " + str(epoch_reward))
                epoch_reward_analysis.append(epoch_reward)
            if episode % 30 == 0: 
                print("for run" + str(episode) + ", our reward was: " + str(score))
            agent.train_batch(epoch_num)
            
    
    game.close()
    
    '''
    from matplotlib import pyplot as plt
    plt.imshow(current_statea, interpolation='nearest')
    plt.show()
    
    tests = current_statea[:400]
    plt.imshow(tests, interpolation='nearest')
    '''







