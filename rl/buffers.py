import math
import random
import numpy as np
import torch



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def display(self):
        print(self.capacity, self.buffer)

    def sample(self, batch_size):
        try:
            batch = random.sample(self.buffer, batch_size)
        except:
            print('buffer: ', self.buffer)
        try:
            state, action, reward, next_state, done = map(np.stack,
                                                        zip(*batch))  # stack for each element
        except:
            print('shape: ', batch.shape)
            print('*batch: ', *batch)
            print('state: ', state)
            print('action: ', action)
            print('reward: ', reward)
            print('done: ', done)
            
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)
