'''
Created on Aug 6, 2017

@author: alon
'''

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from utils import *

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

embedding_dim = 300


class Rnn_model:
    
    def __init__(self, fileName):
        dic = fileToDic(fileName)
        self.dic = dic
        self.keys = self.dic.keys()
        
    def generateBatch(self, batchSize):
        def sample_consistent(key):
            return random.sample(self.dic[key][1], 1)[0]
        def sample_inconsistent(key):
            return random.sample(self.dic[key][2], 1)[0]

        batch_keys = random.sample(self.keys, batchSize)
        batch = [(self.dic[key][0], sample_consistent(key), sample_inconsistent(key)) for key in batch_keys]
        return batch
    
    def apply(self, sent):
        '''
        calculate the model output for the given sent
        '''
        return 0
