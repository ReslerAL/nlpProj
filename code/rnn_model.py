'''
Created on Aug 6, 2017

@author: alon
'''

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
from utils import *


batch_size = 5
embedding_dim = 300


class Rnn_model:
    
    #the init is just for getting the data 
    def __init__(self, data):
        self.data = data
        self.keys = self.data.keys() #this is the questions ids
        self.embedding_indxs = fileToVocabulary(self.data)
        self.vocab_size = len(self.embedding_indxs)
        
    
    def build_model(self):
        
        #the main lstm cell. The second true argument sets peephole like the model used by Weiting
        lstm_cell = tf.contrib.rnn.LSTMCell(embedding_dim, True)
        
        
        
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
