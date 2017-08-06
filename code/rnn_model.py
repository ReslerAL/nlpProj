'''
Created on Aug 6, 2017

@author: alon
'''

import random
from utils import *

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
        batch = [(key, sample_consistent(key), sample_inconsistent(key)) for key in batch_keys]

        return batch
