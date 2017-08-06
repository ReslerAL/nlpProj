'''
Created on Aug 6, 2017

@author: alon
'''



class rnn_model:
    def __init__(self, data, batchSize):
        self.data = data
        self.keys = []
        self.batch_size
        
    def generateBatch(self, batchSize):
        if len(self.keys) == 0:
            self.keys  += self.data.keys()
        