'''
Created on Aug 6, 2017

@author: alon
'''

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
from utils import *
from numpy import float32
from tensorflow.contrib.batching.ops.gen_batch_ops import batch

class Rnn_model:
    
    #raw_data is data structure contain the data as a natural language
    #self.data will contain the data as embedding vectors
    def __init__(self, raw_data, config):
        self.raw_data = raw_data
        self.keys = self.raw_data.keys() #this is the questions ids
        self.vocab = rawDataToVocabulary(raw_data)
        self.data = self.processDate(raw_data)
        self.vocab_size = len(self.vocab)
        self.batch_size = config['batch_size']
        self.embedding_dim = config['embedding_dim']
        
        #the embedding part. We will learn it as part of the model
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))
        
        #the main lstm cell. The second true argument sets peephole like the model used by Weiting
        self.lstm_cell = tf.contrib.rnn.LSTMCell(self.embedding_dim, True)
        
        #init C_0 - the state of the lstm_cell at the beginning
        self._initial_state = self.lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        
    
    def build_model(self):
        
        #input placeholder - each input is a list of embedding indices represent the sentence 
        x = tf.placeholder(tf.int32, [None])
        z_con = tf.placeholder(tf.int32, [None])
        z_incon = tf.placeholder(tf.int32, [None])
        
        #transfer the input to lost embedding vectors we can feed the net with
        x_embeds = tf.nn.embedding_lookup(self.embeddings, x)
        z_con_embeds = tf.nn.embedding_lookup(self.embeddings, z_con)
        z_incon_embeds = tf.nn.embedding_lookup(self.embeddings, z_incon)
        
              
        
        
        loss = 0
        return loss   
        
    
    """
    convert the data to the same structure just instead of words in each place
    we will hold the indices of the word embeddings
    """   
    def processDate(self, raw_data):
        dic = {}
        for qstn_id in raw_data:
            if qstn_id not in dic:
                dic[qstn_id] = [toEmbeddingList(raw_data[qstn_id][0].split(), self.vocab), [], []]
            for con_form in raw_data[qstn_id][1]:
                dic[qstn_id][1].append(toEmbeddingList(con_form.split(), self.vocab))
            for con_form in raw_data[qstn_id][2]:
                dic[qstn_id][2].append(toEmbeddingList(con_form.split(), self.vocab))
        return dic
        
          
    def generateBatch(self):
        def sample_consistent(key):
            return random.sample(self.data[key][1], 1)[0]
        def sample_inconsistent(key):
            return random.sample(self.data[key][2], 1)[0]

        batch_keys = random.sample(self.keys, self.batch_size)
        batch = [(self.data[key][0], sample_consistent(key), sample_inconsistent(key)) for key in batch_keys]
        return batch
    
    def apply(self, sent):
        '''
        calculate the model output for the given sent
        '''
        return 0
    
if __name__ == '__main__':
    embeddings = tf.Variable(tf.random_uniform([5, 2], -1.0, 1.0))
    x = tf.placeholder(tf.int32, [None])
    z = tf.nn.embedding_lookup(embeddings, x)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        res, embds = sess.run([z, embeddings], feed_dict={x : [1,3]})
        print(embds)
        print(res)
    
    
    
    
    
    
    
    
    

