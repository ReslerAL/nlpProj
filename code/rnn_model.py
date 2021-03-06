'''
Created on Aug 6, 2017

@author: alon
'''
 
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
import utils
from utils import *
from numpy import float32
from tensorflow.contrib.batching.ops.gen_batch_ops import batch
import time
import  math
import os
import pickle
from simple_evaluator import Evaluator


class Rnn_model:
    
    #raw_data is data structure contain the data as a natural language
    #self.data will contain the data as embedding vectors
    def __init__(self, raw_data, config):
        self.conf = config
        self.raw_data = raw_data
        self.keys = self.raw_data.keys() #this is the questions ids
        self.vocab = utils.rawDataToVocabulary(raw_data)
        self.data = utils.processData(raw_data, self.vocab)
        self.vocab_size = len(self.vocab)
        self.batch_size = config['batch_size']
        self.embedding_dim = config['embedding_dim']
        self.delta = float(config['delta'])
        
        self.delta_vec = self.delta * tf.ones(self.batch_size, dtype=float32)
        
        self.zero_batch = tf.zeros(self.batch_size, tf.float32)
        
        #the embedding part. We will learn it as part of the model
        self.w = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0), name='W')
        self.w_init = tf.Variable(self.w.initialized_value(), name='W_init')
        
        self.extended_w = tf.concat((self.w, tf.zeros((1, self.embedding_dim))), axis=0)
        
        #the main lstm cell. The second true argument sets peephole like the model used by Weiting at el.
        self.lstm_cell = tf.contrib.rnn.LSTMCell(self.embedding_dim, True)
        
        #train input placeholder - each input is a list of embedding indices represent the sentence 
        self.x = tf.placeholder(tf.int32, [None, None], name='x')
        self.z_con = tf.placeholder(tf.int32, [None, None], name='z_con')
        self.z_incon = tf.placeholder(tf.int32, [None, None], name='z_incon')
        
        #transfer the input to lost embedding vectors we can feed the net with
        self.x_embeds = tf.nn.embedding_lookup(self.extended_w, self.x)
        z_con_embeds = tf.nn.embedding_lookup(self.extended_w, self.z_con)
        z_incon_embeds = tf.nn.embedding_lookup(self.extended_w, self.z_incon)
        
        #now we need placeholder for the sequence sizes - this is because the data is padded with dummy value
        #and we want the output to be correct (takes on the real end of the sequence, not containing the dummy value)
        self.x_seq_len = tf.placeholder(dtype=tf.int32, shape=(None), name='x_seq_len')
        self.z_con_seq_len = tf.placeholder(dtype=tf.int32, shape=(None))
        self.z_incon_seq_len = tf.placeholder(dtype=tf.int32, shape=(None))
               
        #now run the lstm to get the outputs. Note that output shape is batch_size * sequence_len * embedding_dim
        #sequence len is the max len of a sentence in the batch
        #last_state is a tuple with the following values:
        # last_state.h - this is the last output according to the sequence length parameter (this is h_t)
        # last_state.c - this is the last cell state according to the sequence length parameter (this is c_t)
        
        _, self.x_last_state = tf.nn.dynamic_rnn(cell=self.lstm_cell, dtype=tf.float32, 
                                                 sequence_length=self.x_seq_len, inputs=self.x_embeds)
        
        _, self.z_con_last_state = tf.nn.dynamic_rnn(cell=self.lstm_cell, dtype=tf.float32, 
                                                sequence_length=self.z_con_seq_len, inputs=z_con_embeds)
          
        _, self.z_incon_last_state = tf.nn.dynamic_rnn(cell=self.lstm_cell, dtype=tf.float32, 
                                                 sequence_length=self.z_incon_seq_len, inputs=z_incon_embeds)
                
        #compute the loss from the correct outputs of the batches
        self.x_z_con_sim = self.cosineSim(self.x_last_state.h, self.z_con_last_state.h)
        self.x_z_incon_sim = self.cosineSim(self.x_last_state.h, self.z_incon_last_state.h)
                
        self.loss_vec = tf.maximum(self.zero_batch, self.delta - self.x_z_con_sim + self.x_z_incon_sim)
        self.loss1 = tf.reduce_mean(self.loss_vec)
        self.loss2 = self.getRegularizationLoss()
        self.loss = self.loss1 + self.loss2        
        
        #evaluation hack to enable tensorflow's Java api support
        self.y1 = tf.add(self.x_last_state.h, self.x_last_state.h, name='y')
        self.y2 = tf.subtract(self.y1, self.x_last_state.h, name='embd')
        
        #add subgrapgh that just calculate the embeddings of sinle sentence
        self.input = tf.placeholder(tf.int32, [None, None])
        embeds = tf.nn.embedding_lookup(self.extended_w, self.input)
        _, res = tf.nn.dynamic_rnn(cell=self.lstm_cell, dtype=tf.float32, inputs=embeds)
        self.eval = res.h
        
        self.saver = tf.train.Saver()
        
    def getRegularizationLoss(self):
        vars = tf.trainable_variables()
        W_init, W, lstm_params = None, None, None
        for var in vars:
            if var.name == 'W:0':
                W = var
            elif var.name == 'W_init:0':
                W_init = var
            elif 'kernel' in var.name:
                lstm_params = var
        self.reg1 = self.conf['lambda_c']*2*tf.reduce_sum(tf.nn.l2_loss(lstm_params))
        self.reg2 = self.conf['lambda_w']*2*tf.reduce_sum(tf.nn.l2_loss(W_init-W))
        return self.reg1 + self.reg2
    
    def apply(self, sent):
        assert self.sess != None
        embd_sent = utils.toEmbeddingList2(sent.split(), self.vocab)
        return self.apply_embd(embd_sent)
    
    def apply_embd(self, embd_sent):
        res = self.sess.run(self.eval, feed_dict={self.input:[embd_sent]})
        return res[0]
    
    def apply_batch(self, inputs, inputes_lengths):
        res = self.sess.run(self.x_last_state.h, feed_dict={self.x: inputs, self.x_seq_len: inputes_lengths})
        return res
    
    """
    saving the model and the configuration
    each model will be save in separate directory named with time in milliseconds
    """
    def saveModel(self, session):
        dirr = './saved/' + str(int(round(time.time() * 1000))) + '/'
        full_name = dirr + 'lstm-model'
        os.makedirs(dirr,  exist_ok=True)
        self.saver.save(session, full_name)
        #save the configuration
        confFile = dirr + 'configuration.p'
        pickle.dump(self.conf, open(confFile, 'wb'))        
        #also save as text for us
        f = open(dirr + 'configuration.txt', 'w')
        f.write(str(self.conf))
        f.close()
        builder = tf.saved_model.builder.SavedModelBuilder(dirr + '/pbmodel' )
        builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING])
        builder.save()
        print('model saved to {} directory'.format(dirr))
        return dirr

        
    def load(self, model_dir, session):
        loader = tf.train.import_meta_graph(model_dir + 'lstm-model' + '.meta')
        loader.restore(session, tf.train.latest_checkpoint(model_dir))
        f = open(model_dir + 'configuration.txt', 'r')
        print('loading model with configuration:\n', f.read())
        f.close()
        self.sess = session #save the session for evaluation
    
    def setSession(self, session):
        self.sess = session
    
    """
    x and y are batch (array) of vectors (actually tensors). return batch (array) of the cossine similarite
    between x and y (actually a tensor) 
    """
    def cosineSim(self, x, y):
        x_nl = tf.nn.l2_normalize(x, 1)
        y_nl = tf.nn.l2_normalize(y, 1)
        return tf.reduce_sum(tf.multiply(x_nl, y_nl), axis=1)
    
    
    """
    return 3 batches. One with the questions and the other two with the consistent and inconsistent forms
    return also the sequence lengths before padding for each batch
    """ 
    #with probability q get the best one according to the model otherwise sample randomly 
    def generateBatch2(self, q):
                
        batch_keys = random.sample(self.keys, self.batch_size)
        coins = np.random.binomial(1, q, self.batch_size)
        batch_x = [self.data[key][0] for key in batch_keys]
        batch_z_con = [self.getSample(self.data[key][0], self.data[key][1], coins[i]) for i, key in enumerate(batch_keys)]
        batch_z_incon = [self.getSample(self.data[key][0], self.data[key][2], coins[i]) for i, key in enumerate(batch_keys)]
        return [(self.padBatch(batch_x), self.getSequenceLength(batch_x)),
                (self.padBatch(batch_z_con), self.getSequenceLength(batch_z_con)),
                (self.padBatch(batch_z_incon), self.getSequenceLength(batch_z_incon))]
        
       
    def getSample(self, ques, forms, coin):
        if coin == 0:
            return random.sample(forms, 1)[0]
        q_embd = self.apply_embd(ques)
        forms_embds = [self.apply_embd(form) for form in forms]
        similarities = [Evaluator.cosine_sim(q_embd, canonical_embedding) for canonical_embedding in forms_embds]
        idx = np.argmax(similarities)
        return forms[idx]
    
    
    def getBestsSamples(self, key):
        num_con = len(self.data[key][1])
        alls = [self.data[key][0]] + self.data[key][1] + self.data[key][2]
        lenghts = self.getSequenceLength(alls)
        embeddings = self.apply_batch(self.padBatch(alls), lenghts)
        qestion_embedding = embeddings[0]
        consistents = embeddings[1:num_con+1]
        inconsistents = embeddings[num_con+1:]
        consistent_similarities = [Evaluator.cosine_sim(qestion_embedding, consistent) for consistent in consistents]
        inconsistent_similarities = [Evaluator.cosine_sim(qestion_embedding, inconsistent) for inconsistent in inconsistents]
        idx_con= np.argmax(consistent_similarities)
        idx_incon = np.argmax(inconsistent_similarities)
        return self.data[key][1][idx_con], self.data[key][2][idx_incon]

    def getRandomSamples(self, key):
        return random.sample(self.data[key][1], 1)[0], random.sample(self.data[key][2], 1)[0]
    
    def generateBatch(self, q):
                
        batch_keys = random.sample(self.keys, self.batch_size)
        coins = np.random.binomial(1, q, self.batch_size)
        batch_x = [self.data[key][0] for key in batch_keys]
        
        z_con_samples   = [0]*self.batch_size
        z_incon_samples = [0]*self.batch_size
        
        for i in range(self.batch_size):
            if coins[i] == 0:
                con_sample, incon_sample = self.getRandomSamples(batch_keys[i])
            else:
                con_sample, incon_sample = self.getBestsSamples(batch_keys[i])
            z_con_samples[i] = con_sample
            z_incon_samples[i] = incon_sample
        
        return [(self.padBatch(batch_x), self.getSequenceLength(batch_x)),
                (self.padBatch(z_con_samples), self.getSequenceLength(z_con_samples)),
                (self.padBatch(z_incon_samples), self.getSequenceLength(z_incon_samples))]
    
    """
    pad all the sequence in the batch according to the max sequence length.
    pad it with self.vocab_size so the value is the index of the constant dummy embedding
    """
    def padBatch(self, batch):
        result = [0]*len(batch)
        max_len = getMaxLength(batch)
        for i, sample in enumerate(batch):
            result[i] = np.pad(sample, (0,max_len-len(sample)), 'constant', constant_values=self.vocab_size)
        return result
    
    def getSequenceLength(self, batch):
        result = [0]*len(batch)
        for k, sample in enumerate(batch):
            result[k] = len(sample)
        return result

    

