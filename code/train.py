'''
Created on Aug 9, 2017

@author: alon
'''

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
from rnn_model import *
from utils import *
import time
import sys

config = {
    'batch_size' : 15,
    'embedding_dim' : 10,
    'data_file' : '../project_train_data.csv',
    'delta' : 1.0,
    'learning_rate' : 0.001,
    'num_epocs' : 1,
    'data_size'  : 369488,
    'print_freq' : 10
    }

print('run configuration:', config)
print("getting the data... ")      
raw_data = fileToDic(config['data_file'])

print("building the model...")
model = Rnn_model(raw_data, config)
optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(model.loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
  
batches_to_run = 10#config['num_epocs']*config['data_size'] // config['batch_size']
print(str.format('Starting to train. {} batches to go...', batches_to_run))
count = 1
losses = []
start = time.time()

# x_example = [[3,2,54,73],[3,5,12080, 12080]]
# x_len = [4, 2]
#  
# res = sess.run(model.x_last_state, feed_dict={model.x : x_example, model.x_seq_len : x_len})
# print('res.h', res.h, '\n')


while count <= batches_to_run:
    batch = model.generateBatch()
    dict = {model.x : batch[0][0], model.x_seq_len : batch[0][1],
                      model.z_con : batch[1][0], model.z_con_seq_len : batch[1][1],
                      model.z_incon : batch[2][0], model.z_incon_seq_len : batch[2][1]}
    _, loss = sess.run([optimizer, model.loss], feed_dict=dict)
    losses.append(loss)
    
    if count % config['print_freq'] == 0:
        timer = time.time() - start
        print(str.format('{} batches run. {} batches left', count, batches_to_run - count))
        print(str.format('{} last batches average loss = {}', config['print_freq'], np.average(losses[-config['print_freq']])))
        h,m,s = formatTimer(timer)
        print(str.format('Train time: {}:{}:{}\n', h, m, s))
        
    count += 1

print("training finished. Saving the model...")
model.saveModel(sess)
print("************      The End       ***************")

# model.load('1502789449262', sess)
# res = sess.run(model.x_last_state, feed_dict={model.x : x_example, model.x_seq_len : x_len})
# print('res.h after', res.h, '\n')
# model.saveModel(sess)


#     r1, r2, r3, r4, r7,r8,r9 = sess.run([model.loss, model.loss_vec, model.x_z_con_sim, model.x_z_incon_sim, 
#                                    model.x_last_state, model.z_con_last_state, model.z_incon_last_state],
#                               feed_dict={model.x : batch[0][0], model.x_seq_len : batch[0][1],
#                                          model.z_con : batch[1][0], model.z_con_seq_len : batch[1][1],
#                                          model.z_incon : batch[2][0], model.z_incon_seq_len : batch[2][1]})
# #     res = sess.run(result, feed_dict={x : [1,2,3]})
#     print("loss is:", r1, '\n')
#     print("loss vector is:", r2, '\n')
#     print("x_z_con_sim is:", r3, '\n')
#     print("x_z_incon is:", r4, '\n')
#     print('x outputs:' , r7, '\n')
#     print('z_con outputs:' , r8, '\n')
#     print('z_incon outputs:' , r9, '\n')

if __name__ == '__main__':
    print("in train main")
#     dic = {'1' : 'alon', '2' : "dani"}
#     for val in dic:
#         print(val)











