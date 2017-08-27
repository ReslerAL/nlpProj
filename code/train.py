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
import os
import argparse
import shutil


#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

config = {
    'batch_size' : 100,
    'embedding_dim' : 300,
    'data_file' : '/home/alon/workspace/nlp/project_train_data.csv',
    'delta' : 0.4,
    'learning_rate' : 0.0005,
    'num_epocs' : 10,
    'data_size'  : 369488,
    'lambda_c' :0.001,
    'lambda_w' : 1e-06,
    'print_freq' : 100
    }


parser = argparse.ArgumentParser()

parser.add_argument("-lr", help="learning rate")
parser.add_argument("-margin", help="the delta from the loss")
parser.add_argument("-batch_size", help="batch size to use")
parser.add_argument("-dim", help="the embeddings dimension ")
parser.add_argument("-lc", help="regularization of the lstm weights")
parser.add_argument("-lw", help="regularization of the embedding matrix")
parser.add_argument("-epocs", help="Number of epocs to run")

args = parser.parse_args()

if args.lr != None:
    config['learning_rate'] = float(args.lr)
if args.margin != None:
    config['delta'] = float(args.margin)
if args.batch_size != None:
    config['batch_size'] = int(args.batch_size)
if args.dim != None:
    config['embedding_dim'] = int(args.dim)
if args.lc != None:
    config['lambda_c'] = float(args.lc)
if args.lw != None:
    config['lambda_w'] = float(args.lw)
if args.epocs != None:
    config['num_epocs'] = int(args.epocs)
    

print('run configuration:', config)
print("getting the data... ")      
raw_data = fileToDic(config['data_file'])

print("building the model...")
model = Rnn_model(raw_data, config)
optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(model.loss)

batches_to_run = config['num_epocs']*config['data_size'] // config['batch_size']
print(str.format('Starting to train. {} batches to go...', batches_to_run))
count = 1
losses = []
start = time.time()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

f = open('out.txt', 'w')

while count <= batches_to_run:
    batch = model.generateBatch()
    dic = {model.x : batch[0][0], model.x_seq_len : batch[0][1],
                      model.z_con : batch[1][0], model.z_con_seq_len : batch[1][1],
                      model.z_incon : batch[2][0], model.z_incon_seq_len : batch[2][1]}
    _, loss = sess.run([optimizer, model.loss], feed_dict=dic)
    losses.append(loss)
    
    if count % config['print_freq'] == 0:
        timer = time.time() - start
        print(str.format('{} batches run. {} batches left', count, batches_to_run - count))
        loss = np.average(losses[-10:])
        print(str.format('{} last batches average loss = {}', 10, loss))
        h,m,s = formatTimer(timer)
        print(str.format('Train time: {}:{}:{}\n', h, m, s))
        f.write('batches: {}. {} last batches average loss: {}\n'.format(count, 10, loss))
        
    count += 1

print("training finished. Saving the model...")
model_dir = model.saveModel(sess)
f.close()
shutil.move('out.txt', model_dir + 'out.txt')
print("************      The End       ***************")











