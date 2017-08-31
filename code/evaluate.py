'''
Created on Aug 15, 2017

@author: alon
'''

import sys
import os
import tensorflow as tf
from utils import rawDataToVocabulary
from evaluator import *
from rnn_model import *
import argparse
from simple_model import *
import simple_evaluator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
model evaluation. Can evaluate the simple model and the lstm model. 
For lstm model run:
    python3 evalute.py -model lstm -dir <model_dir> -evaldata ../evaluation_data.tsv [-verbose true] [-elimination 0.3] [-softmax true]
For simple model run:
    python3 evalute.py -model simple -file <embeddingsfile> -evaldata ../evaluation_data.tsv [-verbose false] [-elimination 0.3] [-softmax false]
"""

parser = argparse.ArgumentParser()

parser.add_argument("-model", help="Model for evaluation (lstm or simple)")
parser.add_argument("-dir", help="directory for the saved lstm model")
parser.add_argument("-embdsfile", help="file of the word embeddings from Wieting")
parser.add_argument("-evaldata", help="The Evaluation data file")
parser.add_argument("-verbose", help="Print to log (optional)")
parser.add_argument("-t", help="the minimal number of logical forms required for elimination (optional - default is 0)")
parser.add_argument("-elimination", help="Elimination percent (optional - default is 0.5")
parser.add_argument("-softmax", help="Use softmax weights for reward (optional - default is false)")
parser.add_argument("-grid", help="perform grid search over p's and t's")

args = parser.parse_args()

if args.model == 'lstm':
    sess = tf.Session()
    model = getModelFromFile(args.dir, sess)
    model.prepEval()
else:  # simple model
    model = SimpleModel(args.embdsfile)

verbose = False
if (args.verbose != None and args.verbose.lower() == 'true'):
    verbose = True
softmax = False
if (args.softmax != None and args.softmax.lower() == 'true'):
    softmax = True
elimination = 0.5
if (args.elimination != None):
    elimination = float(args.elimination)
t = 0
if (args.t != None):
    t = float(args.t)
grid = False
if (args.grid != None and args.grid.lower() == 'true'):
    grid = True

print('evaluating with configuration: elimination {}    softmax {}'.format(elimination, softmax))
evaluator = simple_evaluator.Evaluator(model, args.evaldata)
if not grid:
    print("threshold : " + str(t))
    print("elimination : " + str(elimination))
    evaluator.eval(elimination, t, softmax, verbose)
else:
    for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99):
        for t in (1, 3, 5, 7, 10, 25, 50, 100):
            print("threshold : " + str(t))
            print("elimination : " + str(p))
            evaluator.eval(p, t, softmax, False)
            print()
