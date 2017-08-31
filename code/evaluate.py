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

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
parser.add_argument("-elimination", help="Elimination percent (optional - default is to try 10: 0.1, 0.2, ...)")
parser.add_argument("-softmax", help="Use softmax weights for reward (optional - default is false)")

args = parser.parse_args()

if args.model == 'lstm':
    sess = tf.Session()
    model = getModelFromFile(args.dir, sess)
    model.prepEval()
else: #simple model
    model = SimpleModel(args.embdsfile)

verbose = False
if (args.verbose != None and args.verbose.lower() == 'true'):
    verbose = True
softmax = False
if (args.softmax != None and args.softmax.lower() == 'true'):
    softmax = True
elimination = None
if (args.elimination != None):
    elimination = float(args.elimination)

print('evaluating with configuration: elimination {}    softmax {}'.format(elimination, softmax))    
evaluator = simple_evaluator.Evaluator(model, args.evaldata)
if args.elimination != None:
    evaluator.eval(elimination, softmax, verbose)
else:
    for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        evaluator.eval(p, softmax, verbose)
        print()

