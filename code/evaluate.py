'''
Created on Aug 15, 2017

@author: alon
'''

import sys
import os
import tensorflow as tf
from utils import *
from evaluator import *

"""
evaluate a model given by the argv[1] on data given by argv[2]
running command example:
python3 evaluate.py 1502891900064 evaluation_data.tsv
"""

eval_data_file = '../filename'
if len(sys.argv) < 3:
    print("missing arguments. run command should be like:\n" + 
          "python3 evaluate.py 1502789449262 ../evaluation_data.tsv")
    quit()

model_dir = '../saved/' + sys.argv[1] + '/'
data_file = sys.argv[2]

if not os.path.isdir(model_dir):
    print('directory {} does not exist'.format(model_dir))
    quit()

if not os.path.exists(data_file):
    print('data file {} does not exist'.format(data_file))
    quit()

sess = tf.Session()
model = getModelFromFile(model_dir, sess)

input = [45, 343, 54, 24, 786, 2345, 234]
res = model.apply(input, sess)
print(res)


# evautor = Evaluator(model, eval_data_file)
# result = evautor.eval(sess)













  
