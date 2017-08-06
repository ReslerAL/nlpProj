'''
Created on Aug 2, 2017

@author: alon
'''
from utils import *
import numpy as np
import tensorflow as tf



if __name__ == '__main__':
    dic = fileToDic('../project_train_data.csv')
    print (list(enumerate(dic[6750][2])))
    
