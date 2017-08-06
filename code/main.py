'''
Created on Aug 2, 2017

@author: alon
'''
from utils import *
import numpy as np
import tensorflow as tf



if __name__ == '__main__':
    dic = fileToDic('../project_train_data.csv')
    print ("question:", dic[6750][0])
    print ("consistent:")
    for line in dic[6750][1]:
        print (line)
    print ("")
    print ("inconsistent:")
    for line in dic[6750][2]:
        print (line)
    
