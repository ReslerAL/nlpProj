'''
Created on Aug 2, 2017

@author: alon
'''
import numpy as np
import tensorflow as tf


####### Testing functions #######
def test_fileToDic():
    from utils import fileToDic
    dic = fileToDic('../project_train_data.csv')
    print ("question:", dic[6750][0])
    print ("consistent:")
    for line in dic[6750][1]:
        print (line)
    print ("")
    print ("inconsistent:")
    for line in dic[6750][2]:
        print (line)

def test_generateBatch():
    from rnn_model import Rnn_model
    rnn = Rnn_model("../project_train_data.csv")
    for entry in rnn.generateBatch(10):
        print ("----------")
        print (entry)

if __name__ == '__main__':
    test_generateBatch()
