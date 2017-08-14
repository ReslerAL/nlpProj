'''
Created on Aug 2, 2017

@author: alon
'''
from __future__ import print_function, division
import numpy as np
from fileinput import filename
import random
import re
from numpy import average
from builtins import str
import math
#from aifc import data



def removeApostrophe(str):
    return str.replace("'", "")


def fileToDic(fileName):
    """
    take input file where each line is <key><separtor><value>
    and return a dictionary of <question_id :[question, [consistent canonical forms],[inconsistent canonical forms] ]> 
    """
    dic = {}
    max_sent_size = 0
    max_sent = ''
#     ttl_len = 0
    count = 0
    with open(fileName) as f:
        for line in f:
            #Parse line
            line = removeApostrophe(line)
            line = line.split('\t')
            assert len(line) == 5
            lid, question, canonical, logicalForm, isConsistent = line
#             ttl_len += len(question.split())
#             ttl_len += len(canonical.split())
            count += 1
#             if len(question.split()) > max_sent_size:
#                 max_sent_size = len(question.split())
#                 max_sent = question
#             if len(canonical.split()) > max_sent_size:
#                 max_sent_size =len(canonical.split())
#                 max_sent = canonical
            lid = int(lid)
            #Convert isConsistent to boolean
            assert isConsistent in ("True\n", "False\n"), "isConsistent: <" + str(isConsistent) + ">"
            isConsistent = isConsistent == "True\n"

            #Add line to dictionary
            if lid not in dic:
                dic[lid] = [question, [], []]
            #Add the logical form to the consistent or incosistent unique list
            if isConsistent:
                dic[lid][1] = list(set(dic[lid][1] + [canonical]))
            else:
                dic[lid][2] = list(set(dic[lid][2] + [canonical]))
#     print("max lenght is " + str(max_sent_size))
#     print("max sent is \n" + max_sent)
#     print("average len is " + str(ttl_len/count))
#     print("count is " + str(count))
    return dic

def rawDataToVocabulary(dataDic):
    """
    take the dataDic return by last function and create dictionary that map word to index. 
    The words need to be taken from the questions and canonical forms
    """
    vocab = {}
    for value in dataDic.values():
        addToVocab(value[0].split(), vocab)
        for sent in value[1]:
            addToVocab(sent.split(), vocab)
        for sent in value[2]:
            addToVocab(sent.split(), vocab)        
    return vocab

def addToVocab(words, vocab):
    n = len(vocab)
    for word in words:
        if word not in vocab:
            vocab[word] = n
            n += 1

def correct_distribution_mass(self, pred_dist, labels):
    '''
    pred_dist - vector that represent prediction distribution (normalized)
    labals - zero one vector for correct/incorrect
    return l1_norm(<pred_dist, labels>)
    '''
    return np.inner(pred_dist, labels)

def basic_normalize(vec):
    vec = vec + 1
    vec = vec / np.linalg.norm(vec)
    return vec

def softmax_normalize(vec):
    e_vec = np.exp(vec - np.max(vec))
    return e_vec / e_vec.sum()


"""
convert list of words to list of the word indices according to the embeddingsToIndx dictionary
if a new word is found it will be added to the embeddingsToIndx dic
"""
def toEmbeddingList(word_list, embeddingsToIndx):
    res = []
    n = len(embeddingsToIndx)
    for word in word_list:
        if word not in embeddingsToIndx:
            embeddingsToIndx[word] = n
            n += 1
        res.append(embeddingsToIndx[word])
    return res

"""
array - array of arrays 
return the maximun length of array in arrays
"""
def getMaxLength(arrays):
    max = 0
    for array in arrays:
        if len(array) > max:
            max = len(array)
    return max

def formatTimer(timer):
    sec = math.floor(timer)
    hours = round(sec // 3600)
    sec = sec - 3600*hours
    minutes = round(sec // 60)
    sec = sec - 60*minutes
    return hours, minutes, sec
 
if __name__ == '__main__':
    print("in utils main") 
    timer = 5605.0228905677795
    print(formatTimer(timer))
      
    
    
    
#     raw_data = fileToDic('../project_train_data.csv')
#     print(len(raw_data))
#     print(raw_data[32])
#     vocab = rawDataToVocabulary(raw_data)
#     print(len(vocab))
#     print(vocab['chart'], vocab['name'], vocab['dog'])

    
    
    
    
    
    
    
        


