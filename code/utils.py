'''
Created on Aug 2, 2017

@author: alon
'''

import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from fileinput import filename


def removeApostrophe(str):
    return str.replace("'", "")


def fileToDic(fileName):
    """
    take input file where each line is <key><separtor><value>
    and return a dictionary of <question_id :[question, [consistent canonical forms],[inconsistent canonical forms] ]> 
    """
    dic = {}
    with open(fileName) as f:
        for line in f:
            #Parse line
            line = removeApostrophe(line)
            line = line.split('\t')
            assert len(line) == 5
            lid, question, canonical, logicalForm, isConsistent = line
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
    return dic

def fileToVocabulary(dataDic):
    """
    take the dataDic return by last function and create dictionary that map word to index. 
    The words need to be taken from the quistions and canonical forms
    """
    return 0

def toOneHot(wordsIdxs, word):
    """
    return np array that represent one hot vector of size len(wordsIdxs.keys()) where the entry that have one is wordsIdxs[word]
    """
    return 0

def correct_distribution_mass(self, pred_dist, labels):
    '''
    pred_dist - vector that represent prediction distribution (normalized)
    labals - zero one vector for correct/incorrect
    return l1_norm(<pred_dist, labels>)
    '''
    return np.inner(pred_dist, labels)

def basic_normalize(vec):
    vec = vec + 1
    vec = vec / numpy.linalg.norm(vec)
    return vec

def softmax_normalize(vec):
    e_vec = np.exp(vec - np.max(vec))
    return e_vec / e_vec.sum()


