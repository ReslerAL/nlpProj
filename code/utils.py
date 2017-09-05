'''
Created on Aug 2, 2017

@author: alon
'''
from __future__ import print_function, division
import numpy as np
from fileinput import filename
import random
import pickle
from numpy import average
from builtins import str
import math
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from rnn_model import *
from simple_model import *


def fileToDic(fileName, clean):
    """
    take input file where each line is <key><separtor><value>
    and return a dictionary of <question_id :[question, [consistent canonical forms],[inconsistent canonical forms] ]> 
    """
    dic = {}
    with open(fileName) as f:
        for line in f:
            line = line.split('\t')
            assert len(line) == 5
            lid, question, canonical, logicalForm, isConsistent = line
            for sym in "-+.^:,?!'()\"":
                    question = question.replace(sym, "")
            for sym in "-+^:,?!'()\"":
                canonical = canonical.replace(sym, "")
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
    if clean:
        clean_consistent(dic)
    return cleanData(dic)

def getDataLen(data):
    count = 0
    for lid in data:
        count += len(data[lid][1])
        count += len(data[lid][2])
    return count
    
"""for some reason there are sentences without inconsistent forms - remove them"""
def cleanData(dic):
    rem = []
    for lid in dic:
        if len(dic[lid][1]) == 0 or len(dic[lid][2]) == 0:
            rem += [lid]
    for lid in rem:
        del dic[lid]
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

def extendVocab(vocab, dataDic):
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

def correct_distribution_mass(pred_dist, labels):
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

def cosine_sim(x, y):
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))

def clean_consistent(dic, p=0.8):
    model = SimpleModel("paragram-phrase-XXL.txt")
    print("cleaning dataeset...")
    i = 0
    dic_len = len(dic)
    for qid in dic.keys():
        if (i % 100 == 0):
            print("cleaned " + str(i) + "/" + str(dic_len))
        #process similarities
        question_embedding = model.apply(dic[qid][0])
        const_canonical_embeddings = [model.apply(canonical) for canonical in dic[qid][1]]
        sims = [cosine_sim(question_embedding, canonical_embedding) for canonical_embedding in const_canonical_embeddings]
        sims = np.array(sims)
        sims = np.squeeze(sims)
        if sims.shape == ():
            sims = (sims.tolist(),)
        #clear data
        ranked = np.argsort(sims)
        threshold = int(p * len(sims))
        kept_idx = ranked[threshold:]
        const = np.array(dic[qid][1])
        dic[qid][1] = list(const[kept_idx])
        i = i + 1


"""
convert the data to the same structure just instead of words in each place
we will hold the indices of the word embeddings
"""   
def processData(raw_data, vocab):
    dic = {}
    for qstn_id in raw_data:
        if qstn_id not in dic:
            dic[qstn_id] = [toEmbeddingList(raw_data[qstn_id][0].split(), vocab), [], []]
        for con_form in raw_data[qstn_id][1]:
            dic[qstn_id][1].append(toEmbeddingList(con_form.split(), vocab))
        for con_form in raw_data[qstn_id][2]:
            dic[qstn_id][2].append(toEmbeddingList(con_form.split(), vocab))
    return dic
    
    
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

"""
model directory is a subdirectory in 'saved' name after the saving time
"""
def getModelFromFile(model_dir, sess):
    confFile = model_dir + 'configuration.p'
    config = pickle.load(open(confFile, 'rb'))
    raw_data = pickle.load( open( "raw_data_0.6.p", "rb" ) )#fileToDic(config['data_file'], True)
    model = Rnn_model(raw_data, config)
    model.load(model_dir, sess)
    return model

def my_cosine_similarity(x, y):
    print("" + x.reshape(1, -1))
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))

if __name__ == '__main__':
    raw_data = fileToDic('./train_data_version2.tsv', True)
    pickle.dump( raw_data, open( "raw_data2_0.6.p", "wb" ) )
    print("raw data was saved")
    

    
