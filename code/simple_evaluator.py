'''
Created on Aug 6, 2017

@author: alon
'''

import numpy as np
import utils
from utils import *

class Evaluator:
    
    def __init__(self, model, fileName):
        self.model = model
        self.dic = self.parseFile(fileName)
        self.basic_normalized_eval_list = []
        self.softmax_eval_list = []
        self.elimination_eval_list = []

    def eval(self):
        for lid in self.dic.keys():
            question_embedding = self.model.apply(self.dic[lid][0])
            correct_canonical_embeddings = [self.model.apply(canonical) for canonical in self.dic[lid][1]]
            incorrect_canonical_embeddings = [self.model.apply(canonical) for canonical in self.dic[lid][2]]
            correct_cosine_similarities = [my_cosine_similarity(question_embedding, canonical_embedding) for canonical_embedding in correct_canonical_embeddings]
            incorrect_cosine_similarities = [my_cosine_similarity(question_embedding, canonical_embedding) for canonical_embedding in incorrect_canonical_embeddings]
            all_cosine_similarities = correct_cosine_similarities + incorrect_cosine_similarities
            all_cosine_similarities = np.array(all_cosine_similarities)
        
            labels = np.array(([1] * len(correct_cosine_similarities)) + ([0] * len(incorrect_cosine_similarities)))
            all_cosine_similarities = np.squeeze(all_cosine_similarities)

            self.basic_normalized_eval_list.append(self.basic_normalized_eval(all_cosine_similarities, labels))
            self.softmax_eval_list.append(self.softmax_eval(all_cosine_similarities, labels))
            self.elimination_eval_list.append(self.elimination_eval(all_cosine_similarities, labels, 5))
        return (np.mean(self.basic_normalized_eval_list), np.mean(self.softmax_eval_list), np.mean(self.elimination_eval_list))
        
    def parseFile(self, fileName):
        dic = {}
        with open(fileName) as f:
            for line in f:
                #Parse line
                line = utils.removeApostrophe(line)
                line = line.split('\t')
                assert len(line) == 5
                lid, question, canonical, logical, isCorrect = line
                lid = int(lid)
                for sym in "-+.^:,?!'":
                    question = question.replace(sym, "")
                for sym in "-+^:,?!'":
                    canonical = canonical.replace(sym, "")
                #Convert isCorrect to boolean
                assert isCorrect in ("True\n", "False\n"), "isCorrect: <" + str(isCorrect) + ">"
                isCorrect = isCorrect == "True\n"

                #Add line to dictionary
                if lid not in dic:
                    dic[lid] = [question, [], []]
                #Add the logical form to the correct or incorrect unique list
                if isCorrect:
                    dic[lid][1] = list(set(dic[lid][1] + [canonical]))
                else:
                    dic[lid][2] = list(set(dic[lid][2] + [canonical]))
        return dic
    
    def basic_normalized_eval(self, similarityVector, labels):
        '''
        return the distribution mass on the correct samples when using regular normalization 
        '''
        similarityVector = basic_normalize(similarityVector)
        return correct_distribution_mass(similarityVector, labels)
        
    
    def softmax_eval(self, similarityVector, labels):
        '''
        return the distribution mass on the correct samples when using softmax to normalized 
        '''
        similarityVector = softmax_normalize(similarityVector)
        return correct_distribution_mass(similarityVector, labels)
    
    def elimination_eval(self, similarityVector, labels, size):
        '''
        eliminate the minimum size of the pred_dist and return #of correct eliminated / size - 
        this is the fraction of incorrect vectors that were eliminated  
        '''

        eliminated = np.argsort(similarityVector)[:size]
        inversed_labels = 1 - labels
        guessed_right = np.sum(inversed_labels[eliminated])
        
        return float(guessed_right) / float(size)
        
        
