'''
Created on Aug 6, 2017

@author: alon
'''

import sklearn
from utils import *

"""
Evaluate for the lstm model. In the evaluation data we have question with few examples of correct and incorrect logical forms.
We want to measure how much the model know to distinguish between the two. One way to so is to transfer the model as scoring 
function and then to normal the scoring to a distribution and then check how much distribution mass the model put on the correct forms.
This is done twice, first with regular normalization the second using softmax to normalize.
Another evaluation method isusing the model to eliminate x forms suspected to be incorrect and then check how much of the eliminated are
correct eliminations (i.e incorrect forms)
"""
class Evaluator:
    
    def __init__(self, model, fileName):
        self.model = model
        self.raw_data = self.parseFile(fileName)
        self.dic = processData(self.raw_data, model.vocab)
        
     
    def eval(self, sess): 
        self.basic_normalized_eval_list = []
        self.softmax_eval_list = []
        self.elimination_eval_list = []
        for lid in self.dic.keys():
            question_embedding = self.model.apply(self.dic[lid][0], sess)
            correct_canonical_embeddings = [self.model.apply(canonical, sess) for canonical in self.dic[lid][1]]
            incorrect_canonical_embeddings = [self.model.apply(canonical, sess) for canonical in self.dic[lid][2]]
            correct_cosine_similarities = [my_cosine_similarity(question_embedding, canonical_embedding) for canonical_embedding in correct_canonical_embeddings]
            incorrect_cosine_similarities = [my_cosine_similarity(question_embedding, canonical_embedding) for canonical_embedding in incorrect_canonical_embeddings]
            all_cosine_similarities = correct_cosine_similarities + incorrect_cosine_similarities
            all_cosine_similarities = np.array(all_cosine_similarities)
            labels = np.array(([1] * len(correct_cosine_similarities)) + ([0] * len(incorrect_cosine_similarities)))
            all_cosine_similarities = np.squeeze(all_cosine_similarities)
            self.basic_normalized_eval_list.append(self.basic_normalized_eval(all_cosine_similarities, labels))
            self.softmax_eval_list.append(self.softmax_eval(all_cosine_similarities, labels))
            self.elimination_eval_list.append(self.elimination_eval(all_cosine_similarities, labels, 5))
        res = {}
        res['basic normalized evaluation'] = np.mean(self.basic_normalized_eval_list)
        res['softmax normalized evaluation'] = np.mean(self.softmax_eval_list)
        res['elimination evaluation'] = np.mean(self.elimination_eval_list)
        return res
    
    
    def parseFile(self, fileName):
        """
        take input file where each line is <key><separtor><value>
        and return a dictionary
        """
        dic = {}
        with open(fileName) as f:
            for line in f:
                #Parse line
                line = removeApostrophe(line)
                line = line.split('\t')
                assert len(line) == 5
                lid, question, canonical, logical, isCorrect = line
                lid = int(lid)
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

        
