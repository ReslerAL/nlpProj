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
                    dic[lid] = [question, [], [], [], []]
                #Add the logical form to the correct or incorrect unique list
                if isCorrect:
                    dic[lid][1] = list(set(dic[lid][1] + [canonical]))
                    dic[lid][3] = list(set(dic[lid][3] + [logical]))
                else:
                    dic[lid][2] = list(set(dic[lid][2] + [canonical]))
                    dic[lid][4] = list(set(dic[lid][4] + [logical]))
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
        

    #p is proportion to keep
    #Skipping when proportion is less the one
    #actual number of items to keep is rounded down
    #
    #verbose means print analysis per question
    def eval2(self, p=.2, verbose=True):
        #F_ratio is F+/F, i.e. ratio of correct items in the eliminated items
        F_ratio_list = []
        #S_ratio is S-/F, i.e. ratio of spurious items in the kept items
        S_ratio_list = []
        count_questions_that_lost_correct_canonical_forms = 0
        count_questions_that_were_discarded = 0
        count_total_number_of_questions = 0

        for lid in self.dic.keys():
            #Process cosine similarities for all canonicals
            question_embedding = self.model.apply(self.dic[lid][0])
            correct_canonical_embeddings = [self.model.apply(canonical) for canonical in self.dic[lid][1]]
            incorrect_canonical_embeddings = [self.model.apply(canonical) for canonical in self.dic[lid][2]]
            correct_cosine_similarities = [my_cosine_similarity(question_embedding, canonical_embedding) for canonical_embedding in correct_canonical_embeddings]
            incorrect_cosine_similarities = [my_cosine_similarity(question_embedding, canonical_embedding) for canonical_embedding in incorrect_canonical_embeddings]
            all_cosine_similarities = correct_cosine_similarities + incorrect_cosine_similarities
            all_cosine_similarities = np.array(all_cosine_similarities)
            labels = np.array(([1] * len(correct_cosine_similarities)) + ([0] * len(incorrect_cosine_similarities)))
            all_cosine_similarities = np.squeeze(all_cosine_similarities)
            if all_cosine_similarities.shape == ():
                all_cosine_similarities = (all_cosine_similarities.tolist(),)

            canonical_forms = self.dic[lid][1] + self.dic[lid][2]
            logical_forms = self.dic[lid][3] + self.dic[lid][4]

            count_total_number_of_questions += 1
            c = len(correct_canonical_embeddings)
            i = len(incorrect_canonical_embeddings)
            n = c + i
            #P is how many to keep
            P = int(p * n)
            
#            if P < 1:
#                count_questions_that_were_discarded += 1
#                #Next question
#                continue

            ranked = np.argsort(all_cosine_similarities)
            keep_zone = True
            if verbose:
                print("###################################################")
                print(str(lid) + ".", self.dic[lid][0])
            S_minus = 0
            F_plus = 0
            F = 0
            S = 0
            for j in range(len(ranked)):
                current_canonical_index = ranked[n-j-1]
                canonical = canonical_forms[current_canonical_index]
                logical = logical_forms[current_canonical_index]
                label = labels[current_canonical_index]
                similarity = all_cosine_similarities[current_canonical_index]
                if (j >= P) and keep_zone:
                    keep_zone = False
                    if verbose:
                        print("---------------------------------------------------")
                if (not keep_zone) and label == 1:
                    F_plus += 1
                if keep_zone and label == 0:
                    S_minus += 1
                if keep_zone:
                    S += 1
                else:
                    F += 1
                if verbose:
                    print ("\t", str(j), "logical:", logical, "canonical:", canonical, "correct:", label, "similarity:", similarity)

            if F_plus == 0:
                F_ratio = 0
            else:
                count_questions_that_lost_correct_canonical_forms += 1
                F_ratio = float(F_plus) / float(F)

            if S_minus == 0:
                S_ratio = 0
            else:
                S_ratio = float(S_minus) / float(S)

            F_ratio_list.append(F_ratio)
            S_ratio_list.append(S_ratio)

        if verbose:
            print("###################################################")
            print("")
        print("p proportion used:", p)
        print("Mean S ratio:", np.mean(S_ratio_list))
        print("Mean F ratio:", np.mean(F_ratio_list))
        print("count_questions_that_lost_correct_canonical_forms:", count_questions_that_lost_correct_canonical_forms)
        #print("count_questions_that_were_discarded:", count_questions_that_were_discarded)
        print("count_total_number_of_questions:", count_total_number_of_questions)
