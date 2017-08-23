'''
Created on Aug 6, 2017
code.
@author: alon
'''

import numpy as np
import math

from sklearn.metrics.pairwise import cosine_similarity

#utils functions
def remove_apostrophe(str):
    return str.replace("'", "")

def my_cosine_similarity(x, y):
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))

class Evaluator:
    
    def __init__(self, model, fileName):
        self.model = model
        self.dic = self.parseFile(fileName)
        self.basic_normalized_eval_list = []
        self.softmax_eval_list = []
        self.elimination_eval_list = []

    def parseFile(self, fileName):
        dic = {}
        with open(fileName) as f:
            for line in f:
                #Parse line
                line = remove_apostrophe(line)
                line = line.split('\t')
                assert len(line) == 5
                lid, question, canonical, logical, isCorrect = line
                if lid == "":
                    break
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
    

    def process_similarities(self, qid):
        question_embedding = self.model.apply(self.dic[qid][0])
        correct_canonical_embeddings = [self.model.apply(canonical) for canonical in self.dic[qid][1]]
        spurious_canonical_embeddings = [self.model.apply(canonical) for canonical in self.dic[qid][2]]
        correct_cosine_similarities = [my_cosine_similarity(question_embedding, canonical_embedding) for
                                       canonical_embedding in correct_canonical_embeddings]
        spurious_cosine_similarities = [my_cosine_similarity(question_embedding, canonical_embedding) for
                                         canonical_embedding in spurious_canonical_embeddings]

        sims = correct_cosine_similarities + spurious_cosine_similarities
        sims = np.array(sims)
        labels = np.array(([1] * len(correct_cosine_similarities)) + ([0] * len(spurious_cosine_similarities)))
        sims = np.squeeze(sims)
        if sims.shape == ():
            sims = (sims.tolist(),)
        return sims, labels

    def sims_to_rewards(self, qid, sims, p=0.2, verbose=True):
        rewards = [1]*len(sims)
        ranked = np.argsort(sims)
        threshold = int(p * len(sims))
        eliminated = ranked[:threshold]
        for i in eliminated:
            rewards[i] = 0

        if (verbose):
            j = 0
            logical_forms = self.dic[qid][3] + self.dic[qid][4]
            canons = self.dic[qid][1] + self.dic[qid][2]
            question = self.dic[qid][1] + self.dic[qid][2]
            for i in reversed(ranked):

                if len(ranked) - j == threshold:
                    print("---------------------------------------------------")

                print("\t", str(j), "logical:", logical, "canonical:", canonical, "correct:", label, "similarity:",
                      sims[i])
                j = j + 1

        return rewards

    def score_results(self, rewards, lables):
        kept = [index for index, value in enumerate(rewards) if value == 1]
        eliminated = [index for index, value in enumerate(rewards) if value == 0]
        assert len(kept) > 0
        kept_noise = 1 - (np.sum(lables[kept]) / len(kept))
        if (len(eliminated) > 0):
            eliminated_noise = np.sum(lables[eliminated]) / len(eliminated)
        else:
            eliminated_noise = 0
        return kept_noise, eliminated_noise

    def eval(self, p=.2, verbose=True):
        kept_noise_list = []
        eliminated_noise_list = []
        for qid in self.dic.keys():
            sims, lables = self.process_similarities(qid)
            rewards = self.sims_to_rewards(sims, p)
            kept_noise, eliminated_noise = self.score_results(rewards, lables)
            kept_noise_list.append(kept_noise)
            eliminated_noise_list.append(eliminated_noise)

        print("number of questions:", len(self.dic.keys()))
        print("p: elimination proportion used:", p)
        print("Mean kept noise:", np.mean(kept_noise_list))
        print("Max kept noise:", np.max(kept_noise_list))
        print("Min kept noise:", np.min(kept_noise_list))
        print("Mean eliminated noise:", np.mean(eliminated_noise_list))


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
            P = math.ceil(p * n)
            
            if (i < 1) or (c < 1):
                count_questions_that_were_discarded += 1
                #Next question
                continue

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
        print("count_questions_that_were_discarded:", count_questions_that_were_discarded)
        print("count_total_number_of_questions:", count_total_number_of_questions)
