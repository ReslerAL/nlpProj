'''
Created on Aug 6, 2017
code.
@author: alon
'''

import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

class Evaluator:
    
    def __init__(self, model, fileName):
        self.model = model
        self.dic = self.parseFile(fileName)

    @staticmethod
    def remove_apostrophe(s):
        return s.replace("'", "")

    @staticmethod
    def cosine_sim(x, y):
        return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))


    def parseFile(self, fileName):
        dic = {}
        print ("read evaluation dataset: ", fileName)
        with open(fileName) as f:
            for line in f:
                #Parse line
                line = Evaluator.remove_apostrophe(line)
                line = line.split('\t')
                assert len(line) == 5
                lid, question, canonical, logical, isCorrect = line
                if lid == "":
                    break
                lid = int(lid)
                for sym in "-+.^:,?!'()\"":
                    question = question.replace(sym, "")
                for sym in "-+^:,?!'()\"":
                    canonical = canonical.replace(sym, "")
                #Convert isCorrect to boolean
                assert isCorrect in ("True\n", "False\n"), "isCorrect: <" + str(isCorrect) + ">"
                isCorrect = isCorrect == "True\n"

                #Add line to dictionary
                if lid not in dic:
                    dic[lid] = [question, [], [], [], []]
                #Add the logical form to the correct or incorrect unique list
                if isCorrect:
                    if (canonical not in dic[lid][1]):
                        dic[lid][1].append(canonical)
                        dic[lid][3].append(logical)
                else:
                    if (canonical not in dic[lid][2]):
                        dic[lid][2].append(canonical)
                        dic[lid][4].append(logical)
        return dic

    def process_similarities(self, qid):
        question_embedding = self.model.apply(self.dic[qid][0])
        correct_canonical_embeddings = [self.model.apply(canonical) for canonical in self.dic[qid][1]]
        spurious_canonical_embeddings = [self.model.apply(canonical) for canonical in self.dic[qid][2]]
        correct_cosine_similarities = [Evaluator.cosine_sim(question_embedding, canonical_embedding) for
                                       canonical_embedding in correct_canonical_embeddings]
        spurious_cosine_similarities = [Evaluator.cosine_sim(question_embedding, canonical_embedding) for
                                         canonical_embedding in spurious_canonical_embeddings]

        sims = correct_cosine_similarities + spurious_cosine_similarities
        sims = np.array(sims)
        labels = np.array(([1] * len(correct_cosine_similarities)) + ([0] * len(spurious_cosine_similarities)))
        sims = np.squeeze(sims)
        if sims.shape == ():
            sims = (sims.tolist(),)
        return sims, labels

    @staticmethod
    def sims_to_rewards_softmax(sims, p=0.2, t=5):
        elim_rewards = Evaluator.sims_to_rewards_elimination(sims, p, t)
        elim_sims = np.multiply(elim_rewards, sims)
        elim_sims = [float("-inf") if x == 0 else x for x in elim_sims]
        scaled = np.multiply(elim_sims, [100]*len(elim_sims))
        e_vec = np.exp(scaled - np.max(scaled))
        rewards = e_vec / e_vec.sum()
        return rewards

    @staticmethod
    def sims_to_rewards_elimination(sims, p=0.2, t=3):
        rewards = [1]*len(sims)
        if len(sims) < t:
            return rewards
        ranked = np.argsort(sims)
        threshold = int(p * len(sims))
        eliminated = ranked[:threshold]
        for i in eliminated:
            rewards[i] = 0
        return rewards

    @staticmethod
    def sims_to_rewards(sims, p=0.2, t=0, soft=True):
        if soft:
            return Evaluator.sims_to_rewards_softmax(sims, p, t)
        else:
            return Evaluator.sims_to_rewards_elimination(sims, p, t)

    @staticmethod
    def score_results(rewards, lables):
        kept = [index for index, value in enumerate(rewards) if value > 0]
        eliminated = [index for index, value in enumerate(rewards) if value == 0]
        assert len(kept) > 0
        rewards = np.array(rewards)
        #print("DEBUG rewards: ", rewards)
        correct_rewards = np.multiply(lables[kept], rewards[kept])
        #print("DEBUG kept_lables: ", lables[kept])
        #print("DEBUG kept_rewards: ", rewards[kept])
        #print("DEBUG correct: ", correct_rewards)
        kept_noise = 1 - (np.sum(correct_rewards) / np.sum(rewards))
        if len(eliminated) > 0:
            eliminated_noise = np.sum(lables[eliminated]) / len(eliminated)
        else:
            eliminated_noise = 0
        return kept_noise, eliminated_noise

    def eval(self, p=.2, t=0, soft=False, verbose=True):
        kept_noise_list = []
        eliminated_noise_list = []
        correct_elim_cnt = 0
        err_id = 0

        for qid in self.dic.keys():
            sims, lables = self.process_similarities(qid)
            rewards = Evaluator.sims_to_rewards(sims, p, t, soft)

            kept_noise, eliminated_noise = Evaluator.score_results(rewards, lables)
            kept_noise_list.append(kept_noise)
            eliminated_noise_list.append(eliminated_noise)
            if kept_noise == 1:
                correct_elim_cnt = correct_elim_cnt + 1

                #print question stats
                if verbose and eliminated_noise > 0:
                    j = 0
                    ranked = np.argsort(sims)
                    logical_forms = self.dic[qid][3] + self.dic[qid][4]
                    canons = self.dic[qid][1] + self.dic[qid][2]
                    threshold = int(p * len(ranked))
                    print("###################################################")
                    print(str(err_id) + ". " + str(qid) + ": ", self.dic[qid][0])
                    err_id = err_id + 1
                    for i in ranked:
                        logical = logical_forms[i]
                        canonical = canons[i]
                        label = lables[i]
                        sim = sims[i]
                        reward = rewards[i]
                        if j == threshold:
                            print("---------------------------------------------------")
                        print("\t", str(j), "correct:", label, "reward:", reward, "similarity:", sim, "canonical:", canonical,
                              "logical:", logical)
                        j = j + 1
                    print("")

        # print final results
        print("number of questions: ", len(self.dic.keys()))
        print("p: elimination proportion used: ", p)
        print("Mean kept noise: ", np.mean(kept_noise_list))
        print("Mean eliminated noise: ", np.mean(eliminated_noise_list))
        print("Number of questions without correct: ", correct_elim_cnt)
