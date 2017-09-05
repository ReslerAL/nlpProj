
import numpy as np
import utils




class SimpleModel:
    def __init__(self, file_name):
        print("loading model's parameters...")
        with open(file_name, encoding='latin-1') as fh:
            data = np.loadtxt(fh, dtype="object")
        #Create a list of embedings and a mapping from word to embeddings index
        np.random.seed(0)
        self.words = {_[1]:_[0] for _ in enumerate(data[:,0])}
        self.embeddings = data[:,1:].copy().astype("float64")

    def apply(self, sent, sess=None, debug=False):
        sum_embeddings = 0
        num_embeddings = 0
        max_index = max(self.words.values())

        ignore = ["where"]
        sentwords = sent.split()
        resultwords = [word for word in sentwords if word.lower() not in ignore]
        resultwords = list(set([word for word in sentwords if word.lower() not in ignore]))
        sent = ' '.join(resultwords)
        for word in sent.split():
            num_embeddings += 1
            if not word in self.words:
                if debug:
                    print("rare word: ", word)
                max_index += 1
                new_index = max_index
                new_embedding = np.random.uniform(0, 1, 300)
                self.words[word] = new_index
                self.embeddings = np.vstack((self.embeddings, new_embedding))
                assert(np.all(self.embeddings[self.words[word]] == new_embedding))
            if debug:
                print("word: ", word)
                print("embed: ", self.embeddings[self.words[word]])
            sum_embeddings = sum_embeddings + self.embeddings[self.words[word]]
        if num_embeddings == 0:
            return 0
        if debug:
            print("sum embed: ", sum_embeddings)
            print("num embed: ", num_embeddings)

        return sum_embeddings / num_embeddings

    def apply_debug(self, sent):
        num_embeddings = 0
        max_index = max(self.words.values())
        embed_dic = {}

        for word in sent.split():
            num_embeddings += 1
            if not word in self.words:
                max_index += 1
                new_index = max_index
                new_embedding = np.random.uniform(0, 1, 300)
                self.words[word] = new_index
                self.embeddings = np.vstack((self.embeddings, new_embedding))
                assert(np.all(self.embeddings[self.words[word]] == new_embedding))
            embed_dic[word] = self.embeddings[self.words[word]]
        return embed_dic


#Example usage: python3 simple_model.py
if __name__ == '__main__':
    import simple_evaluator
    import sys
    data_file = "../evaluation_data.tsv"
    model = SimpleModel("../paragram-phrase-XXL.txt")
    evaluator = simple_evaluator.Evaluator(model, data_file)
    verbose = True
    if verbose:
        evaluator.eval(0.5, True, verbose)
    else:
        for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            evaluator.eval(p, True, verbose)
            print()

