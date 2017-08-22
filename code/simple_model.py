import numpy as np
import codecs

class SimpleModel:
    def __init__(self, file_name):
        with open(file_name, encoding='latin-1') as fh:
            data = np.loadtxt(fh, dtype="object")
        #Create a list of embedings and a mapping from word to embeddings index
        self.words = {_[1]:_[0] for _ in enumerate(data[:,0])}
        self.embeddings = data[:,1:].copy().astype("float64")

    def apply_ignore_unknown_words(self, sent, sess=None):
        sum_embeddings = 0
        num_embeddings = 0
        for word in sent.split():
            if not word in self.words:
                #TODO: we want another way of handling missing words
                print("Ignoring unknown word: " +  word)
            else:
                num_embeddings += 1
                sum_embeddings = sum_embeddings + self.embeddings[self.words[word]]
        if num_embeddings == 0:
            return 0
        return sum_embeddings / num_embeddings

    def apply_generate_new_embeddings_for_unknown_words(self, sent, sess=None):
        sum_embeddings = 0
        num_embeddings = 0
        max_index = max(self.words.values())
        for word in sent.split():
            num_embeddings += 1
            if not word in self.words:
                max_index += 1
                new_index = max_index
                new_embedding = np.random.normal(0, 0.4, 300)
                self.words[word] = new_index
                self.embeddings = np.vstack((self.embeddings, new_embedding))
                assert(np.all(self.embeddings[self.words[word]] == new_embedding))
            sum_embeddings = sum_embeddings + self.embeddings[self.words[word]]
        if num_embeddings == 0:
            return 0
        return sum_embeddings / num_embeddings

    def apply(self, sent, sess=None):
        return self.apply_generate_new_embeddings_for_unknown_words(sent, sess)


#Example usage: python3 simple_model.py
if __name__ == '__main__':
    import simple_evaluator
    import sys
    data_file = "../evaluation_data.tsv"
    model = SimpleModel("../paragram-phrase-XXL.txt")
    evaluator = simple_evaluator.Evaluator(model, data_file)
    if True:
        for p in (0.0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0):
            evaluator.eval2(p, False)
            print()
    else:
        evaluator.eval2(0.2, True)

