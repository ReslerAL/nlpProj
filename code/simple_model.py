import numpy as np
import codecs

class SimpleModel:
    def __init__(self, file_name):
        with open(file_name, encoding='latin-1') as fh:
            data = np.loadtxt(fh, dtype="object")
        #Create a list of embedings and a mapping from word to embeddings index
        self.words = {_[1]:_[0] for _ in enumerate(data[:,0])}
        self.embeddings = data[:,1:].copy().astype("float64")

    def apply(self, sent, sess=None):
        sum_embeddings = 0
        num_embeddings = 0
        for word in sent.split():
            if not word in words:
                #TODO: we want another way of handling missing words
                print("Ignoring unknown word: " +  word)
            else:
                num_embeddings += 1
            sum_embeddings = sum_embeddings + embeddings[words[word]]
        if num_embeddings == 0:
            return 0
        return sum_embeddings / num_embeddings


#Example usage: python3 simple_model.py ../evaluation_data.tsv
if __name__ == '__main__':
    import evaluator
    import sys
    data_file = sys.argv[1] #e.g. ../evaluation_data.tsv
    model = SimpleModel("../paragram-phrase-XXL.txt")
    evaluator = evaluator.Evaluator(model, data_file)
    sess = None
    res = evaluator.eval(sess)
    print(res)
