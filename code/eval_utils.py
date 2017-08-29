from builtins import str
from sklearn.metrics.pairwise import cosine_similarity


def remove_apostrophe(str):
    return str.replace("'", "")

def my_cosine_similarity(x, y):
    print("" + x.reshape(1, -1))
    return cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))