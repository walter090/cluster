import numpy as np


class Classifier(object):
    def __init__(self, k, reviews=None, vocab=None, distance='euclidean', vector='frequency',
                 keep_punc=False, keep_stopwords=False):
        self.k = k
        self.distance = distance
        self.vector = vector
        self.keep_punc = keep_punc
        self.keep_stopwords = keep_stopwords
        self.vocab = vocab
        self.reviews = reviews

    def preprocess_data(self):
        import preporcessing
        from random import shuffle
        pos, neg, self.vocab = preporcessing.run(keep_punc=self.keep_punc,
                                                 keep_stopwords=self.keep_stopwords)
        # positive and negative reviews in vectors
        pos = [preporcessing.vectorize(review, sentiment='+', vec=self.vector, vocabulary=self.vocab)
               for review in pos]
        neg = [preporcessing.vectorize(review, sentiment='-', vec=self.vector, vocabulary=self.vocab)
               for review in neg]
        self.reviews = shuffle(pos + neg)

    def distance(self, vector1, vector2):
        import math
        distance = 0
        length = len(vector1) - 1
        if self.distance == 'euclidean':
            for i in range(length):
                distance += (vector1[i] - vector2[i]) ** 2
            return math.sqrt(distance)
        else:
            for i in range(length):
                distance += np.abs(vector1[i] - vector2[i])
            return distance

    def fit(self):
        raise NotImplementedError
