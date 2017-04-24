import numpy as np


class Classifier(object):
    accuracies_ = []
    accuracy_ = None

    def __init__(self, k, pos=None, neg=None, reviews=None, vocab=None,
                 distance='euclidean', vector='frequency',
                 keep_punc=False, keep_stopwords=False):
        self.k = k
        self.distance = distance
        self.vector = vector
        self.keep_punc = keep_punc
        self.keep_stopwords = keep_stopwords
        self.vocab = vocab
        self.reviews = reviews
        self.pos = pos
        self.neg = neg
        if reviews is None or vocab is None or pos is None or neg is None:
            self.preprocess_data()

    def preprocess_data(self):
        import preporcessing
        from random import shuffle
        pos, neg, self.vocab = preporcessing.run(keep_punc=self.keep_punc,
                                                 keep_stopwords=self.keep_stopwords)
        # positive and negative reviews in vectors
        self.pos = [preporcessing.vectorize(review, sentiment='+', vec=self.vector, vocabulary=self.vocab)
                    for review in pos]
        self.neg = [preporcessing.vectorize(review, sentiment='-', vec=self.vector, vocabulary=self.vocab)
                    for review in neg]
        self.reviews = self.pos + self.neg
        shuffle(self.reviews)

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

    def predict(self, samples, targets):
        raise NotImplementedError

    def fit(self, pos, neg):
        raise NotImplementedError

    def score(self, n=5):
        """
        cv accuracy scorer
        :param n: 
        :return: 
        """
        length = len(self.reviews)
        size = length // n
        print(self.reviews[0])

        for i in range(0, length, size):
            train = self.reviews[: i] + self.reviews[i + size:]
            test = self.reviews[i: i + size]
            test_true = [review[-1] for review in test]
            self.fit(self.pos, self.neg)
            predictions = self.predict(train, test)
            self.accuracies_.append(sum(predictions[predictions[k] == test_true[k]]
                                        for k in range(len(test_true))))
        self.accuracy_ = np.mean(self.accuracies_)
        print('in {0} cv tests, accuracies are {1}'.format(n, self.accuracies_))
        return self.accuracy_
