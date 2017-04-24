from classifier import Classifier


class KNNClassifier(Classifier):
    """
        self.k = k
        self.distance = distance
        self.vector = vector
        self.keep_punc = keep_punc
        self.keep_stopwords = keep_stopwords
        self.pos = pos
        self.neg = neg
        self.vocab = vocab
    """
    def fit(self):
        Classifier.preprocess_data(self)

    def k_neighbors(self, samples, target, k):
        import operator
        distances = []
        for i, point in enumerate(samples):
            distances.append((Classifier.distance(self, point, target), point[-1]))
        closest = sorted(distances, key=operator.itemgetter(0))[: k]
        return list(zip(*closest))[0]


