from classifier import Classifier


class NearestCentroidClassifier(Classifier):
    pos_center_ = []
    neg_center_ = []

    def fit(self, pos, neg):
        self.pos_center_ = self.center(pos)
        self.neg_center_ = self.center(neg)

    @staticmethod
    def center(reviews):
        from operator import add
        import numpy as np
        summation = np.zeros(len(reviews[0]) - 1)
        for review in reviews:
            summation = map(add, summation, review[: -1])
        return [value / len(reviews) for value in summation]

    def predict(self, samples, targets):
        # samples argument ignored
        import random
        predictions = []
        candidates = ['+', '-']
        for target in targets:
            to_pos = self.distance(self.pos_center_, target)
            to_neg = self.distance(self.neg_center_, target)
            if to_pos > to_neg:
                predictions.append('-')
            elif to_neg > to_pos:
                predictions.append('+')
            else:
                predictions.append(candidates[random.randint(0, 1)])
        return predictions
