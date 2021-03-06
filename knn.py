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

    def fit(self, pos, neg):
        pass

    def k_neighbors(self, samples, target):
        import operator
        distances = []
        for i, point in enumerate(samples):
            distances.append((self.get_distance(point[: -1], target[: -1]), point[-1]))
        closest = sorted(distances, key=operator.itemgetter(0))[: self.k]
        ############
        # print(target[-1])
        # pred = list(zip(*closest))[1]
        # pos_vote = 0
        # neg_vote = 0
        # for point in pred:
        #     if point == '+':
        #         pos_vote += 1
        #     else:
        #         neg_vote += 1
        # print(pred)
        # print('pos {0} neg {1}'.format(pos_vote, neg_vote))
        # print('+' if pos_vote > neg_vote else '-')
        #################
        return list(zip(*closest))[1]

    def predict(self, samples, targets):
        import random
        closests = [self.k_neighbors(samples, targets[i]) for i in range(len(targets))]
        votes = []
        candidates = ['+', '-']
        for result in closests:
            pos_vote = 0
            neg_vote = 0
            for point in result:
                if point == '+':
                    pos_vote += 1
                else:
                    neg_vote += 1
            if pos_vote == neg_vote:
                votes.append(candidates[random.randint(0, 1)])
            else:
                votes.append('+' if pos_vote > neg_vote else '-')
        return votes
