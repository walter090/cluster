

def run(k,
        mode='knn',
        distance='euclidean',
        keep_punc=False,
        keep_stopwords=False):
    from knn import KNNClassifier
    from ncc import NearestCentroidClassifier
    if mode == 'knn':
        clf = KNNClassifier(k, distance=distance, keep_stopwords=keep_stopwords, keep_punc=keep_punc)
        clf.score()
    elif mode == 'ncc':
        clf = NearestCentroidClassifier(k, distance=distance, keep_stopwords=keep_stopwords, keep_punc=keep_punc)
        clf.score()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    punc_parse = parser.add_mutually_exclusive_group()
    punc_parse.add_argument('--punc', dest='keep_punc', action='store_true')
    punc_parse.add_argument('--no-punc', dest='keep_punc', action='store_false')
    stop_parse = parser.add_mutually_exclusive_group()
    stop_parse.add_argument('--stop', dest='keep_stopwords', action='store_true')
    stop_parse.add_argument('--no-stop', dest='keep_stopwords', action='store_false')
    parser.add_argument('--distance', dest='distance', default='euclidean')
    parser.add_argument('-k', dest='k')

    parser.set_defaults(keep_punc=False, keep_stopwords=False)
    args = parser.parse_args()

    run(k=args.k, distance=args.distance, keep_punc=args.keep_punc, keep_stopwords=args.keep_stopwords)
