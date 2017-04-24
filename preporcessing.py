import os
from collections import Counter
from nltk.tokenize import RegexpTokenizer, word_tokenize


def preprocess(path, punctuation=False):
    all_reviews = os.listdir(path)
    tokenizer = RegexpTokenizer(r'\w+')

    review_articles = []
    for review in all_reviews:
        with open(os.path.join(path, review)) as reader:
            review_str = reader.read()
            review_str = tokenizer.tokenize(review_str) if not punctuation\
                else word_tokenize(review_str)
            review_articles.append(review_str)

    return review_articles


def vocab(reviews):
    vocabulary = set()
    for review in reviews:
        for word in review:
            vocabulary.add(word)
    return list(vocabulary)


def get_stopwords(reviews):
    vocab_dup = []
    for review in reviews:
        for word in review:
            vocab_dup.append(word)
    stop_words = Counter(vocab_dup).most_common(30)
    return stop_words


def remove_stopwords(reviews, stopwords):
    return [[word for word in review if word not in stopwords]
            for review in reviews]


def vectorize(review, sentiment, vocabulary, vec='frequency'):
    assert vec == 'frequency' or vec == 'binary'
    import numpy as np
    vector = np.zeros(len(vocabulary))
    for i, word in enumerate(vocabulary):
        if vec == 'frequency':
            vector[i] = review.count(word)
        elif vector[i]:
            continue
    return vector + sentiment


def run(keep_punc=False, keep_stopwords=False):
    pos = preprocess('data/pos', keep_punc)
    neg = preprocess('data/neg', keep_punc)
    vocabulary = vocab(pos + neg)
    stopwords = get_stopwords(vocabulary)
    if not keep_stopwords:
        pos = remove_stopwords(pos, stopwords)
        neg = remove_stopwords(neg, stopwords)
    return pos, neg, vocabulary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    punc_parse = parser.add_mutually_exclusive_group()
    punc_parse.add_argument('--punc', dest='keep_punc', action='store_true')
    punc_parse.add_argument('--no-punc', dest='keep_punc', action='store_false')
    stop_parse = parser.add_mutually_exclusive_group()
    stop_parse.add_argument('--stop', dest='keep_stopwords', action='store_true')
    stop_parse.add_argument('--no-stop', dest='keep_stopwords', action='store_false')

    parser.set_defaults(keep_punc=False, keep_stopwords=False)
    args = parser.parse_args()

    run(keep_punc=args.keep_punc, keep_stopwords=args.keep_stopwords)