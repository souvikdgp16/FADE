import collections
import nltk


def get_tokens(corpus):
    return nltk.word_tokenize(corpus)


def unigram(corpus):
    tokens = get_tokens(corpus)
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model[f] = 1
            continue
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word] / N
    return model
