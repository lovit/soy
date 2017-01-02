from collections import defaultdict
import json
import pickle
import pprint
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import pairwise_distances
import numpy as np
from gensim.models import Word2Vec


class Word2Vec_NER_Trainer:

    def __init__(self, corpus_file):
        self.corpus_file = corpus_file


    def train_word2vec(self, min_count=10, size=100, window=5, workers=3):
        self.word2vec_model = Word2Vec(Word2vecCorpus(self.corpus_file), min_count=min_count, size=size, window=window, workers=workers)
        
        
    def load_word2vec(self, pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                self.word2vec_model = pickle.load(f)
                self._word2index()
        except Exception as e:
            print(e)

            
    def _word2index(self):
        self.word2index = defaultdict(lambda: -1, {word:i for i, word in enumerate(self.word2vec_model.index2word)})
    
    def most_similar(self, query, topn=10):
        return self.word2vec_model.most_similar(query, topn=topn)


    def extract_wordfilter(self, seed_words, ranges, min_count=1):
        word_filters = defaultdict(lambda: 0)

        for num_doc, doc in enumerate(Word2vecCorpus(self.corpus_file)):
            len_doc = len(doc)

            for i, word in enumerate(doc):
                if not word in seed_words:
                    continue

                for rng in ranges:
                    if (i + rng[0] >= 0) and (i + rng[1] < len_doc):
                        if rng[0] < 0:
                            context = tuple([doc[i+r] for r in range(rng[0],0)] + [doc[i+r] for r in range(1, rng[1]+1)])
                        
                        else:
                            context = tuple([''] + [doc[i+r] for r in range(1, rng[1]+1)])
                        word_filters[(rng, context)] += 1

        return {filter_:freq for filter_, freq in word_filters.items() if freq >= min_count}


    def train_wordfilter_coefficient(self, seed_words, wordfilters):
        mined_words = defaultdict(lambda: defaultdict(lambda: 0))
        filter_set = {wordfilter for (rng, wordfilter) in wordfilters}
        ranges = {rng for (rng, wordfilter) in wordfilters}

        for num_doc, doc in enumerate(Word2vecCorpus(self.corpus_file)):
            len_doc = len(doc)

            for rng in ranges:
                (fb, fe) = rng

                if len_doc < (fe - fb + 1):
                    continue

                words = doc[-fb:-fe]
                contexts = []

                for i, word in enumerate(doc):
                    if (i + fb < 0) or (i + fe >= len_doc):
                        continue
                    contexts.append(tuple([doc[i+r] for r in range(fb, fe+1) if r != 0]))

                for i, context in enumerate(contexts):
                    if context in filter_set:
                        mined_words[(rng, context)][words[i]] += 1

        result = []

        seeds_idx = sorted([self.word2index[seed] for seed in seed_words])
        seeds_vec = [self.word2vec_model.syn0[idx] for idx in seeds_idx]

        for ((rng, context), word2freq) in sorted(mined_words.items(), key=lambda x:sum(x[1].values()), reverse=True):

                word_freq = [(self.word2index[word], freq) for (word, freq) in word2freq.items()]
                word_freq = [v for v in word_freq if v[0] != -1]
                word_freq = sorted(word_freq)
                idx = [pair[0] for pair in word_freq]
                word_vec = self.word2vec_model.syn0[idx]
                sum_freq = sum([v[1] for v in word_freq])

                score = 0

                for seed_vec in seeds_vec:
                    sim = 1 + -1 * pairwise_distances(word_vec, seed_vec, metric='cosine')
                    score += sum([wf[1] * s for wf, s in zip(word_freq, sim)]) / sum_freq

                score /= len(seed_words)
                result.append((context, rng, score, sum_freq))

        return result


    def wrapping_filter(self, context, key, score,  frequency, entity_name):
        return {'C[-1]': ''.join(context[:-1*key[0]]), 
                'C[1]' : ''.join(context[-1*key[1]:]), 
                'W[%d]' % key[0]: list(context[:-1*key[0]]), 
                'W[%d]' % key[1]: list(context[-1*key[1]:]), 
                'coefficient': score,
                'training_frequency': frequency,
                'entity_name': entity_name}


class Word2vecCorpus:
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        with open(self.fname, encoding='utf-8') as f:
            for line in f:
                yield line.split()
