from collections import defaultdict
import os
import sys

from soy.ml.graph import dict_graph
from soy.nlp.extractors import CohesionProbability
from soy.utils import IntegerEncoder


class LRNounExtractor:
    
    def __init__(self, r_score_file=None):
        if (r_score_file == None) or (not os.path.exists(r_score_file)):
            print('Use default r_score_file')
            r_score_file = __file__[:-9] + 'noun_score_sejong'
        self._load_r_score(r_score_file)
        
        
    def _load_r_score(self, fname):
        self.r_score = {}
        try:
            with open(fname, encoding='utf-8') as f:
                for num_line, line in enumerate(f):
                    r, score = line.split('\t')
                    score = float(score)
                    self.r_score[r] = score
            print('%d r features was loaded' % len(self.r_score))
        except FileNotFoundError:
            print('r_score_file was not found')
        except Exception as e:
            print('%s parsing error line (%d) = %s' % (e, num_line, line))
    
    
    def predict(self, r_features):
        '''
        Parameters
        ----------
            r_features: dict
                r 빈도수 dictionary
                예시: {을: 35, 는: 22, ...}
        '''
        
        score = 0
        norm = 0
        unknown = 0
        
        for r, freq in r_features.items():
            if r in self.r_score:
                score += freq * self.r_score[r]
                norm += freq
            else:
                unknown += freq
        
        return (0 if norm == 0 else score / norm, 
                0 if (norm + unknown == 0) else norm / (norm + unknown))

    def build_lrgraph(self, docs, max_l_length=10, max_r_length=6, min_count=10):    
        """
        Parameters
        ----------
            docs: iterable object which has string
            
            max_l_length: int
            
            max_r_length: int
            
            min_count: int
            
        It computes subtoken frequency first. 
        After then, it builds lr-graph with sub-tokens appeared at least min count
        """
        
        _ckpt = int(len(docs) / 40)
        
        counter_l = defaultdict(lambda: 0)
        counter_r = defaultdict(lambda: 0)
        
        for num_doc, doc in enumerate(docs):
            for sent in doc.split('  '):
                for token in sent.split():

                    if not token:
                        continue

                    token_len = len(token)

                    for i in range(1, min(max_l_length, token_len)+1):
                        counter_l[token[:i]] += 1

                    for i in range(1, min(max_r_length, token_len)):
                        counter_r[token[-i:]] += 1
            
            if num_doc % _ckpt == 0:
                sys.stdout.write('\rscanning: %s%s (%.3f %s)' % 
                                 ('#' * int(num_doc/_ckpt),
                                  '-' * (40 - int(num_doc/_ckpt)), 
                                  100.0 * num_doc / len(docs), '%') )
        print('\rscanning completed')
        
        def filter_extreme(d, min_count):
            return {k:v for k,v in d.items() if v > min_count}
        
        counter_l = filter_extreme(counter_l, min_count)
        counter_r = filter_extreme(counter_r, min_count)
        counter_r[''] = 0
        print('(L,R) has (%d, %d) tokens' % (len(counter_l), len(counter_r)))
        
        lr_graph = dict_graph()
        encoder = IntegerEncoder()
        encoder.fit('')
        
        for num_doc, doc in enumerate(docs):
            for sent in doc.split('  '):
                for token in sent.split():    

                    if not token:
                        continue

                    token_len = len(token)

                    for i in range(1, min(max_l_length, token_len)+1):

                        l = token[:i]
                        r = token[i:]

                        if (not l in counter_l) or (not r in counter_r):
                            continue

                        l = encoder.fit(l)
                        r = encoder.fit(r)
                        lr_graph.add(l, r, 1)

            if num_doc % _ckpt == 0:
                sys.stdout.write('\rbuilding lr-graph: %s%s (%.3f %s)' % 
                                 ('#' * int(num_doc/_ckpt),
                                  '-' * (40 - int(num_doc/_ckpt)), 
                                  100.0 * num_doc / len(docs), '%') )
        sys.stdout.write('\rbuilding lr-graph completed')
        
        del counter_l
        del counter_r
                        
        return lr_graph, encoder


    def extract(self, docs, noun_threshold=0.05, known_threshold=0.05, word_extraction='cohesion', noun_candidates={}, kargs={}):
        """
        Parameters
        ----------
            docs: list of str

            nounscore_threshold: float

            known_threshold: float

            word_extraction: str
                possible value = ['cohesion', 'branch']

        Returns
        -------
            noun list
        """

        lr_graph, encoder = self.build_lrgraph(docs)

        class sents:        
            def __init__(self, docs):
                self.docs = docs
                self.num_sent = 0
                for doc in docs:
                    for sent in doc.split('  '):
                        self.num_sent += 1
            def __iter__(self):
                for doc in docs:
                    for sent in doc.split('  '):
                        yield sent
            def __len__(self):
                return self.num_sent

        if word_extraction == 'cohesion':

            cohesion_min_count = kargs.get('cp_min_count', 10)
            cohesion_min_probability = kargs.get('cp_min_prob', 0.05)
            cohesion_min_droprate = kargs.get('cp_min_droprate', 0.8)

            cohesion = CohesionProbability(kargs.get('cp_min_l',1), kargs.get('cp_max_l',10), kargs.get('cp_min_r',1), kargs.get('cp_max_r',6))
            cohesion.train(docs)
            cohesion.prune_extreme_case(cohesion_min_count)

            noun_candidates = cohesion.extract(min_count=cohesion_min_count, min_droprate=cohesion_min_droprate, min_cohesion=(cohesion_min_probability, 0), remove_subword=True)
            noun_candidates = {k:v for k,v in noun_candidates.items() if v[0] > cohesion_min_probability}

        # Prediction
        if not noun_candidates:
            noun_candidates = {}
            print('cannot find word candidates')

        nouns = dict()
        noun_candidates = sorted(noun_candidates.items(), key=lambda x:len(x[0]))

        for word, word_score in noun_candidates:
            if not word in encoder.mapper:
                continue

            r_features = lr_graph.outb(encoder.encode(word))
            r_features = {encoder.decode(r, unknown='Unk'):f for r,f in r_features.items()}
            if 'Unk' in r_features: del r_features['Unk']

            if (not r_features) or (list(r_features.keys()) == ''):
                for e in range(1, len(word) + 1):
                    subword = word[:e]
                    suffix = word[e:]

                # Add word if compound
                if (subword in nouns) and (suffix in nouns):
                    score1 = nouns[subword]
                    score2 = nouns[suffix]
                    nouns[word] = score1 if score1[0] > score2[0] else score2
                    break

                if (subword in nouns) and (self.r_score.get(suffix,0.0) < noun_threshold):
                    break

            noun_score = self.predict(r_features)

            if noun_score[0] > noun_threshold:
                nouns[word] = noun_score

        # 공교롭게도 = 공교롭게 + 도
        removals = set()
        for word in sorted(nouns.keys(), key=lambda x:len(x[0]), reverse=True):
             if len(word) <= 2:
                 continue

             if word[-1] == '.':
                 removals.add(word)

             for e in range(1, len(word)):
                if (word[:e] in nouns) and (self.r_score.get(word[e:], 0.0) > noun_threshold):
                    removals.add(word)
                    break

        for removal in removals:
            del nouns[removal]        

        self.nouns = nouns
        self.lr_graph = lr_graph
        self.lr_graph_encoder = encoder

        return nouns, cohesion 

 
    def extract_and_transform(self, docs, min_count = 10):
        
        self.extract(docs)
        self.transform(docs, min_count)


    def transform(self, doc, noun_set=None):
        if noun_set == None:
            noun_set = self.nouns.keys()

        def left_match(word):
            for i in range(1, len(word) + 1):
                if word[:i] in noun_set:
                    return word[:i]
            return ''

        noun_doc = [[left_match(word) for word in sent.split()] for sent in doc.split('  ')]
        noun_doc = [[word for word in sent if word] for sent in noun_doc]
        return noun_doc
    

    def _postprocessing(self, noun_candidates, lr_graph):
        
        raise NotImplementedError('LRNounExtractor should implement')
