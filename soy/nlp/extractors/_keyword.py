from collections import defaultdict
import os
import pickle
import sys
import time
import numpy as np

from soy.utils import IntegerEncoder, progress

class PrecomputedAssociation:
    def __init__(self, scores=None):
        self._scores = scores if (not scores is None) else {}

    def save(self, fname):
        if fname[-4:] != '.pkl':
            fname = (fname + '.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(self._scores, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            self._scores = pickle.load(f)

    def get_mutual_informations(self, from_words=None, to_words=None, topn_for_a_from_word=0, min_mi=0):
        def check_from_words(args):
            if args is None:
                return tuple(self._scores.keys())
            elif type(args) == int:
                args = (args,)
            elif (type(args) == list) or (type(args) == tuple):
                pass
            else:
                raise TypeError('from_words type must be {None, int, or list/tuple of int}')
            return [w for w in args if w in self._scores]

        def check_to_words(args, w1):
            if args is None:
                args = tuple(self._scores.get(w1, {}).keys())
            elif type(args) == int:
                args = (args,)
            elif (type(args) == list) or (type(args) == tuple):
                pass
            else:
                raise TypeError('to_words type must be {None, int, or list/tuple of int}')
            return args

        from_words = check_from_words(from_words)
        if len(from_words) == 0:
            return []

        MI_w__ = []
        for w1 in from_words:
            to_words_of_w1 = check_to_words(to_words, w1)
            if not to_words_of_w1:
                continue
            MI_w1_ = []
            for w2 in to_words_of_w1:
                mi_w12 = self._scores.get(w1, {}).get(w2, None)
                if (mi_w12 is not None) and (mi_w12 > min_mi):
                    MI_w1_.append((w1, w2, mi_w12))
            if not MI_w1_:
                continue
            if topn_for_a_from_word > 0:
                MI_w1_ = MI_w1_[:topn_for_a_from_word]
            MI_w__ += MI_w1_

        MI_w__ = sorted(MI_w__, key=lambda x:x[2], reverse=True)
        return MI_w__

    
class Association:

    def __init__(self, autobase_target=3.5, autobase_topn=20,
                 autobase_candidates=(1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2)):
        self.autobase_target = autobase_target
        self.autobase_topn = autobase_topn
        self.autobase_candidates = autobase_candidates
        self._P_w12 = {}
        self._F_w1 = {}
        self._F_w2 = {}
        self._P_w1 = {}
        self._P_w2 = {}
        self._verbose = 10000

    def train(self, from_to_encoded_pairs):
        def pair_type_check(pair):
            if len(pair) != 2:
                raise ValueError('Pair length must be two')
            for sent in pair:
                for w in sent:
                    if type(w) != int:
                        raise ValueError('Type of sent in pair must be int')
            return True

        F_w12 = defaultdict(lambda: defaultdict(lambda: 0))
        for num_pair, pair in enumerate(from_to_encoded_pairs):
            pair_type_check(pair)
            from_words, to_words = pair
            for w1 in from_words:
                for w2 in to_words:
                    F_w12[w1][w2] += 1
                self._F_w1[w1] = (self._F_w1.get(w1, 0) + 1)
            for w2 in to_words:
                self._F_w2[w2] = (self._F_w2.get(w2, 0) + 1)
            if num_pair % self._verbose == 0:
                sys.stdout.write('\rassociation training ... %d in %d' % (num_pair + 1, len(from_to_encoded_pairs)))

        for w1, F_w2 in F_w12.items():
            sum_w12 = sum(F_w2.values())
            P_w2 = {w2: f / sum_w12 for w2, f in F_w2.items()}
            self._P_w12[w1] = P_w2

        sum_w1 = sum(self._F_w1.values())
        for w1, f_w1 in self._F_w1.items():
            self._P_w1[w1] = f_w1 / sum_w1

        sum_w2 = sum(self._F_w2.values())
        for w2, f_w2 in self._F_w2.items():
            self._P_w2[w2] = f_w2 / sum_w2

        print('\rassociation training was done')

    def get_all_autobase_mutual_information(self, topn_for_a_from_word=10000, min_mi=0, base=0):
        MI_all = self.get_mutual_informations(None, None, topn_for_a_from_word, min_mi, base)
        precomputed_MI = defaultdict(lambda: {})
        for mi_w12 in MI_all:
            w1 = mi_w12[0]
            w2 = mi_w12[1]
            precomputed_MI[w1][w2] = mi_w12[2]
        return dict(precomputed_MI)

    def get_mutual_informations(self, from_words=None, to_words=None, topn_for_a_from_word=0, min_mi=-10000,
                                base=0):
        def check_from_words(args):
            if (args is None):
                return tuple(self._P_w12.keys())
            elif type(args) == int:
                args = (args,)
            elif (type(args) == list) or (type(args) == tuple):
                if len(args) == 0:
                    return tuple(self._P_w12.keys())
                else:
                    pass
            else:
                raise TypeError('from_words type must be {None, int, or list/tuple of int}')
            return [w for w in args if w in self._P_w12]

        def check_to_words(args, w1):
            if args is None:
                args = []
            elif type(args) == int:
                args = (args,)
            elif (type(args) == list) or (type(args) == tuple):
                pass
            else:
                raise TypeError('to_words type must be {None, int, or list/tuple of int}')
            if len(args) == 0:
                args = tuple(self._P_w12.get(w1, {}).keys())
            return args

        from_words = check_from_words(from_words)
        if len(from_words) == 0:
            return []

        MI_w__ = []
        for w1 in from_words:
            to_words_of_w1 = check_to_words(to_words, w1)
            if not to_words_of_w1: 
                continue
            if base == 0:
                MI_w__ += self._get_autobase_mutual_information(w1, to_words_of_w1, min_mi, topn_for_a_from_word)
            else:
                MI_w__ += self._get_mutual_informations(w1, to_words_of_w1, base, min_mi, topn_for_a_from_word)
        return MI_w__

    def _get_autobase_mutual_information(self, a_from_word, to_words, min_mi, topn):
        MI_base = []
        for base in self.autobase_candidates:
            MI_w1_ = self._get_mutual_informations(a_from_word, to_words, base, min_mi, topn)
            if not MI_w1_:
                continue
            mi_avg = np.mean([v[2] for v in MI_w1_[:self.autobase_topn]])
            target_diff = abs(mi_avg - self.autobase_target)
            MI_base.append((target_diff, MI_w1_))
        if not MI_base:
            return []
        return sorted(MI_base, key=lambda x: x[0])[0][1]

    def _get_mutual_informations(self, a_from_word, to_words, base, min_mi, topn):
        def calculate_mi(p_w12, p_w2, base):
            return np.log((p_w12 + 1e-15) / (p_w2 + base))

        P_w12 = self._P_w12.get(a_from_word, {})
        if not P_w12:
            return []

        MI_w1_ = []
        for w2 in to_words:
            p_w12 = P_w12.get(w2, 0)
            if p_w12 == 0:
                continue
            p_w2 = self._P_w2[w2]
            mi_w12 = calculate_mi(p_w12, p_w2, base)
            if mi_w12 < min_mi:
                continue
            MI_w1_.append((a_from_word, w2, mi_w12))

        if not MI_w1_:
            return []
        if topn > 0:
            MI_w1_ = MI_w1_[:topn]
        return sorted(MI_w1_, key=lambda x: x[2], reverse=True)

    def save(self, fname):
        if fname[-4:] != '.pkl':
            fname = (fname + '.pkl')

        with open(fname, 'wb') as f:
            params = {
                'autobase_target': self.autobase_target,
                'autobase_topn': self.autobase_topn,
                'autobase_candidates': self.autobase_candidates,
                'P_w12': self._P_w12,
                'P_w1': self._P_w1,
                'P_w2': self._P_w2,
                'F_w1': self._F_w1,
                'F_w2': self._F_w2
            }
            pickle.dump(params, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            params = pickle.load(f)
            self.autobase_target = params.get('autobase_target', self.autobase_target)
            self.autobase_topn = params.get('autobase_topn', self.autobase_topn)
            self.autobase_candidates = params.get('autobase_candidates', self.autobase_candidates)
            self._P_w12 = params.get('P_w12', {})
            self._P_w1 = params.get('P_w1', {})
            self._P_w2 = params.get('P_w2', {})
            self._F_w1 = params.get('F_w1', {})
            self._F_w2 = params.get('F_w2', {})
    
'''
class KeywordExtractor:
    
    def __init__(self, tokenize=lambda x:x.split()):
        self.encoder = IntegerEncoder()
        self.doc2term = defaultdict(lambda: defaultdict(lambda: 0))
        self.term2doc = defaultdict(lambda: defaultdict(lambda: 0))
        self.tokenize = tokenize
        self.vocabs = defaultdict(lambda: 0)

        
    def add_docs(self, docs, normalize=False):
        for doc_id, doc in enumerate(docs):
            terms = [self.encoder.encode(term) for term in self.tokenize(doc)]
            terms = [(term, 1) for term in terms if term != -1]
            if not terms:
                continue
            if normalize:
                v = 1/len(terms)
                terms = [(term, v) for (term, _) in terms]
	        for (term, v) in terms:
                self.doc2term[doc_id][term] += v
                self.term2doc[term][doc_id] += v
        
            
    def scan_vocabs(self, docs, verbose=True):
        self.vocabs = defaultdict(lambda: 0)
        for num_doc, doc in enumerate(docs):
            for term in self.tokenize(doc):
                self.vocabs[term] += 1
        
        
    def set_vocabs(self, min_count=1):
        vocabs_ = {term:freq for term, freq in self.vocabs.items() if freq >= min_count}
        for pair in sorted(vocabs_.items(), key=lambda x:x[1], reverse=True):
            self.encoder.fit(pair[0])
        print('num of terms = %d' % len(vocabs_))
        self.count = defaultdict(lambda: 0, {self.encoder.encode(term):freq for term, freq in vocabs_.items()})
        
    def relative_proportion(self, base_words=[], target_words=[], min_proportion=0.7, topn=0, base_min_count=0, target_min_count=0, as_str=False, verbose=True):

        if base_words and type(base_words[0]) == str:
            base_words = {self.encoder.encode(w1) for w1 in base_words}
            base_words = [w for w in base_words if w != -1 and self.count.get(w,0) >= base_min_count]
        elif base_words and type(base_words[0]) == int:
            base_words = [w for w in base_words if w >= 0 and w < len(self.encoder.inverse) and self.count.get(w,0) >= base_min_count]
        else:
            base_words = [w for w in self.term2doc.keys() if self.count.get(w,0) >= base_min_count]
            
        if not base_words:
            return []
        
        if not target_words:
            target_words = {w for w in self.term2doc.keys() if self.count.get(w,0) >= target_min_count}
        elif type(target_words[0]) == str: 
            target_words = {self.encoder.encode(w) for w in target_words}
            target_words = {w for w in target_words if w != -1 and self.count.get(w,0) >= target_min_count}
        else:
            target_words = {w for w in target_words if w >= 0 and w < len(self.encoder.inverse) and self.count.get(w,0) >= target_min_count}

        rp = []
        base_time = time.time()
        
        for num_wb, wb in enumerate(base_words):
            rp.append(self._relative_proportion(wb, target_words, min_proportion))
            if (verbose) and (num_wb % 10 == 0):
                sys.stdout.write('\r%s' % progress(num_wb + 1, len(base_words), header='Relative proportion', base_time=base_time))
        if verbose:
            print('\rRelative proportion was done')
                
        if topn > 0:
            rp = [rp_[:topn] for rp_ in rp]
            
        if as_str:
            return [[self._to_str(pair) for pair in rp_] for rp_ in rp]
        
        return rp
    
    
    def _relative_proportion(self, base_word=None, target_words={-1}, min_proportion=0.7):
        
        def merge_rows(docs, doc2term):
            vector = defaultdict(lambda: 0)
            for doc in docs:
                for term, f in doc2term[doc].items():
                    vector[term] += f
            sum_ = sum(vector.values())
            for term in vector.keys():
                vector[term] = vector[term] / sum_
            return vector
        
        # idx+, idx-
        docs_p = set(self.term2doc[base_word])
        docs_n = set(self.doc2term.keys()) - docs_p
        
        # Proportion
        prop_p = merge_rows(docs_p, self.doc2term)
        prop_n = merge_rows(docs_n, self.doc2term)
        
        # Calculate score
        score = [(base_word, word, (prop_p.get(word,0) / (prop_p.get(word,0) + prop_n.get(word,0)))) for word in target_words if word >= 0 and word != base_word]
        score = sorted([pair for pair in score if pair[2] >= min_proportion], key=lambda x:x[2], reverse=True)
        
        return score
        
        
    def _to_str(self, pair):
        return (self.encoder.decode(pair[0]), self.encoder.decode(pair[1])) + pair[2:]
    
    
    def save_index(self, fname):
        
        raise NotImplemented
        
        
    def save_matrix_as_txt(self, fname):
        
        raise NotImplemented
        
        
    def save_matrix_as_mm(self, fname):
        
        raise NotImplemented
        
    
    def load_index(self, fname):
        self.encoder.load(fname)
        
    
    def load_mm(self, mm_file):
        
        try:
            with open(mm_file, encoding='utf-8') as f:
                for num, line in enumerate(f):
                    if num < 3:
                        continue
                        
                    (doc, term, v) = [int(c) for c in line.split()]
                    doc -= 1
                    term -= 1
                    
                    self.doc2term[doc][term] = v
                    self.term2doc[term][doc] = v
                    
        except Exception as e:
            print(e)
'''
