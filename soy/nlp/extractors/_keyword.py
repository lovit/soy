from collections import defaultdict
import sys
import time

import numpy as np

from soy.utils import IntegerEncoder, progress


class Association:
    
    def __init__(self):
        self.encoder = IntegerEncoder()
        self.w1_w2 = defaultdict(lambda: defaultdict(lambda: 0))
        self.w2_w1 = defaultdict(lambda: defaultdict(lambda: 0))
        self.pw2 = defaultdict(lambda: 0)
        
    def add(self, words1, words2):
        words1 = [self.encoder.fit(w) for w in words1]
        words2 = [self.encoder.fit(w) for w in words2]
        
        for w1 in words1:
            for w2 in words2:
                self.w1_w2[w1][w2] += 1
                self.w2_w1[w2][w1] += 1
    

    def _to_str(self, pair):
        return (self.encoder.decode(pair[0]), self.encoder.decode(pair[1])) + pair[2:]
    
    
    
    def conditional_probability(self, words1=[], words2=[], as_str=False):
        
        if words1 and type(words1[0]) == str:
            words1 = {self.encoder.encode(w1) for w1 in words1}
            words1 = [w1 for w1 in words1 if w1 != -1]
        
        if not words1:
            return []
        
        if not words2:
            words2 = {w2 for w1 in words1 for w2 in self.w1_w2[w1].keys()}
        elif type(words2[0]) == str:
            words2 = {self.encoder.encode(w2) for w2 in words2}
            words2 = [w2 for w2 in words2 if w2 != -1]
        else:
            words2 = [w2 for w2 in words2 if w2 >= 0 and w2 < len(self.encoder.inverse)]
        
        cp = []
            
        for w1 in words1:
            w2_dict = self.w1_w2[w1]
            w2_sum = sum([f for f in self.w1_w2[w1].values()])
            
            if w2_sum == 0:
                continue
                
            for w2 in words2:
                cp_value = w2_dict.get(w2, 0) / w2_sum
                cp.append((w1, w2, cp_value, self.pw2[w2]))
        
        if as_str:
            return [self._to_str(pair) for pair in cp]
        else:
            return cp
        

    def mutual_information(self, words1=[], words2=[], mutual_information_topn=0, base=0.001, autobase_target=3.5, autobase_topn=20, as_str=False):
    
        if len(self.pw2) == 0:
            self._set_ready()
    
        if words1 and type(words1[0]) == str:
            words1 = {self.encoder.encode(w1) for w1 in words1}
            words1 = [w1 for w1 in words1 if w1 != -1]
        else:
            words1 = [w1 for w1 in words1 if w1 >= 0 and w1 < len(self.encoder.inverse)]
            
        if not words1:
            return []
        
        if not words2:
            words2 = list({w2 for w1 in words1 for w2 in self.w1_w2[w1].keys()})
        elif type(words2[0]) == str: 
            words2 = {self.encoder.encode(w2) for w2 in words2}
            words2 = [w2 for w2 in words2 if w2 != -1]
        else:
            words2 = [w2 for w2 in words if w2 >= 0 and w2 < len(self.encndoer.inverse)]

        mi = []
        
        for w1 in words1:
            if base == 0:
                mi.append(self._autobase_mutual_information(w1, words2, autobase_target, autobase_topn))
            else:
                mi.append(self._mutual_information(w1, words2, base))
            
        if mutual_information_topn > 0:
            mi = [mi_[:mutual_information_topn] for mi_ in mi]
            
        if as_str:
            return [[self._to_str(pair) for pair in mi_] for mi_ in mi]
        
        return mi
    
    
    def _mutual_information(self, word1=None, words2=[], base=0.001):

        def get_mi(prob_w2w1, prob_w2, base=0.001):
            return np.log( (prob_w2w1 + 1e-15) / (prob_w2 + base) )
        
        mi = []
        
        w2_dict = self.w1_w2[word1]
        w2_sum = sum([f for f in self.w1_w2[word1].values()])

        if w2_sum == 0:
            return mi

        for w2 in words2:
            pw2w1 = w2_dict.get(w2, 0)/w2_sum
            mi_value = get_mi(pw2w1, self.pw2[w2], base)
            mi.append((word1, w2, mi_value, self.pw2[w2], pw2w1))

        return sorted(mi, key=lambda x:x[2], reverse=True)
    
    
    def _autobase_mutual_information(self, word1=None, words2=[], autobase_target=3.5, autobase_topn=20):
        
        bases = [float('0.'+'0'*decimal+str(i)) for decimal in range(2, 7) for i in [1, 5]]
        
        mi_base = []
        top_avg = []
        
        for base in bases:
            
            mi = self._mutual_information(word1, words2, base)
            mi_base.append( ( base, mi ) )
            top_avg.append( np.mean( [v[2] for v in mi[:autobase_topn]] ) )
        
        best_mi = [(base, abs(avg - autobase_target)) for base, avg in zip(bases, top_avg)]
        best_mi = sorted(best_mi, key=lambda x:x[1])[0][0]
        
        for (base, mi) in mi_base:
            if base == best_mi:
                return mi
    
    
    def _set_ready(self):
        print('First time. Set ready!')
        
        for w2, w1_dict in self.w2_w1.items():
            self.pw2[w2] = sum(w1_dict.values())
        sum_ = sum(self.pw2.values())
        
        for w2 in self.w2_w1.keys():
            self.pw2[w2] /= sum_
        print('  - done')


        
    def save(self, model_prefix):
        try:
            folder = '/'.join(model_prefix.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            fname = model_prefix + '_association_graph'
            with open(fname, 'w', encoding='utf-8') as f:
                for w1, w2_dict in self.w1_w2.items():
                    for w2, v in w2_dict.items():
                        f.write('%d %d %d\n' % (w1, w2, int(v)))
                        
            fname = model_prefix + '_association_encoder'
            self.encoder.save(fname)
        except Exception as e:
            print(e)
            print('filename = %s' % fname)
    
        
    def load(self, model_prefix):
        try:
            fname = model_prefix + '_association_graph'
            with open(fname, encoding='utf-8') as f:
                
                self.w1_w2 = defaultdict(lambda: defaultdict(lambda: 0))
                self.w2_w1 = defaultdict(lambda: defaultdict(lambda: 0))
                self.pw2 = defaultdict(lambda: 0)

                for line in f:
                    w1, w2, v = line.split()
                    w1 = int(w1)
                    w2 = int(w2)
                    v = int(v)
                    self.w1_w2[w1][w2] = v
                    self.w2_w1[w2][w1] = v
            
            fname = model_prefix + '_association_encoder'
            self.encoder.load(fname)
            
        except Exception as e:
            print(e)
            print('filename = %s' % fname)


    def load_index(self, fname):
        self.encoder.load(fname)
        
    
    def load_mm(self, mm_file):
        
        try:
            with open(mm_file, encoding='utf-8') as f:
                for num, line in enumerate(f):
                    if num < 3:
                        continue
                        
                    (w1, w2, v) = [int(c) for c in line.split()]
                    w1 -= 1
                    w2 -= 1
                    
                    self.w1_w2[w1][w2] = v
                    self.w2_w1[w2][w1] = v
                    
        except Exception as e:
            print(e)




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
        elif base_words and type(base_word[0]) == int:
            base_words = [w for w in base_words if w1 >= 0 and w < len(self.encoder.inverse) and self.count.get(w,0) >= base_min_count]
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
            target_words = {w for w in target_words if w >= 0 and w < len(self.encndoer.inverse) and self.count.get(w,0) >= target_min_count}

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
              
