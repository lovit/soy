from collections import defaultdict
from soy.utils import IntegerEncoder

import numpy as np


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


    def relative_proportion(self, words1=[], words2=[], min_proportion=0.7, topn=0):
        
        raise NotImplemented 
        
    
    def _relative_proportion(self, word1=None, words2=[], min_proportion=0.7, topn=0):
        
        # Find idx+, idx-
        
        # Merge frequency
        
        # Calculate score
        

        
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
