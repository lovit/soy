from collections import defaultdict
import math
import sys

import numpy as np

from soy.utils._utils import IntegerEncoder




class CohesionProbability:
    
    def __init__(self, left_min_length=2, left_max_length=10, right_min_length=1, right_max_length=6):
        
        self.left_min_length = left_min_length
        self.left_max_length = left_max_length
        self.right_min_length = right_min_length
        self.right_max_length = right_max_length
        
        self.L = defaultdict(int)
        self.R = defaultdict(int)
    
    
    def get_cohesion_probability(self, word):
        
        if not word:
            return (0, 0, 0, 0)
        
        word_len = len(word)

        l_freq = 0 if not word in self.L else self.L[word]
        r_freq = 0 if not word in self.R else self.R[word]
        
        l_cohesion = 0
        r_cohesion = 0
        
        # forward cohesion probability (L)
        if (self.left_min_length <= word_len) and (word_len <= self.left_max_length):
            
            l_sub = word[:self.left_min_length]
            l_sub_freq = 0 if not l_sub in self.L else self.L[l_sub]
            
            if l_sub_freq > 0:
                l_cohesion = np.power( (l_freq / float(l_sub_freq)), (1 / (word_len - len(l_sub) + 1.0)) )
        
        # backward cohesion probability (R)
        if (self.right_min_length <= word_len) and (word_len <= self.right_max_length):
            
            r_sub = word[-1 * self.right_min_length:]
            r_sub_freq = 0 if not r_sub in self.R else self.R[r_sub]
            
            if r_sub_freq > 0:
                r_cohesion = np.power( (r_freq / float(r_sub_freq)), (1 / (word_len - len(r_sub) + 1.0)) )
            
        return (l_cohesion, r_cohesion, l_freq, r_freq)

    
    def get_all_cohesion_probabilities(self):
        
        cp = {}
        words = set(self.L.keys())
        for word in self.R.keys():
            words.add(word)
        
        for word in words:
            cp[word] = self.get_cohesion_probability(word)
            
        return cp
        
        
    def counter_size(self):
        return (len(self.L), len(self.R))
    
                            
    def prune_extreme_case(self, min_count):
        
        before_size = self.counter_size()
        self.L = defaultdict(int, {k:v for k,v in self.L.items() if v > min_count})
        self.R = defaultdict(int, {k:v for k,v in self.R.items() if v > min_count})
        after_size = self.counter_size()
    
        return (before_size, after_size)
        
        
    def train(self, sents, num_for_pruning = 0, min_count = 5):
        
        for num_sent, sent in enumerate(sents):            
            for word in sent.split():
                
                if not word:
                    continue
                    
                word_len = len(word)
                
                for i in range(self.left_min_length, min(self.left_max_length, word_len)+1):
                    self.L[word[:i]] += 1
                
#                 for i in range(self.right_min_length, min(self.right_max_length, word_len)+1):
                for i in range(self.right_min_length, min(self.right_max_length, word_len)):
                    self.R[word[-i:]] += 1
                    
            if (num_for_pruning > 0) and ( (num_sent + 1) % num_for_pruning == 0):
                self.prune_extreme_case(min_count)
                
        if (num_for_pruning > 0) and ( (num_sent + 1) % num_for_pruning == 0):
                self.prune_extreme_case(min_count)
                
                    
    def load(self, fname):
        try:
            with open(fname, encoding='utf-8') as f:
                
                next(f) # SKIP: parameters(left_min_length left_max_length ...
                token = next(f).split()
                self.left_min_length = int(token[0])
                self.left_max_length = int(token[1])
                self.right_min_length = int(token[2])
                self.right_max_length = int(token[3])
                
                next(f) # SKIP: L count
                is_right_side = False
                
                for line in f:
                    
                    if '# R count' in line:
                        is_right_side = True
                        continue
                        
                    token = line.split('\t')
                    if is_right_side:
                        self.L[token[0]] = int(token[1])
                    else:
                        self.R[token[0]] = int(token[1])
                        
        except Exception as e:
            print(e)
            
        
    def save(self, fname):
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                
                f.write('# parameters(left_min_length left_max_length right_min_length right_max_length)\n')
                f.write('%d %d %d %d\n' % (self.left_min_length, self.left_max_length, self.right_min_length, self.right_max_length))
                
                f.write('# L count')
                for word, freq in self.L.items():
                    f.write('%s\t%d\n' % (word, freq))
                    
                f.write('# R count')
                for word, freq in self.R.items():
                    f.write('%s\t%d\n' % (word, freq))                
                    
        except Exception as e:
            print(e)

    
    def words(self):
        words = set(self.L.keys())
        words = words.union(set(self.R.keys()))
        return words




class BranchingEntropy:
    
    def __init__(self, min_length=2, max_length=7):
        
        self.min_length = min_length
        self.max_length = max_length
        
        self.encoder = IntegerEncoder()
        
        self.L = defaultdict(lambda: defaultdict(int))
        self.R = defaultdict(lambda: defaultdict(int))
    
    
    def get_all_access_variety(self):

        av = {}
        words = set(self.L.keys())
        words += set(self.R.keys())
        
        for word in words:
            av[word] = self.get_access_variety(word)
            
        return av

    
    def get_access_variety(self, word, ignore_space=False):
        
        return (len(self.get_left_branch(word, ignore_space)), len(self.get_right_branch(word, ignore_space)))
        
        
    def get_all_branching_entropies(self, ignore_space=False):
        
        be = {}
        words = set(self.L.keys())
        for word in self.R.keys():
            words.add(word)
        
        for word in words:
            be[self.encoder.decode(word)] = self.get_branching_entropy(word, ignore_space)
            
        return be

    
    def get_branching_entropy(self, word, ignore_space=False):
        
        be_l = self.entropy(self.get_left_branch(word, ignore_space))
        be_r = self.entropy(self.get_right_branch(word, ignore_space))
        return (be_l, be_r)

    
    def entropy(self, dic):
        
        if not dic:
            return 0.0
        
        sum_count = sum(dic.values())
        entropy = 0
        
        for freq in dic.values():
            prob = freq / sum_count
            entropy += prob * math.log(prob)
            
        return -1 * entropy

    
    def get_left_branch(self, word, ignore_space=False):
        
        if isinstance(word, int):
            word_index = word
        else:
            word_index = self.encoder.encode(word)
            
        if (word_index == -1) or (not word_index in self.L):
            return {}
        
        branch = self.L[word_index]
        
        if ignore_space:
            return {w:f for w,f in branch.items() if not ' ' in self.encoder.decode(w, unknown=' ')}
        else:
            return branch
        
    
    def get_right_branch(self, word, ignore_space=False):
        
        if isinstance(word, int):
            word_index = word
        else:
            word_index = self.encoder.encode(word)
            
        if (word_index == -1) or (not word_index in self.R):
            return {}
        
        branch = self.R[word_index]
        
        if ignore_space:
            return {w:f for w,f in branch.items() if not ' ' in self.encoder.decode(w, unknown=' ')}
        else:
            return branch
        
        
    def counter_size(self):
        return (len(self.L), len(self.R))
    
                            
    def prune_extreme_case(self, min_count):
        
        # TODO: encoder remove & compatify
        before_size = self.counter_size()
        self.L = defaultdict(lambda: defaultdict(int), {word:dic for word,dic in self.L.items() if sum(dic.values()) > min_count})
        self.R = defaultdict(lambda: defaultdict(int), {word:dic for word,dic in self.R.items() if sum(dic.values()) > min_count})
        after_size = self.counter_size()

        return (before_size, after_size)
        
        
    def train(self, sents, min_count=5, num_for_pruning = 10000):
        
        for num_sent, sent in enumerate(sents):

            sent = sent.strip()
            if not sent:
                continue

            sent = ' ' + sent.strip() + ' '
            length = len(sent)

            for i in range(1, length - 1):
                for window in range(self.min_length, self.max_length + 1):

                    if i+window-1 >= length:
                        continue

                    word = sent[i:i+window]
                    if ' ' in word:
                        continue

                    word_index = self.encoder.fit(word)

                    if sent[i-1] == ' ':
                        left_extension = sent[max(0,i-2):i+window]
                    else:
                        left_extension = sent[i-1:i+window]

                    if sent[i+window] == ' ':
                        right_extension = sent[i:min(length,i+window+2)]
                    else:
                        right_extension = sent[i:i+window+1]                            

                    if left_extension == None or right_extension == None:
                        print(sent, i, window)

                    left_index = self.encoder.fit(left_extension)
                    right_index = self.encoder.fit(right_extension)
                    
                    self.L[word_index][left_index] += 1
                    self.R[word_index][right_index] += 1

            if (num_for_pruning > 0) and ( (num_sent + 1) % num_for_pruning == 0):
                before, after = self.prune_extreme_case(min_count)
                sys.stdout.write('\rnum sent = %d: %s --> %s' % (num_sent, str(before), str(after)))

        if (num_for_pruning > 0) and ( (num_sent + 1) % num_for_pruning == 0):
            self.prune_extreme_case(min_count)
            sys.stdout.write('\rnum_sent = %d: %s --> %s' % (num_sent, str(before), str(after)))


                    
    def load(self, model_fname, encoder_fname):

        self.encoder.load(encoder_fname)
        
        try:
            with open(model_fname, encoding='utf-8') as f:
                
                next(f) # SKIP: parameters (min_length, max_length)
                token = next(f).split()
                self.min_length = int(token[0])
                self.max_length = int(token[1])
                
                next(f) # SKIP: left side extension
                is_right_side = True
                
                for line in f:
                    
                    if '# right side extension' in line:
                        is_right_side = True
                        continue
                        
                    token = line.split();
                    word = int(token[0])
                    extension = int(token[1])
                    freq = int(token[2])
                    
                    if is_right_side:
                        self.R[word][extension] = freq
                    else:
                        self.L[word][extension] = freq
                
        except Exception as e:
            print(e)
        
        
    def save(self, model_fname, encoder_fname):
        
        self.encoder.save(encoder_fname)
        
        try:
            with open(model_fname, 'w', encoding='utf-8') as f:
                
                f.write("# parameters (min_length max_length)\n")
                f.write('%d %d\n' % (self.min_length, self.max_length))
                
                f.write('# left side extension\n')
                for word, extension_dict in self.L.items():
                    for extension, freq in extension_dict.items():
                        f.write('%d %d %d\n' % (word, extension, freq))
                
                f.write('# right side extension\n')
                for word, extension_dict in self.R.items():
                    for extension, freq in extension_dict.items():
                        f.write('%d %d %d\n' % (word, extension, freq))
                        
        except Exception as e:
            print(e)
            

    def words(self):
        return set(self.encoder.inverse)
