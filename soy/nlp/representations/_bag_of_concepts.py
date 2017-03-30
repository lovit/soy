from collections import defaultdict
import sys
import numpy as np


class ConceptMapperBuilder:
    
    def __init__(self, vocabulary_list, vocabulary_count, max_covered_terms_for_a_concept=20, max_concept_for_a_term=1, beta=3.5):
        self.vocabulary = vocabulary_list
        self.vocab_count = vocabulary_count
        self.vocab2int = None if not vocabulary_list else {w:i for i,w in enumerate(vocabulary_list)}
        
        self.max_covered_terms_for_a_concept = max_covered_terms_for_a_concept
        self.max_concept_for_a_term = max_concept_for_a_term
        self.beta = beta
        
    def encode(self, vocab):
        return self.vocab2int.get(vocab, -1)
    
    def count(self, vocab):
        return self.vocab_count.get(vocab, 0)
    
    def decode(self, vocab_index):
        if (0 <= vocab_index < len(self.vocabulary)):
            return self.vocabulary[vocab_index]
        return None

    def _progress(self, i, n, head=None):
        if head == None: head = ''
        sys.stdout.write('\r%s %.3f %s (%d in %d)' % (head, 100 * (i+1) / n, '%', i+1, n) )
        
    def _to_dictdict(self, dict_list):
        return {k1:{k2:v for k2,v in l} for k1, l in dict_list.items()}
    
    def _encode_dictdict(self, dd):
        encoded_dd = {self.encode(k1):{self.encode(k2):s for k2, s in d.items()} for k1, d in dd.items()}
        encoded_dd = {k1:{k2:s for k2, s in d.items() if k2 != -1}  for k1, d in encoded_dd.items() if k1 != -1}
        encoded_dd = {k1:d for k1, d in encoded_dd.items() if d}
        return encoded_dd
    
    def build_mapper(self, knn, encode_as_index=False, ensure_proper_knn=False):
        knn = self._check_knn_type(knn)
        print('\rchecked knn graph was done')
        
        if not ensure_proper_knn:
            knn = self._check_words(knn)
            print('\rchecked words in knn graph')
        
        rknn = self.reverse_knn(knn)        
        print('\rbuilding reverse knn graph was done')
        
        mapper, anchor_to_words = self._build_initial_mapper(rknn)
        print('initial mapper was built')

        if self.max_concept_for_a_term > 1:
            mapper, anchor_to_words = self._expand_representative_words(knn, mapper, anchor_to_words)
            print('mapper was expanded for multi concepts')
        
        mapper = self._to_dictdict(mapper)
        anchor_to_words = self._to_dictdict(anchor_to_words)
        
        if encode_as_index:
            mapper = self._encode_dictdict(mapper)
            anchor_to_words = self._encode_dictdict(anchor_to_words)
            print('words in mapper were encoded as word index')
        
        return mapper, anchor_to_words
    
    def _check_knn_type(self, knn):
        knn_ = {}
        for num, (from_word, neighbors) in enumerate(knn.items()):
            if num % 2000 == 0:
                self._progress(num, len(knn), 'checking knn graph type')
            if not neighbors:
                continue
            if type(neighbors) == dict:
                neighbors = list(neighbors.items())            
            neighbors = sorted(neighbors, key=lambda x:x[1], reverse=True)
            knn_[from_word] = neighbors
        return knn_
    
    def _check_words(self, knn):
        def is_proper_word(word):
            if not self.vocab_count:
                return True
            return word in self.vocab_count
        
        knn_ = {}
        for num, (from_word, neighbors) in enumerate(knn.items()):
            if not is_proper_word(from_word):
                continue
            if num % 2000 == 0:
                self._progress(num, len(knn), 'checking words in knn graph')
            neighbors = [neighbor for neighbor in neighbors if is_proper_word(neighbor[0])]
            if not neighbors:
                continue
            knn_[from_word] = neighbors
        return knn_
    
    def reverse_knn(self, knn):
        '''Build reverse k nearest neighbor graph
        
        Parameters:
        knn: dict of list
            knn[word] = [(word, score), (word, score), ... ]
            the list is sorted by score with reverse order
        '''
        rknn = defaultdict(lambda: [])
        for num, (from_word, neighbors) in enumerate(knn.items()):
            for to_word, sim in neighbors:
                rknn[to_word].append((from_word, sim))
            if num % 2000 == 0:
                self._progress(num, len(knn), 'building reverse k-NN')
        rknn = dict(rknn)

        for to_word, from_words in rknn.items():
            from_words = sorted(from_words, key=lambda x:x[1], reverse=True)            
            rknn[to_word] = from_words

        return rknn
    
    def _build_initial_mapper(self, rknn):
        mapper = {}
        anchor_to_words = {}

        sorted_rknn = sorted(rknn.items(), key=lambda x:self.count(x[0]), reverse=True)

        for anchor_word, from_words in sorted_rknn:
            if (anchor_word in mapper) or (len(from_words) < 1):
                continue
            
            covered_words = [(anchor_word, 1.0)]
            
            for from_word, sim in from_words:
                if len(covered_words) >= self.max_covered_terms_for_a_concept:
                    break
                    
                if (from_word in mapper):
                    continue
                    
                covered_words.append((from_word, sim))
                
            if len(covered_words) <= 1:
                continue
            
            for from_word, sim in covered_words:
                mapper[from_word] = [(anchor_word, sim)]
                anchor_to_words[anchor_word] = covered_words
        
        return mapper, anchor_to_words
    
    def _expand_representative_words(self, knn, mapper, anchor_to_words):
        mapper_ = {}
        
        for word, anchor_words in mapper.items():
            anchor_word = anchor_words[0][0]
            appended_anchors = []
            
            if (word in knn) == False:
                continue
            
            for knn_word, sim in knn[word]:
                if len(appended_anchors) >= (self.max_concept_for_a_term - 1):
                    break

                if (knn_word in anchor_to_words) and (knn_word != word) and (knn_word != anchor_word):
                    appended_anchors.append((knn_word, sim))
                    anchor_to_words[knn_word].append((word, sim))

            mapper_[word] = (anchor_words + appended_anchors)

        for word, anchors_words in mapper_.items():
            anchors_words = self._normalize(anchors_words)
            mapper_[word] = anchors_words

        return mapper_, anchor_to_words

    def _normalize(self, anchors_words):
        normalized = [(word, np.exp(w) ** self.beta) for word, w in anchors_words]
        sum_ = sum([w for _, w in normalized])
        normalized = [(word, w/sum_) for word, w in normalized]
        return normalized

