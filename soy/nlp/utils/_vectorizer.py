from collections import Counter
import sys
from scipy.sparse import csr_matrix

class Vectorizer:
    
    def __init__(self, vocabs, weights=None, default_weight):
        self._default_weight = default_weight
        if type(vocabs) == str:
            self.load(vocabs)
        else:
            self._vocab2int = {vocab:idx for idx, vocab in enumerate(vocabs)}
            self._int2vocab = vocabs
            self._weights = weights
            self._num_vocab = len(vocabs)
    
    def __len__(self):
        return self._num_vocab
    
    def _encode(self, tokens):
        bow = Counter(tokens)
        bow = {self._vocab2int[v]:w for v,w in bow.items() if v in self._vocab2int}
        if self._weights != None:
            weighted_bow = {v:(w * self._weights.get(v, self._default_weight)) for v,w in bow.items()}
            return weighted_bow
        return bow
    
    def encode_to_dict(self, tokens, normalize=False, norm='l2'):
        bow = self._encode(tokens)
        if normalize:
            norm = np.sum(bow.values()) if norm == 'l1' else np.sqrt(sum([v**2 for v in bow.values()]))
            normed_bow = {v:w/norm for v,w in bow.items()}
            return normed_bow
        return bow
    
    def encode_to_sparse_vector(self, tokens, normalize=False, norm='l2'):
        bow = self._encode(tokens)
        row = []
        col = []
        data = []
        for t,w in sorted(bow.items()):
            row.append(0)
            col.append(t)
            data.append(w)
        if normalize:
            norm = np.sum(bow.values()) if norm == 'l1' else np.sqrt(sum([v**2 for v in bow.values()]))
            normed_data = [w/norm for w in data]
            return csr_matrix((normed_data, (row, col)), shape=(1, self._num_vocab))
        return csr_matrix((data, (row, col)), shape=(1, self._num_vocab))
    
    def _decode(self, bow):
        decoded_bow = [(self._int2vocab[v], w) for v,w in bow if (0 <= v < self._num_vocab) ]
        return decoded_bow
    
    def decode_from_dict(self, int_dict):
        bow = self._decode(int_dict.items())
        return bow
    
    def decode_from_sparse_vector(self, csr_vector):
        idxs = csr_vector.nonzero()[1]
        weight = csr_vector.data
        bow = self._decode(zip(idxs, weight))
        return bow
    
    def transform(self, docs, verbose=0):
        n = len(docs)
        row = []
        col = []
        data = []
        for row_id, tokens in enumerate(docs):
            for col_id, weight in self._encode(tokens).items():
                row.append(row_id)
                col.append(col_id)
                data.append(weight)
            if (verbose > 0) and (row_id % verbose == 0):
                sys.stdout.write('\rtransforming ... (%d in %d)' % (row_id + 1, len(docs)))
        if (verbose > 0):
            print('\rtransforming was done. shape = (%d, %d)' % (n, self._num_vocab))
        return csr_matrix((data, (row, col)), shape=(n, self._num_vocab))
        
    def save(self, fname, delimiter='\t'):
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                if self._weights == None:
                    self._weights = {}
                for idx, vocab in enumerate(self._int2vocab):
                    if idx in self._weights:
                        f.write('%s%s%f\n' % (vocab, delimiter, self._weights[idx]))
                    else:
                        f.write('%s\n' % vocab)
                if len(self._weights) == 0:
                    self._weights = None
        except Exception as e:
            print(e)
        
    def load(self, fname, delimiter='\t'):
        try:
            with open(fname, encoding='utf-8') as f:
                self._vocab2int = {}
                self._int2vocab = []
                self._weights = {}
                for idx, row in enumerate(f):
                    col = row.replace('\n','').split(delimiter)
                    if len(col) == 1:
                        self._vocab2int[col[0]] = len(self._vocab2int)
                    elif len(col) == 2:
                        self._vocab2int[col[0]] = len(self._vocab2int)
                        self._weights[len(self._vocab2int) - 1] = float(col[1])
                    else:
                        raise ValueError('Vocabulary Indexer can have form <str> or <str, weight>')
                    self._int2vocab.append(col[0])
                if len(self._weights) == 0:
                    self._weights = None
                self._num_vocab = len(self._vocab2int)
                
        except Exception as e:
            print(e)
        