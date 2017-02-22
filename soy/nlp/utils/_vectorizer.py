from collections import Counter
import sys
from scipy.sparse import csr_matrix

class Vectorizer:
    
    def __init__(self, vocabs=None):
        if vocabs == None:
            vocabs = []
        self._vocab2int = {vocab:idx for idx, vocab in enumerate(vocabs)}
        self._int2vocab = vocabs
        self._num_vocab = len(vocabs)
    
    def __len__(self):
        return self._num_vocab
    
    def _encode(self, tokens):
        bow = Counter(tokens)
        bow = {self._vocab2int[v]:w for v,w in bow.items() if v in self._vocab2int}
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
        
    def save(self, fname):
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                for vocab in self._int2vocab:
                    f.write('%s\n' % vocab)
        except Exception as e:
            print(e)
        
    def load(self, fname):
        try:
            with open(fname, encoding='utf-8') as f:
                self._vocab2int = {}
                self._int2vocab = []
                for idx, row in enumerate(f):
                    vocab = row.replace('\n','')
                    if not vocab:
                        raise ValueError('Vocabulary must not be empty str')
                    self._int2vocab.append(vocab)
                    self._vocab2int[vocab] = idx
                self._num_vocab = len(self._vocab2int)
        except Exception as e:
            print(e)
        