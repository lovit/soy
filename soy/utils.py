import collections

class IntegerEncoder:
    
    def __init__(self):
        
        self.mapper = {}
        self.inverse = []
        self.num_object = 0

    def compatify(self):

        fixer = {}
        pull_index = 0
        none_index = []
        
        for i, x in enumerate(self.inverse):
            if x == None:
                none_index.append(i)
                pull_index += 1
            elif pull_index > 0:
                fixed = i - pull_index
                fixer[i] = fixed
                self.mapper[x] = fixed
                
        for i in reversed(none_index):
            del self.inverse[i]
        
        return fixer
    
        
    def decode(self, i, unknown = None):
        if i >= 0 and i < self.num_object:
            return self.inverse[i]
        else:
            return unknown

        
    def encode(self, x, unknown = -1):
        if x in self.mapper:
            return self.mapper[x]
        else:
            return unknown
        
        
    def fit(self, x):
        if x in self.mapper:
            return self.mapper[x]
        else:
            self.mapper[x] = self.num_object
            self.num_object += 1
            self.inverse.append(x)
            return (self.num_object - 1)
        
        
    def keys(self):
        return self.inverse
        
        
    def remove(self, x):
        if x in self.mapper:
            i = self.mapper[x]
            del self.mapper[x]
            self.inverse[i] = None
            self.num_object -= 1
        
        
    def save(self, fname, to_str=lambda x:str(x)):
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                for x in self.inverse:
                    f.write('%s\n' % to_str(x))
        except Exception as e:
            print(e)
        
        
    def load(self, fname, parse=lambda x:x.replace('\n','')):
        try:
            with open(fname, encoding='utf-8') as f:
                for line in f:
                    x = parse(line)
                    self.inverse.append(x)
                    self.mapper[x] = self.num_object
                    self.num_object += 1
        except Exception as e:
            print(e)
            print('line number = %d' % self.num_object)

            
    def __len__(self):
        return self.num_object
        
        
        
class Corpus:
    
    def __init__(self, corpus_fname, num_doc = -1, num_sent = -1):
        self.corpus_fname = corpus_fname
        self.num_doc = num_doc
        self.num_sent = num_sent

        if num_doc <= 0:
            with open(corpus_fname, encoding='utf-8') as f:
                num_sent_tmp = 0
                for doc_id, doc in enumerate(f):
                    self.num_doc = doc_id
                    for sent in doc.split('  '):
                        if not sent.strip():
                            continue
                        num_sent_tmp += 1
                self.num_doc += 1
                if num_sent <= 0:
                    self.num_sent = num_sent_tmp
        
        
    def __iter__(self):
        with open(self.corpus_fname, encoding='utf-8') as f:
            for doc_id, doc in enumerate(f):
                if doc_id >= self.num_doc:
                    break
                yield doc.strip()

                
    def iter_sents(self):
        with open(self.corpus_fname, encoding='utf-8') as f:
            for doc_id, doc in enumerate(f):
                if doc_id >= self.num_doc:
                    break
                for sent in doc.split('  '):
                    sent = sent.strip()
                    if not sent:
                        continue
                    yield sent.strip()
    
    
    def __len__(self):
        return self.num_sent        
