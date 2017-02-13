from soy.utils import DoublespaceLineCorpus
from gensim.models.doc2vec import TaggedDocument

class DoublespaceLineDoc2VecCorpus(DoublespaceLineCorpus):
    
    def __init__(self, corpus_fname, num_doc = -1, num_sent = -1, iter_sent=False, skip_header = 0, label_delimiter=None):
        super().__init__(corpus_fname, num_doc, num_sent, iter_sent, skip_header)
        self.label_delimiter = label_delimiter
        if (label_delimiter != None) and (type(label_delimiter) != str):
            self.label_delimiter = None
            print('label delimiter type should be str, but %s' % type(label_delimiter))

    def __iter__(self):
        with open(self.corpus_fname, encoding='utf-8') as f:
            num_sent = 0
            stop_iter = False
            for _num_doc, doc in enumerate(f):
                
                if stop_iter: break
                if _num_doc >= self.num_doc: break
                if _num_doc < self.skip_header: continue
                
                label = str(_num_doc)
                
                if self.label_delimiter != None:
                    label, doc = doc.split(self.label_delimiter)
                
                if not self.iter_sent:
                    words = doc.split()
                    words = [word for word in words if word]
                    if not words:
                        continue
                    yield TaggedDocument(words=words, tags=['DOC_%s' % label])
                else:
                    for sent in doc.split('  '):
                        words = sent.split()
                        words = [word for word in words if word]
                        if not words:
                            continue
                        tags = ['DOC_%s' % label] if self.label_delimiter != None else ['SENT_%d' % num_sent]
                        yield TaggedDocument(words=words, tags=tags)
                        num_sent += 1


class DoublespaceLineWord2VecCorpus(DoublespaceLineCorpus):
    
    def __init__(self, corpus_fname, num_doc=-1, num_sent=-1, iter_sent=False, skip_header=0):
        super().__init__(corpus_fname, num_doc, num_sent, iter_sent, skip_header)

    def __iter__(self):
        with open(self.corpus_fname, encoding='utf-8') as f:
            num_sent = 0
            stop_iter = False
            for _num_doc, doc in enumerate(f):
                
                if stop_iter: break
                if _num_doc >= self.num_doc: break
                if _num_doc < self.skip_header: continue
                
                if self.iter_sent == False:
                    words = doc.strip().split()
                    words = [word for word in words if word]
                    if not words:
                        continue
                    yield words
                else:
                    for sent in doc.split('  '):
                        words = sent.split()
                        words = [word for word in words if word]
                        if not words:
                            continue
                        yield words
                        num_sent += 1                        