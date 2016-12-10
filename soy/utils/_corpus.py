class DoublespaceLineCorpus:
    
    def __init__(self, corpus_fname, num_doc = -1, num_sent = -1, iter_sent = False, skip_header = 0):
        self.corpus_fname = corpus_fname
        self.num_doc = num_doc
        self.num_sent = num_sent
        self.iter_sent = iter_sent
        self.skip_header = skip_header

        is_part = False
        num_sent_tmp = 0
        
        with open(corpus_fname, encoding='utf-8') as f:
        
            for doc_id, doc in enumerate(f):

                if is_part:
                    break

                if (num_doc > 0) and (doc_id + 1 == num_doc):
                    is_part = True
                    break

                for sent in doc.split('  '):
                    if not sent.strip():
                        continue
                    num_sent_tmp += 1

                    if (num_sent > 0) and (num_sent_tmp == num_sent):
                        is_part = True
                        break
                
            self.num_doc = doc_id + 1 if num_doc < 0 else min(num_doc, (doc_id + 1))
            self.num_sent = num_sent_tmp if num_sent < 0 else min(num_sent_tmp, num_sent)
        
        print('DoublespaceLineCorpus %s has %d docs, %d sents' % ('(partial)' if is_part else '', self.num_doc, self.num_sent))
                

    def __iter__(self):
        
        with open(self.corpus_fname, encoding='utf-8') as f:
            
            num_sent = 0
            stop_iter = False
 
            for _num_doc, doc in enumerate(f):
                
                if stop_iter:
                    break

                if _num_doc >= self.num_doc:
                    break
                    
                if _num_doc < self.skip_header:
                    continue
                    
                if not self.iter_sent:
                    yield doc
                    
                else:
                    for sent in doc.split('  '):

                        num_sent += 1
                        if num_sent > self.num_sent:
                            stop_iter = True
                            break
                        sent = sent.strip()
                        if not sent:
                            continue
                        yield sent.strip()
    
    
    def __len__(self):
        return self.num_sent if self.iter_sent else self.num_doc
