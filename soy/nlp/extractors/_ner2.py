from collections import defaultdict
import pickle
import sys
import time
import psutil
import numpy as np

def remain_time(begin_time, i, n):
    process_time = (time.time() - begin_time)
    required_time = (process_time / (i+1)) * (n - i)
    if required_time > 3600:
        return '%.3f hours' % (required_time / 3600)
    if required_time > 60:
        return '%.3f mins' % (required_time / 60)
    return '%.3f secs' % required_time

def get_available_memory():
    mem = psutil.virtual_memory()
    return 100 * mem.available / (mem.total)

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

class FeatureManager:
    
    def __init__(self, templates=None, feature_begin=-2, feature_end=2):
        self.begin = feature_begin
        self.end = feature_end
        self.templates = templates if templates else self._generate_token_templates()
        self.vocab_to_idx = {}
        self.idx_to_vocab = []
        self.counter = {}

    def _generate_token_templates(self):
        templates = []
        for b in range(self.begin, self.end):
            for e in range(b, self.end+1):
                if (b == 0) or (e == 0):
                    continue
                templates.append((b, e))
        return templates

    def words_to_feature(self, words):
        x =[]
        for i in range(len(words)):
            xi = []
            e_max = len(words)
            for t in self.templates:
                b = i + t[0]
                e = i + t[1] + 1
                if b < 0 or e > e_max:
                    continue
                if b == (e - 1):
                    xi.append(('X[%d]' % t[0], words[b]))
                else:
                    contexts = words[b:e] if t[0] * t[1] > 0 else words[b:i] + words[i+1:e]        
                    xi.append(('X[%d,%d]' % (t[0], t[1]), tuple(contexts)))
            x.append(xi)
        return x

    def words_to_encoded_feature(self, words):
        x = self.words_to_feature(words)
        z = [[self.vocab_to_idx[f] for f in xi if f in self.vocab_to_idx] for xi in x]
        return z

    def scanning_features(self, corpus_fname, pruning_min_count=5, min_count=50):
        counter = defaultdict(lambda: 0)
        with open(corpus_fname, encoding='utf-8') as f:
            for num_sent, sent in enumerate(f):
                words = sent.strip().split()
                if not words:
                    continue
                    
                x = self.words_to_feature(words)
                for word, xi in zip(sent, x):
                    for feature in xi:
                        counter[feature] += 1

                if (num_sent + 1) % 1000000 == 0:
                    before_size = len(counter)
                    before_memory = get_process_memory()
                    counter = defaultdict(lambda: 0, {f:v for f,v in counter.items() if v >= pruning_min_count})
                    sys.stdout.write('\r# features = %d -> %d, (%d in %d) %.3f -> %.3f Gb' % (
                            before_size, len(counter), num_sent, len(corpus), 
                            before_memory, get_process_memory()))

                if num_sent % 1000 == 0:
                    sys.stdout.write('\r# features = %d, (%d in %d) %.3f Gb' % (len(counter), num_sent, len(corpus), get_process_memory()))

        counter = {f:v for f,v in counter.items() if v >= min_count}
        self.idx_to_vocab = list(sorted(counter.keys(), key=lambda x:counter.get(x, 0), reverse=True))
        self.vocab_to_idx = {vocab:idx for idx, vocab in enumerate(self.idx_to_vocab)}
        self.counter = counter
    
    def transform_rawtext_to_zorpus(self, rawtext_fname, zcorpus_fname):
        with open(rawtext_fname, encoding='utf-8') as fi:
            with open(zcorpus_fname, 'w', encoding='utf-8') as fo:
                for num_sent, sent in enumerate(fi):
                    words = sent.strip().split()
                    if not words:
                        fo.write('\n')
                        continue
                    z = self.words_to_encoded_feature(words)
                    for wi, zi in zip(words, z):
                        features = ' '.join([str(zi_) for zi_ in zi]) if zi else ''
                        fo.write('%s\t%s\n' % (wi, features))
                    if num_sent % 50000 == 0:
                        sys.stdout.write('\rtransforming .... (%d sents) %.3f Gb' % (
                                num_sent, get_process_memory()))
                print('\rtransforming has done')
            
    def save(self, fname):        
        with open(fname, 'wb') as f:
            parameters = {
                'feature_begin': self.begin,
                'feature_end': self.end,
                'templates': self.templates,
                'idx_to_vocab': self.idx_to_vocab,
                'vocab_to_idx': self.vocab_to_idx, 
                'counter': self.counter
            }
            pickle.dump(parameters, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            parameters = pickle.load(f)
        self.begin = parameters['feature_begin']
        self.end = parameters['feature_end']
        self.templates = parameters['templates']
        self.idx_to_vocab = parameters['idx_to_vocab']
        self.vocab_to_idx = parameters['vocab_to_idx']
        self.counter = parameters['counter']
    
    
class ZCorpus:
    def __init__(self, fname):
        self.fname = fname
        self.length = 0
        
    def __len__(self):
        if self.length == 0:
            with open(self.fname, encoding='utf-8') as f:
                for num_row, _ in enumerate(f):
                    continue
                self.length = (num_row + 1)
        return self.length
    
    def __iter__(self):
        with open(self.fname, encoding='utf-8') as f:
            for row in f:
                row = row.strip()
                if ('\t' in row) == False: continue
                word, features = row.split('\t')
                features = features.split()
                yield word, features
                

class FeatureCountingNER:
    
    def __init__(self, feature_manager=None):
        self.feature_manager = feature_manager
        self.coefficient = {}
        self.coefficient_ = {}
        self._usg_pos = None
        self._usg_neg = None
        self._usg_features = None
        
    def find_positive_features(self, corpus, ner_seeds, wordset, min_count_positive_features=10):
        self.positive_features = {}
        
        # TODO: file 넣는걸로 바꾸기
        for num_z, (word, features) in enumerate(zcorpus):    
            if num_z % 1000 == 0:
                sys.stdout.write('\r# scanning positive features # = %d, (%.3f %s, %d in %d) %.3f Gb' 
                                 % (len(self.positive_features), (100 * num_z / len(zcorpus)), '%', 
                                    num_z, len(zcorpus), get_process_memory()))

            if (word in ner_seeds) == False:
                continue        
            for feature in features:
                self.positive_features[feature] = self.positive_features.get(feature, 0) + 1

        self.positive_features = {pos_f:v for pos_f, v in self.positive_features.items() if v > min_count_positive_features}
        print('\rscanning positive features was done')
        return self.positive_features
    
    def compute_score_of_features(self, zcorpus):
        def proportion_has_seeds(word_dict):
            n_neg = 0
            n_pos = 0
            for word, freq in word_dict.items():
                if word in ner_seeds:
                    n_pos += freq
                else:
                    n_neg += freq
            return n_pos / (n_pos + n_neg)
        # TODO: corpus -> filename 넣는걸로
        usg_features, usg_pos, usg_neg = self._scan_usage_of_features(zcorpus)
        
        begin_time = time.time()
        n = len(usg_pos)

        proportion_positive_features = {}
        for i, (feature, n_positive) in enumerate(usg_pos.items()):
            # Other functions
        #     proportion_positive_features[feature] = proportion_over_word2vec_sim(usage_of_features[feature], min_similarity=0.5)
            proportion_positive_features[feature] = proportion_has_seeds(usg_features[feature])
            sys.stdout.write('\r(%d in %d) remained %s' % (i+1, n, remain_time(begin_time, i, n)))
        print('\rcomputing score of features was done')
        self.coefficient = proportion_positive_features
        self.coefficient_ = {self.feature_manager.idx_to_vocab[int(f)]:s for f,s in proportion_positive_features.items() }
        self._usg_pos = dict(usg_pos)
        self._usg_neg = dict(usg_neg)
        self._usg_features = {feature:{w:f for w,f in wd.items()} for feature, wd in usg_features.items()}
    
    def _scan_usage_of_features(self, zcorpus):
        usage_of_features = defaultdict(lambda: defaultdict(lambda: 0))
        usage_of_positive_features = defaultdict(lambda: 0)

        for num_z, (word, features) in enumerate(zcorpus):
            if num_z % 1000 == 0:
                sys.stdout.write('\r# scanning usage of positive features (%.3f %s, %d in %d) %.3f Gb' 
                                 % ((100 * num_z / len(zcorpus)), '%', 
                                    num_z, len(zcorpus), get_process_memory()))

            if (word in word2vec_wordset) == False:
                continue
            for feature in features:
                if feature in self.positive_features:
                    usage_of_features[feature][word] += 1
                    if word in ner_seeds:
                        usage_of_positive_features[feature] += 1

        usage_of_negative_features = {feature:(sum(v.values()) - usage_of_positive_features.get(feature, 0)) 
                              for feature, v in usage_of_features.items()}
                
        print('\rscanning usage of positive features was done')
        return usage_of_features, usage_of_positive_features, usage_of_negative_features
        
    def get_coefficient_histogram(self, n_bins=20):        
        heights, centroids = np.histogram(self.coefficient.values(), bins=n_bins)
        for h, c1, c2 in zip(heights, centroids, centroids[1:]):
            print('%.2f ~ %.2f: %.3f' % (c1, c2, h/sum(heights)))
            
    def extract_named_entities_from_zcorpus(self, zcorpus):
        prediction_score = defaultdict(lambda: 0.0)
        prediction_count = defaultdict(lambda: 0.0)

        for num_z, (word, features) in enumerate(zcorpus):
            if num_z % 1000 == 0:
                sys.stdout.write('\r(%d in %d) %.3f %s' % (num_z, len(zcorpus), 100 * num_z / len(zcorpus), '%'))
            if not features: continue
            for feature in features:
                if (feature in self.coefficient) == False:
                    continue
                prediction_score[word] += self.coefficient[feature]
                prediction_count[word] += 1

        prediction_normed_score = {word:score/prediction_count[word] for word, score in prediction_score.items()}
        sorted_scores = sorted(prediction_normed_score.items(), key=lambda x:x[1], reverse=True)
        return sorted_scores
    
    def infer_named_entity_score(self, encoded_features):
        score = 0
        norm = 0
        for f in encoded_features:
            if (f in self.coefficient) == False:
                continue
            score += self.coefficient[f]
            norm += 1
        return (score / norm) if norm > 0 else 0
    
    def save(self, fname):
        with open(fname, 'wb') as f:
            parameters = {
                'usage_of_positive_features': self._usg_pos,
                'usage_of_negative_features': self._usg_neg,
                'usage_of_features': self._usg_features,
                'coefficient': self.coefficient,
                'coefficient_': self.coefficient_,
                
                'feature_manager.feature_begin': self.feature_manager.begin,
                'feature_manager.feature_end': self.feature_manager.end,
                'feature_manager.templates': self.feature_manager.templates,
                'feature_manager.idx_to_vocab': self.feature_manager.idx_to_vocab,
                'feature_manager.vocab_to_idx': self.feature_manager.vocab_to_idx, 
                'feature_manager.counter': self.feature_manager.counter
            }
            pickle.dump(parameters, f)
        
    def load(self, fname):
        with open(fname, 'rb') as f:
            parameters = pickle.load(f)
            self._usg_pos = parameters['usage_of_positive_features']
            self._usg_neg = parameters['usage_of_negative_features']
            self._usg_features = parameters['usage_of_features']
            self.coefficient = parameters['coefficient']
            self.coefficient_ = parameters['coefficient_']
            
            self.feature_manager = FeatureManager()
            self.feature_manager.begin = parameters['feature_manager.feature_begin']
            self.feature_manager.end = parameters['feature_manager.feature_end']
            self.feature_manager.templates = parameters['feature_manager.templates']
            self.feature_manager.idx_to_vocab = parameters['feature_manager.idx_to_vocab']
            self.feature_manager.vocab_to_idx = parameters['feature_manager.vocab_to_idx']
            self.feature_manager.counter = parameters['feature_manager.counter']
