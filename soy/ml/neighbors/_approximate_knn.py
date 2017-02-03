from collections import Counter, defaultdict
import os
import pickle
from pprint import pprint
import time
import numpy as np


class FastCosine():
    
    def __init__(self):
        self.inverted = defaultdict(lambda: [])
        self.idf = {}
        self.num_doc = 0
        self.num_term = 0

        self._base_time = None

    def _get_process_time(self):
        if self._base_time == None:
            self._base_time = time.time()
        process_time = 1000 * (time.time() - self._base_time)
        self._base_time = time.time()
        return process_time
    
    def indexing(self, mm_file):
        t2d, norm_d = self._load_mm(mm_file)
        print('loaded mm')
        
        t2d = self._normalize_weight(t2d, norm_d)
        print('normalized t2d weight')
        
        self._build_champion_list(t2d)
        print('champion list')
        
        self._build_idf(t2d)
        print('idf')
        
        del t2d
    
    def _load_mm(self, mm_file):
        t2d = defaultdict(lambda: {})
        norm_d = defaultdict(lambda: 0)
        
        if not os.path.exists(mm_file):
            raise IOError('mm file not found: %s' % mm_file)
        
        with open(mm_file, encoding='utf-8') as f:
            # Skip three head lines
            for _ in range(3):
                next(f)
            
            # line format: doc term freq
            try:
                for line in f:
                    doc, term, freq = line.split()

                    doc = int(doc) - 1
                    term = int(term) - 1
                    freq = float(freq)
                    
                    t2d[term][doc] = freq
                    norm_d[doc] += freq ** 2
                
                    self.num_doc = max(self.num_doc, doc)
                    self.num_term = max(self.num_term, term)
                
                self.num_doc += 1
                self.num_term += 1
            except Exception as e:
                print('mm file parsing error %s' % line)
                print(e)
        
        for d, v in norm_d.items():
            norm_d[d] = np.sqrt(v)
        
        return t2d, norm_d
    
    def _normalize_weight(self, t2d, norm_d):
        def div(d_dict, norm_d):
            norm_dict = {}
            for d, w in d_dict.items():
                norm_dict[d] = w / norm_d[d]
            return norm_dict
                
        for t, d_dict in t2d.items():
            t2d[t] = div(d_dict, norm_d)
            
        return t2d
        
    def _build_champion_list(self, t2d):
        def pack(wd):
            '''
            chunk: [(w1, (d11, d12, ...)), (w2, (d21, d22, d23, ...)), ... ] 
            return (
                    (w1, w2, w3, w4),
                    (len(d1), len(d2), len(d3), len(d4)),
                    ({d11, d12}, 
                     {d21, d22, d23},
                     {d31, d32},
                     {d41, d42, d43, d44}
                    )
                ) 
            '''
            w_array, d_array = zip(*wd)
            len_array = tuple([len(d_list) for d_list in d_array])
            d_array = [set(d_list) for d_list in d_array] # set 처리가 바뀜
            return (w_array, len_array, d_array) 
            
        for t, d_dict in t2d.items():
            wd = defaultdict(lambda: [])
            
            for d, w in d_dict.items():
                wd[w].append(d)
                
            wd = sorted(wd.items(), key=lambda x:x[0], reverse=True)
            self.inverted[t] = pack(wd)
        
    def _build_idf(self, t2d):
        for t, d_dict in t2d.items():
            self.idf[t] = np.log(self.num_doc / len(d_dict))
        
    def rneighbors(self, query, query_range=0.2, candidate_factor=3.0, earlystop_cut=1.0, w_cut=0.5, score_as_add=True, compute_true_cosine=False):
        # TODO
        raise NotImplementedError

    def kneighbors(self, query, n_neighbors=10, candidate_factor=3.0, earlystop_cut=1.0, w_cut=0.5, score_as_add=True, compute_true_cosine=False):
        '''query: {term:weight, ..., }
        
        '''
        
        times = {}
        self._get_process_time()
        
        query = self._check_query(query)
        if not query:
            return {}, {}
        times['check_query_type'] = self._get_process_time()
        
        query = self._order_search_term(query)
        times['order_search_term'] = self._get_process_time()
        
        n_candidates = int(n_neighbors * candidate_factor)
        scores, info = self._retrieve_similars(query, n_candidates, earlystop_cut, w_cut, score_as_add)
        scores = scores[:n_neighbors]
        times['retrieval_similars'] = self._get_process_time()
        
        if compute_true_cosine:
            neighbors_idx, _ = zip(*scores)
            scores = self._exact_computation(query, neighbors_idx)
        times['true_cosine_computation'] = self._get_process_time()
        times['whole_querying_process'] = sum(times.values())
        info['time [mil.sec]'] = times
        return scores, info
    
    def _check_query(self, query):
        sum_ = sum(v ** 2 for v in query.values())
        sum_ = np.sqrt(sum_)
        return {t:w/sum_ for t,w in query.items() if t in self.idf}

    def _order_search_term(self, query):
        query = [(qt, qw, qw * self.idf[qt]) for qt, qw in query.items()]
        query = sorted(query, key=lambda x:x[2], reverse=True)
        return query
    
    def _retrieve_similars(self, query, n_candidates, earlystop_cut=0.5, w_cut=0.2, score_as_add=False):

        def select_champs(champ_list, n_candidates, w_cut=0.5):
            w_cut_threshold = (champ_list[0][0] * w_cut)
            sum_num = 0
            for i, (w, num, docs) in enumerate(zip(*champ_list)):
                if w < w_cut_threshold:
                    break
                sum_num += num
                if sum_num >= n_candidates:
                    break
            return [champ_list[0][:i], champ_list[1][:i], champ_list[2][:i]]

        scores = {}
        remain_proportion = 1
        
        n_computation = 0
        n_considered_terms = 0

        for qt, qw, tfidf in query:

            n_considered_terms += 1
            
            champ_list = self._get_champion_list(qt)
            if champ_list == None:
                continue

            champ_list = select_champs(champ_list, n_candidates, w_cut)

            for w, num, docs in zip(*champ_list):
                for d in docs:
                    scores[d] = scores.get(d, 0) + (w if score_as_add else qw * w)
                n_computation += num

            if (remain_proportion * tfidf) < earlystop_cut:
                break

        info = {
            'n_computation': n_computation, 
            'n_considered_terms': n_considered_terms, 
            'n_terms_in_query': len(query),
            'n_candidate': len(scores),
            'calculated_percentage': (1 - remain_proportion)
        }
            
        return sorted(scores.items(), key=lambda x:x[1], reverse=True), info
            
    def _get_champion_list(self, term):
        return self.inverted.get(term, None)
        
    def _exact_computation(self, query, similar_idxs):
        scores = {}
        for qt, qw, _ in query:
            
            champ_list = self._get_champion_list(qt)
            if champ_list == None:
                continue

            for w, num, docs in zip(*champ_list):
                for d in similar_idxs:
                    if d in docs:
                        scores[d] = scores.get(d, 0) + (w * qw)
        
        return sorted(scores.items(), key=lambda x:x[1], reverse=True)
    
    def save(self, model_prefix):
        self._save_inverted_index('%s_inverted_index' % model_prefix)
        self._save_idf('%s_idf' % model_prefix)
    
    def shape(self):
        return (self.num_doc, self.num_term)
    
    def _save_inverted_index(self, inverted_index_file):
        try:
            with open(inverted_index_file, 'wb') as f:
                pickle.dump(dict(self.inverted), f)
        except Exception as e:
            print(e, 'from _save_inverted_index()')
    
    def _save_idf(self, idf_file):
        try:
            with open(idf_file, 'wb') as f:
                pickle.dump(self.idf, f)
        except Exception as e:
            print(e, 'from _save_idf()')
    
    def load(self, model_prefix):
        self._load_inverted_index('%s_inverted_index' % model_prefix)
        self._load_idf('%s_idf' % model_prefix)
        
    def _load_inverted_index(self, inverted_index_file):
        try:
            with open(inverted_index_file, 'rb') as f:
                self.inverted = pickle.load(f)
        except Exception as e:
            print(e, 'from _load_inverted_index()')
    
    def _load_idf(self, idf_file):
        try:
            with open(idf_file, 'rb') as f:
                self.idf = pickle.load(f)
        except Exception as e:
            print(e, 'from _load_idf()')