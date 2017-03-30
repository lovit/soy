from collections import Counter, defaultdict
import os
import pickle
from pprint import pprint
import time
import numpy as np


class FastCosine():
    
    def __init__(self):
        self._inverted = defaultdict(lambda: [])
        self._idf = {}
        self._max_dw = {}
        self.num_doc = 0
        self.num_term = 0

        self._base_time = None

    def _get_process_time(self):
        if self._base_time == None:
            self._base_time = time.time()
        process_time = 1000 * (time.time() - self._base_time)
        self._base_time = time.time()
        return process_time
    
    def indexing(self, mm_file, max_num_doc=-1):
        t2d, norm_d = self._load_mm(mm_file, max_num_doc)
        print('loaded mm')
        
        t2d = self._normalize_weight(t2d, norm_d)
        print('normalized t2d weight')
        
        self._build_champion_list(t2d)
        print('builded champion list')
        
        self._build_idf(t2d)
        print('computed search term order (idf)')
        
        del t2d
    
    def _load_mm(self, mm_file, max_num_doc=-1):
        t2d = defaultdict(lambda: {})
        norm_d = defaultdict(lambda: 0)
        max_dw = defaultdict(lambda: 0)
        
        if not os.path.exists(mm_file):
            raise IOError('mm file not found: %s' % mm_file)
        
        with open(mm_file, encoding='utf-8') as f:
            # Skip three head lines
            for _ in range(2):
                next(f)
            
            nums = next(f).split()
            nums = [int(n) for n in nums]
            self.num_doc = nums[0]
            self.num_term = nums[1]
            
            # line format: doc term freq
            try:
                for line in f:
                    doc, term, freq = line.split()

                    doc = int(doc) - 1
                    term = int(term) - 1
                    freq = float(freq)

                    if (0 < max_num_doc) and (max_num_doc <= doc):
                        self.num_doc = max_num_doc
                        continue
                    
                    t2d[term][doc] = freq
                    norm_d[doc] += freq ** 2
                    max_dw[doc] = max(max_dw[doc], freq)
            except Exception as e:
                print('mm file parsing error %s' % line)
                print(e)
        
        for d, v in norm_d.items():
            norm_d[d] = np.sqrt(v)
            max_dw[d] = (max_dw[d] / norm_d[d]) if v != 0 else 0
        
        self._max_dw = dict(max_dw)
        
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
            self._inverted[t] = pack(wd)
        
    def _build_idf(self, t2d):
        for t, d_dict in t2d.items():
            self._idf[t] = np.log(self.num_doc / len(d_dict))
    
    def rneighbors(self, query, min_cosine_range=0.8, remain_tfidf_threshold=1.0, weight_factor=0.5, normalize_query_with_tfidf=False):
        raise NotImplemented

#         times = {}
#         self._get_process_time()
        
#         query = self._check_query(query, normalize_query_with_tfidf)
#         if not query:
#             return [], {}
#         times['check_query_type'] = self._get_process_time()
        
#         query = self._order_search_term(query)
#         times['order_search_term'] = self._get_process_time()
        
#         scores, info = self._retrieve_within_range(query, remain_tfidf_threshold, weight_factor)
        
#         times['whole_querying_process'] = sum(times.values())
#         info['time [mil.sec]'] = times
#         return scores, info
    
#     def _retrieve_within_range(self, query, remain_tfidf_threshold=1.0, weight_factor=0.5, scoring_by_adding=False):
        
#         def select_champs(champ_list, threshold):
#             more_than_threshold = -1
#             for i, (w, num, docs) in enumerate(zip(*champ_list)):
#                 if w < threshold:
#                     break
#                 more_than_threshold = i
#             i = more_than_threshold
#             if i < 0:
#                 return None
#             return [champ_list[0][:i+1], champ_list[1][:i+1], champ_list[2][:i+1]]
        
#         def get_max_query_term_weight(query):
#             max_weight = [0]
#             max_ = 0
#             for qt, qw, _ in reversed(query[1:]):
#                 max_ = max(max_, qw)
#                 max_weight.append(max_)
#             return list(reversed(max_weight))

#         scores = {}
#         remain_proportion = 1
        
#         n_computation = 0
#         n_considered_terms = 0
        
#         max_qtws = get_max_query_term_weight(query)
#         expand_candidates = True

#         for (qt, qw, tfidf), max_qtw in zip(query, max_qtws):

#             n_considered_terms += 1
#             remain_proportion -= (qw ** 2)
            
#             champ_list = self._get_champion_list(qt)
#             if champ_list == None:
#                 continue

#             threshold = max(0, qw * weight_factor)
#             champ_list = select_champs(champ_list, threshold)
#             if champ_list == None:
#                 continue

# #            print('qt = %d, qw = %.3f, t = %.3f' % (qt, qw, threshold))
# #            print(champ_list) # DEVCODE

#             if expand_candidates:   
#                 for w, num, docs in zip(*champ_list):
#                     for d in docs:
#                         scores[d] = scores.get(d, 0) + (w if scoring_by_adding else qw * w)
#                     n_computation += num
#             else:
#                 # TODO
#                 pass

#             if (remain_proportion * tfidf) < remain_tfidf_threshold:
#                 break

#         info = {
#             'n_computation': n_computation, 
#             'n_considered_terms': n_considered_terms, 
#             'n_terms_in_query': len(query),
#             'n_candidate': len(scores),
#             'calculated_percentage': (1 - remain_proportion)
#         }
            
#         return sorted(scores.items(), key=lambda x:x[1], reverse=True), info

    def kneighbors(self, query, n_neighbors=10, candidate_factor=10.0, remain_tfidf_threshold=1.0, max_weight_factor=0.5, scoring_by_adding=False, compute_true_cosine=True, normalize_query_with_tfidf=True, include_terms=None, exclude_terms=None):
        '''query: {term:weight, ..., }
        
        '''
        
        times = {}
        self._get_process_time()
        
        query = self._check_query(query, normalize_query_with_tfidf)
        if not query:
            return [], {}
        times['check_query_type'] = self._get_process_time()
        
        query = self._order_search_term(query)
        times['order_search_term'] = self._get_process_time()
        
        n_candidates = int(n_neighbors * candidate_factor)
        scores, info = self._retrieve_similars(query, n_candidates, remain_tfidf_threshold, max_weight_factor, scoring_by_adding, include_terms, exclude_terms)
        scores = scores[:n_neighbors]
        times['retrieval_similars'] = self._get_process_time()
        
        if compute_true_cosine and scores:
            neighbors_idx, _ = zip(*scores)
            scores = self._exact_computation(query, neighbors_idx)
        times['true_cosine_computation'] = self._get_process_time()
        times['whole_querying_process'] = sum(times.values())
        info['time [mil.sec]'] = times
        return scores, info
    
    def _check_query(self, query, normalize_query_with_tfidf=False):
        query = {t:w for t,w in query.items() if ((t in self._idf) and (0 <= t < self.num_term))}
        if normalize_query_with_tfidf:
            query = {t:w * self._idf[t] for t,w in query.items()}
        sum_ = sum(v ** 2 for v in query.values())
        sum_ = np.sqrt(sum_)
        query = {t:w/sum_ for t,w in query.items()}
        return query

    def _order_search_term(self, query):
        query = [(qt, qw, qw * self._idf[qt]) for qt, qw in query.items()]
        query = sorted(query, key=lambda x:x[2], reverse=True)
        return query
    
    def _retrieve_similars(self, query, n_candidates, remain_tfidf_threshold=0.5, max_weight_factor=0.2, scoring_by_adding=False, include_terms=None, exclude_terms=None):

        def select_champs(champ_list, n_candidates, max_weight_factor=0.2):
            threshold = (champ_list[0][0] * max_weight_factor)
            sum_num = 0
            for i, (w, num, docs) in enumerate(zip(*champ_list)):
                sum_num += num
                if ((i > 0) and (w < threshold)) or ((i > 0) and (n_candidates > 0) and (sum_num >= n_candidates)):
                    i = i - 1
                    break
            return [champ_list[0][:i+1], champ_list[1][:i+1], champ_list[2][:i+1]]

        include_candidates = None
        if include_terms:
            include_candidates = self._get_docs_having_all_terms(include_terms)
        
        exclude_candidates = {}
        if exclude_terms:
            exclude_candidates = self._get_docs_having_at_least_one(exclude_terms)

        scores = {}
        remain_proportion = 1
        
        n_computation = 0
        n_considered_terms = 0

        for qt, qw, tfidf in query:

            n_considered_terms += 1
            remain_proportion -= (qw ** 2)
            
            champ_list = self._get_champion_list(qt)
            if champ_list == None:
                continue
            
            champ_list = select_champs(champ_list, n_candidates, max_weight_factor)
            
            if include_candidates == None:
                for w, num, docs in zip(*champ_list):
                    for d in docs:
                        if d in exclude_candidates:
                            continue
                        scores[d] = scores.get(d, 0) + (w if scoring_by_adding else qw * w)
                    n_computation += num
            else:
                for w, num, docs in zip(*champ_list):
                    for d in docs:
                        if ((d in include_candidates) == False) or (d in exclude_candidates):
                            continue
                        scores[d] = scores.get(d, 0) + (w if scoring_by_adding else qw * w)
                        n_computation += num

            if (remain_proportion * tfidf) < remain_tfidf_threshold:
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
        return self._inverted.get(term, None)

    def get_all_docs(self, term):
        w, num, docs = self._inverted.get(term, (None, None, None))
        if not docs:
            return {}
        return {d for doc_tuple in docs for d in doc_tuple}
    
    def _get_docs_having_all_terms(self, terms):
        intersections = {}
        for term in terms:
            if (term in self._inverted) == False:
                continue
            if not intersections:
                intersections = self.get_all_docs(term)
                continue
            intersections = intersections.intersection(self.get_all_docs(term))
        return intersections
    
    def _get_docs_having_at_least_one(self, terms):
        unions = set()
        for term in terms:
            if (term in self._inverted) == False:
                continue
            unions.update(self.get_all_docs(term))
        return unions

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
    
    def shape(self):
        return (self.num_doc, self.num_term)
    
    def save(self, fname):
        if fname[-4:] != '.pkl':
            fname = fname + '.pkl'
        params = {
            'inverted_index': dict(self._inverted),
            'idf': self._idf,
            'max_dw': self._max_dw,
            'num_doc': self.num_doc,
            'num_term': self.num_term
        }
        with open(fname, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, fname):
        with open(fname, 'rb') as f:
            params = pickle.load(f)
        self._inverted = params.get('inverted_index', {})
        self._idf = params.get('idf', {})
        self._max_dw = params.get('max_dw', {})
        self.num_doc = params.get('num_doc', 0)
        self.num_term = params.get('num_term', 0)
