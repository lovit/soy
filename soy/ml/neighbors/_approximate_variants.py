from collections import defaultdict
import pickle
import time
import numpy as np

from ._approximate import FastCosine


class FastQueryExpansionCosine(FastCosine):

    def __init__(self):
        FastCosine.__init__(self)
        self.num_concept = 0
        self.term_to_concept = {}

    def indexing(self, mm_file, expansion_file, max_num_doc=-1):

        t2d, norm_dt = self._load_mm(mm_file, max_num_doc)
        print('loaded mm')

        t2c = self._load_expansion_rules(expansion_file)
        self._set_term_to_concept_mapper(t2c)
        print('loaded term to concept mapper')

        c2d, norm_dc = self._as_concept_vector(t2d, t2c)
        print('convert term vector as concept vector')
        
        t2d = self._normalize_weight(t2d, norm_dt)
        print('normalized t2d weight')
        
        c2d = self._normalize_weight(c2d, norm_dc)
        print('normalized c2d weight')
        
        t2d = self._merge_term_and_concept_matrix(t2d, c2d)
        print('merged term mat. and concept mat.')
        
        self._build_champion_list(t2d)
        print('builded champion list')
        
        self._build_idf(t2d)
        print('computed search term order (idf)')

        del t2d
        del c2d

    def _load_expansion_rules(self, expansion_file):
        with open(expansion_file, 'rb') as f:
            t2c = pickle.load(f)
        self.num_concept = max({c for cdict in t2c.values() for c in cdict.keys()}) + 1
        return t2c

    def _set_term_to_concept_mapper(self, t2c):
        for t, c2w in t2c.items():
            t2c_ = {(c + self.num_term):cw for c, cw in c2w.items()}
            self.term_to_concept[t] = t2c_
                
    def _as_concept_vector(self, t2d, t2c):
        c2d = defaultdict(lambda: defaultdict(lambda: 0.0))
        norm_dc = defaultdict(lambda: 0.0)
        
        for term, d2w in t2d.items():
            c2w = t2c.get(term, {})
            if not c2w:
                continue
            for d, dw in d2w.items():
                for c, cw in c2w.items():
                    w = (dw * cw)
                    c2d[c][d] += w

        for c, d2w in c2d.items():
            for d, dw in d2w.items():
                norm_dc[d] += dw ** 2
        
        for d, dw in norm_dc.items():
            norm_dc[d] = np.sqrt(dw)
        
        return c2d, norm_dc
    
    def _merge_term_and_concept_matrix(self, t2d, c2d):
        extended_t2d = {t:{d:w for d,w in d2w.items()} for t, d2w in t2d.items()}
        for c, d2w in c2d.items():
            extended_t2d[c + self.num_term] = dict(d2w)
        return extended_t2d
    
    def rneighbors(self):
        raise NotImplemented
    
    def _retrieve_within_range(self):
        raise NotImplemented

    def kneighbors(self, query, n_neighbors=10, candidate_factor=10.0, 
                   remain_tfidf_threshold=1.0, max_weight_factor=0.5, 
                   scoring_by_adding=False, compute_true_cosine=True, 
                   normalize_query_with_tfidf=True, 
                   expansion_terms=None, include_terms=None, exclude_terms=None):
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
        
        if expansion_terms:
            if include_terms:
                expansion_terms = [t for t in expansion_terms if (t in include_terms) == False]
            query = self.query_expansion(query, expansion_terms)
            
        n_candidates = int(n_neighbors * candidate_factor)
        scores, info = self._retrieve_similars(query, n_candidates, remain_tfidf_threshold, max_weight_factor, scoring_by_adding, include_terms, exclude_terms)
        scores = scores[:n_neighbors]
        times['retrieval_similars'] = self._get_process_time()
        
        if expansion_terms:
            exp_query_term = [t[0] for t in query]
            exp_query_term = [t if t < self.num_term else 'C%d' % (t - self.num_term) for t in exp_query_term]
            info['expanded_query_terms'] = exp_query_term
        
        if compute_true_cosine:
            neighbors_idx, _ = zip(*scores)
            scores = self._exact_computation(query, neighbors_idx)
        times['true_cosine_computation'] = self._get_process_time()
        times['whole_querying_process'] = sum(times.values())
        info['time [mil.sec]'] = times
        return scores, info
    
    def query_expansion(self, query, expansion_terms):
        concept_index = {}
        expanded_query = []
        
        for qt, qw, tfidf in query:
            if (qt in expansion_terms) == False:
                expanded_query.append((qt, qw, tfidf))
                continue
            
            concepts = self.term_to_concept.get(qt, {})
            if not concepts:
                expanded_query.append((qt, qw, tfidf))
                continue
            
            c_norm = np.sqrt(sum(v ** 2 for v in concepts.values()) / qw ** 2)
            for ct, cw in concepts.items():
                if ct in concept_index:
                    i = concept_index[ct]
                    ctw = expanded_query[i]
                    expanded_query[i] = (ctw[0], ctw[1] + cw / c_norm, ctw[2])
                    continue
                expanded_query.append((ct, cw / c_norm, tfidf))
                concept_index[ct] = len(expanded_query) - 1
                
        return expanded_query

    def save(self, fname):
        if fname[-4:] != '.pkl':
            fname = fname + '.pkl'
        params = {
            'inverted_index': dict(self._inverted),
            'idf': self._idf,
            'max_dw': self._max_dw,
            'num_doc': self.num_doc,
            'num_term': self.num_term,
            'num_concept': self.num_concept,
            'term_to_concept': self.term_to_concept
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
        self.num_concept = params.get('num_concept', 0)
        self.term_to_concept = params.get('term_to_concept', {})
