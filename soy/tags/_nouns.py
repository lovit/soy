class LRNounExtractor:
    
    def __init__(self, r_score_file):
        self._load_r_score(r_score_file)
        
        
    def _load_r_score(self, fname):
        self.r_score = {}
        try:
            with open(fname, encoding='utf-8') as f:
                for num_line, line in enumerate(f):
                    r, score = line.split('\t')
                    score = float(score)
                    self.r_score[r] = score
            print('%d r features was loaded' % len(self.r_score))
        except FileNotFoundError:
            print('r_score_file was not found')
        except Exception as e:
            print('%s parsing error line (%d) = %s' % (e, num_line, line))
    
    
    def predict(self, r_features):
        '''
        Parameters
        ----------
            r_features: dict
                r 빈도수 dictionary
                예시: {을: 35, 는: 22, ...}
        '''
        
        score = 0
        norm = 0
        unknown = 0
        
        for r, freq in r_features.items():
            if r in self.r_score:
                score += freq * self.r_score[r]
                norm += freq
            else:
                unknown += freq
        
        return (score / norm, norm / (norm + unknown))

