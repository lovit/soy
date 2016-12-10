from collections import defaultdict
from datetime import date, timedelta

def daterange(start_date, end_date, as_str=True):
    for n in range(int ((end_date - start_date).days)):
        if as_str:
            yield str(start_date + timedelta(n))
        else:
            yield start_date + timedelta(n)


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
    
    
    def __getitem__(self, x):
        if type(x) == int:
            if x < self.num_object:
                return self.inverse[x]
            else:
                return None
        if x in self.mapper:
            return self.mapper[x]
        else:
            return -1    
        
        
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
