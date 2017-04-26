from collections import defaultdict
from datetime import date, timedelta
import time
import psutil


def daterange(start_date, end_date, as_str=True):
    for n in range(int ((end_date - start_date).days)):
        if as_str:
            yield str(start_date + timedelta(n))
        else:
            yield start_date + timedelta(n)


def progress(i, n, length=30, header='', base_time = None):
    
    perc = int(length * i / float(n))
    message = ('\r%s: ' % header) + ('#' * perc) + ('-' * (30 - perc)) + ' (%.3f %s)' % (100 * i / n, '%')
    
    if base_time != None:
        remain_time = ((time.time() - base_time) / i * (n - i + 1))
        
        if remain_time > 10000:
            message += ' remained %.3f hours' % (remain_time / 3600.0)
        elif remain_time > 600:
            message += ' remained %.3f mins' % (remain_time / 60.0)
        else:
            message += ' remained %.3f secs' % (remain_time)
            
    return message

def get_available_memory():
    mem = psutil.virtual_memory()
    return 100 * mem.available / (mem.total)

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


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
