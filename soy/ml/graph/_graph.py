import os
from collections import defaultdict

import numpy as np


class Graph:
    
    def __init__(self):
        self.node2int = defaultdict(lambda: len(self.node2int))
        self.int2node = {}
        self.inb = defaultdict(lambda: defaultdict(lambda: 0))
        self.outb = defaultdict(lambda: defaultdict(lambda: 0))

    def add(self, from_node, to_node, weight=1.0, undirected=False):
        from_node = self._type_check(from_node)
        to_node = self._type_check(to_node)
        self.inb[to_node][from_node] += weight
        self.outb[from_node][to_node] += weight
        
        if undirected:
            self.inb[from_node][to_node] += weight
            self.outb[to_node][from_node] += weight
        
    def _type_check(self, node, add_unknown=True):
        if type(node) != int:
            if add_unknown:
                idx = self.node2int[node]
                self.int2node[idx] = node
                return idx
            else:
                return self.node2int.get(node, -1)
        return node
    
    def inbounds(self, to_node):
        if (type(to_node) != int) and ((to_node in self.node2int) == False):
            return {}
        to_node = self._type_check(to_node)
        return dict(self.inb.get(to_node, {}))
    
    def outbounds(self, from_node):
        if (type(from_node) != int) and ((from_node in self.node2int) == False):
            return {}
        from_node = self._type_check(from_node)
        return dict(self.outb.get(from_node, {}))
    
    def normalize(self, base='from'):
        if base == 'from':
            self._outbound_normalize()
        else:
            self._inbound_normalize()
    
    def _inbound_normalize(self):
        for to_node in self.inb.keys():
            inbs = self.inb[to_node]
            sum_ = sum(inbs.values())
            
            for from_node, w in inbs.items():
                inbs[from_node] = w / sum_
                self.outb[from_node][to_node] = w / sum_
            self.inb[to_node] = inbs
        
    def _outbound_normalize(self):
        for from_node in self.outb.keys():
            outbs = self.outb[from_node]
            sum_ = sum(outbs.values())
    
            for to_node, w in outbs.items():
                outbs[to_node] = w / sum_
                self.inb[to_node][from_node] = w / sum_
            self.outb[from_node] = outbs
            
    def nodes(self):
        node_set = set(self.inb.keys())
        node_set.update(set(self.outb.keys()))
        return node_set
        

class GraphInterface:
    
    def add(self, f, t, w):
        raise NotImplemented

    def remove_edge(self, f=None, t=None):
        '''
        Parameters
        ----------
            f, t: int or None
                If f is None and t is int, remove all edges of which destination is t
                If t is None and f is int, remove all edges of which source is f            
                if both f and t are None, do nothing
        '''
        raise NotImplemented
        
    def as_undiriected(self):
        raise NotImplemented
        
    def outb(self, node):
        raise NotImplemented

    def outb_nodes(self):
        raise NotImplemented
    
    def inb(self, node):
        raise NotImplemented
        
    def inb_nodes(self):
        raise NotImplemented

    def E(self):
        raise NotImplemented

    def N(self):
        raise NotImplemented
        
    def normalize_edge(self, method='sum'):
        '''
        Parameters
        ----------
            method: str
                available value = ['sum', 'exp']
        '''
        raise NotImplemented
        
    def load(self, fname, delimiter=',', skip_head = 1):
        if not os.path.exists(fname):
            raise FileNotFoundError('graph file was not found: %s' % fname)
        else:
            with open(fname, encoding='utf-8') as f:
                try:
                    for num_line, line in enumerate(f):
                        # skip header E = %d, N = %d
                        if num_line < skip_head:
                            continue    
                        u, v, w = line.replace('\n', '').split(delimiter)
                        u = int(u)
                        v = int(v)
                        w = float(w)
                        self.add(u, v, w)
                except:
                    raise ValueError('parsing error (line %d) %s' % (num_line, line) )
    
    def save(self, fname, delimiter=','):
        raise NotImplemented


class list_graph(GraphInterface):
    
    def __init__(self, fname = None):
        self._outb_n = []
        self._outb_w = []
        self._inb_n = []
        self._inb_w = []

        if fname:
            self.load(fname)

    def __getitem__(self, uv, not_exist = 0.0):
        u = uv[0]
        v = uv[1]
        if max(u,v) > len(self._outb_n):
            return not_exist
        if v in self._outb_n[u]:
            return self._outb_n[v]
        else:
            return not_exist
            
    def add(self, f, t, w):
        raise NotImplemented
        
    def remove_edge(self, f=None, t=None):
        '''
        Parameters
        ----------
            f, t: int or None
                If f is None and t is int, remove all edges of which destination is t
                If t is None and f is int, remove all edges of which source is f            
                if both f and t are None, do nothing
        '''
        raise NotImplemented
        
    def as_undiriected(self):
        raise NotImplemented
        
    def outb(self, node):
        
        raise NotImplemented
    
    def outb_nodes(self):
        
        raise NotImplemented
        
    def inb(self, node):
        
        raise NotImplemented
        
    def inb_nodes(self):
        
        raise NotImplemented
    
    def E(self):
        
        raise NotImplemented
    
    def N(self):
        
        raise NotImplemented
        
    def normalize_edge(self, method='sum'):
        '''
        Parameters
        ----------
            method: str
                available value = ['sum', 'exp']
        '''
        
    def save(self, fname):
        
        raise NotImplemented
        
    def to_dictgraph(self):
        
        raise NotImplemented


class dict_graph(GraphInterface):
    
    def __init__(self, fname = None):
        self._outb = defaultdict(lambda: defaultdict(lambda: 0.0))
        self._inb = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        if fname:
            self.load(fname)

    def __getitem__(self, uv, not_exist = 0.0):
        u = uv[0]
        v = uv[1]
        if u in self._outb:
            if v in self._outb[u]:
                return self._outb[u][v]
            else:
                return not_exist
        return not_exist
            
    def add(self, f, t, w):
        self._outb[f][t] += w
        self._inb[t][f] += w
        
        if self._outb[f][t] == 0.0:
            del self._outb[f][t]
            del self._inb[t][f]
            
            if not self._outb[f]:
                del self._outb[f]
            
            if not self._inb[t]:
                del self._inb[t]

    def remove_edge(self, f=None, t=None):
        '''
        Parameters
        ----------
            f, t: int or None
                If f is None and t is int, remove all edges of which destination is t
                If t is None and f is int, remove all edges of which source is f            
                if both f and t are None, do nothing
        '''
        if (f == None) and (t == None):
            return 0
        
        elif (f != None) and (t != None):
            w = self.__getitem__((f,t))
            if w == 0.0:
                return 0
            else:
                self.add(f, t, -w)
                return 1
            
        elif (f == None):
            f_list = [f for f in self._inb[t]]
            num = len(f_list)
            for f in f_list:
                self.add(f, t, -1 * self.__getitem__((f,t)))
            if t in self._inb:
                del self._inb[t] 
            return num
        
        elif (t == None):
            t_list = [t for t in self._outb[f]]
            num = len(t_list)
            for t in t_list:
                self.add(f, t, -1 * self.__getitem__((f,t)))
            if f in self._outb:
                del self._outb[f] 
            return num
    
    def as_undiriected(self):
        outb_copy = {u:{v:w for v,w in v_dict.items()} for u,v_dict in self._outb.items()}        
        for u,v_dict in outb_copy.items():
            for v,w in v_dict.items():
                self.add(v,u,w)
        del outb_copy
        
    def outb(self, node):
        if node in self._outb:
            return self._outb[node]
        else:
            return defaultdict(lambda: 0.0)
    
    def outb_nodes(self):
        return self._outb.keys()
        
    def inb_nodes(self):
        return self._inb.keys()
    
    def inb(self, node):
        if node in self._inb:
            return self._inb[node]
        else:
            return defaultdict(lambda: 0.0)
    
    def E(self):
        return sum([len(v) for v in self._outb.values()])
    
    def N(self):
        return len(set(list(self._outb.keys()) + list(self._inb.keys())))
    
    def normalize_edge(self, method='sum'):
        '''
        Parameters
        ----------
            method: str
                available value = ['sum', 'exp']
                default is 'sum'
        '''
        def normalize_exp(d):
            return normalize_sum({k:np.exp(v) for k,v in d.items()})
        
        def normalize_sum(d):
            sum_ = sum(d.values())
            return defaultdict(lambda: 0.0, {k:v/sum_ for k,v in d.items()})
        
        for u, v_dict in self._outb.items():
            self._outb[u] = normalize_exp(v_dict) if method == 'exp' else normalize_sum(v_dict)
        for v, u_dict in self._inb.items():
            self._inb[v] = normalize_exp(u_dict) if method == 'exp' else normalize_sum(u_dict)
                    
    def save(self, fname, delimiter=','):
        fname = fname.replace('\\', '/')
        folder = '/'.join(fname.split('/')[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('E = %d, N = %d\n' % (self.E(), self.N()))
            for u, v_dict in self._outb.items():
                for v, w in self._outb[u].items():
                    f.write('%d%s%d%s%f\n' % (u, delimiter, v, delimiter, w))
        
    def to_listgraph(self):
        
        raise NotImplemented




