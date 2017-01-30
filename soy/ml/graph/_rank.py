from collections import defaultdict
import sys


class PageRank:
    
    def __init__(self, graph):
        # TODO type check
        self.graph = graph
        
    def train(self, beta=0.85, max_iter=30, bias={}, debug={}):
        if self.graph == None:
            raise TypeError('PageRank input must be soy.ml.Graph')
        
        node_set = self.graph.nodes()
        
        rank = {node:1.0 for node in node_set}
        for num_iter in range(1, max_iter + 1):
            
            rank_ = {}
            for node in node_set:
                
                inbs = self.graph.inbounds(node)
                if not inbs:
                    continue
                    
                rank_[node] = sum((rank.get(from_node, 0) * w) for from_node, w in inbs.items())
                rank_[node] = beta * rank_[node] + ( 1 - beta ) * rank.get(node, 1)
            
            rank = rank_
            sys.stdout.write('\riter = %d in %d' % (num_iter, max_iter))
            
            if debug:
                print()
                for node in debug:
                    _node = self.graph._type_check(node, add_unknown=False)
                    if _node == -1: continue
                    print('[node: %s] = %.5f' % (str(self.graph.int2node[_node]), rank.get(_node, 0)) )
                print()
        
        return rank