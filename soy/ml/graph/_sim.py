from collections import defaultdict

from soy.ml.graph import GraphInterface, list_graph, dict_graph

class RandomWalkWithRestart:
    
    def __init__(self, graph, name=None):
        
        if not self._check_normalization(graph):
            return None
        
        self.g = graph
        self.N = graph.N()
        self.name = name if name else ''
        
    def _check_normalization(self, g):
        for u in g.outb_nodes():
            sum_ = sum([w for _, w in g.outb(u).items()])
            if not (0.99 < sum_ < 1.01):
                print('You should normalize graph edge first (u = %d has weight sum = %.3f)' % (u, sum_))
                return False
        return True

    def get_similarity(self, node, max_steps = 6, df = 0.85, bipartite=False):
        '''
        Parameters
        ----------
            node: int
                Node id
            num_steps: int
                Number of random walk steps 
            df: float
                Decaying factor that should be in [0, 1]
        Returns
        -------
            list like with similarity value
        '''
        sim = defaultdict(lambda: 0.0, {node:1.0})
        
        num_step = 0
        while num_step < max_steps:
            
            next_walkers = defaultdict(lambda: 0.0)
            for walker, weight in sim.items():
                for outb, outbw in self.g.outb(walker).items():
                    next_walkers[outb] += weight * outbw
            
            if df > 0:
                if (not bipartite) or (num_step % 2 == 1):
                    for walker, weight in next_walkers.items():
                        next_walkers[walker] *= df
                    next_walkers[node] += (1 - df)
            
            sim = next_walkers
            num_step += 1
            
        return sim
