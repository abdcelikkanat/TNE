import networkx as nx
import numpy as np

_random_walk_strategy_names = ['deepwalk', 'node2vec']


class RandomWalks:

    g = None  # networkx graph
    strategy_name = ""  # random walk strategy
    N = 0  # number of walks
    L = 0  # walk length
    opts = dict()  # optional parameters
    walks = []  # the generated walks

    def __init__(self, g, strategy_name, N, L, opts):

        if isinstance(g, nx.Graph):
            self.g = g

        if strategy_name in _random_walk_strategy_names:
            self.strategy_name = strategy_name
        else:
            raise ValueError("Invalid random walk strategy name!")

        if N > 0:
            self.N = N

        if L > 0:
            self.L = L

        if isinstance(opts, dict):
            self.opts = opts

        # Perform random walks
        generate_walks = getattr(self, '_' + self.strategy_name)
        generate_walks()

    def _deepwalk(self):
        alpha = 0.0
        if 'dw_alpha' in self.opts:
            alpha = self.opts['dw_alpha']

        self.walks = []
        node_list = list(self.g.nodes())
        for _ in range(self.N):
            np.random.shuffle(node_list)

            for node in node_list:
                walk = [node]
                while len(walk) < self.L:
                    current_node = walk[-1]
                    if np.random.random() >= alpha:
                        walk.append(np.random.choice(a=list(nx.neighbors(self.g, current_node)),size=1)[0])
                    else:
                        walk.append(walk[0])

                self.walks.append(walk)

    def _node2vec(self):

        raise ValueError("Not implemented!")

    def get_walks(self):

        return self.walks

    def write_walks(self, output_path):

        with open(output_path, 'w') as f:

            for walk in self.walks:
                f.write("{}\n".format(" ".join(str(w) for w in walk)))
