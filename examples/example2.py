import os
import sys
sys.path.insert(0, "./")
sys.path.insert(1, "../")
sys.path.insert(2, "./examples/")
import numpy as np
import matplotlib.pyplot as plt
from consts import *
from utils.tools import read_corpus_file
from tne.tne import TNE
import networkx as nx

# Set all parameters #
params = {}
params['comm_detection_method'] = "lda"
params['number_of_comms'] = 2
# Common parameters
params['node_embedding_size'] = 96
params['comm_embedding_size'] = 32
# Parameters for SkipGram
params['window_size'] = 10
params['num_of_workers'] = 3
# Parameters for LDA
params['lda_alpha'] = 50 / float(params['number_of_comms'])  # Default is 50 / K
params['lda_beta'] = 0.1  # Default is 0.1
params['lda_number_of_iters'] = 1000  # Default is 1000
# Parameters for hmm
params['hmm_p0'] = 0.3
params['hmm_t0'] = 0.2
params['hmm_e0'] = 0.1
params['hmm_number_of_iters'] = 1000
params['hmm_subset_size'] = 100


corpus_path = os.path.join(BASE_FOLDER, "examples", "corpus", "karate.corpus")
dataset_path = os.path.join(BASE_FOLDER, "examples", "datasets", "karate.gml")
params['graph_path'] = dataset_path

# Read the graph
g = nx.read_gml(dataset_path)
# Read the walks
walks = read_corpus_file(corpus_path)
# Call the main class
tne = TNE(walks=walks, params=params)

# Get the community assignments
phi = np.asarray(tne.phi)
phi = phi / np.sum(tne.phi, 0)

plt.figure()
node2comm = {tne.id2node[nodeId]: np.argmax(phi, 0)[nodeId] for nodeId in tne.id2node.keys()}

colors=['r', 'b', 'g', 'y', 'm', 'k']
print(tne.K)
pos = nx.spring_layout(g)
for commId in range(tne.K):
    nodelist = [node for node in node2comm.keys() if node2comm[node] == commId]
    nx.draw_networkx_nodes(g, pos,
                           nodelist=nodelist,
                           node_color=colors[commId],
                           node_size=100,
                           alpha=0.8)
nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.8)
plt.show()
