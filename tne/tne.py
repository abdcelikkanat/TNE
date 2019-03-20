import os
import sys
import time
import numpy as np
import networkx as nx
from settings import *
from utils.utils import *
from ext.gensim_wrapper.models.word2vec import Word2VecWrapper
from gensim.utils import smart_open
import community as louvain

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ext/deepwalk/deepwalk")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ext/node2vec/src")))
lda_exe_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ext/gibbslda/lda"))


try:
    import graph as deepwalk
    import node2vec
    if not os.path.exists(lda_exe_path):
        raise ImportError
except ImportError:
    raise ImportError("An error occurred during loading the external libraries!")


class WalkIterator:
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for walk in self.corpus:
            yield walk


class TNE:
    def __init__(self, graph_path=None, temp_folder="", params={}):
        self.graph = None
        self.graph_name = ""
        self.number_of_nodes = 0
        self.number_of_communities = 0
        self.corpus = []
        self.topic_corpus = []
        self.N = 0
        self.L = 0
        self.number_of_communities = 0
        self.temp_folder = temp_folder
        self.params = params
        self.model = None

        if graph_path is not None:
            self.read_graph(graph_path)

    def set_temp_folder(self, temp_folder):
        self.temp_folder = temp_folder

    def get_number_of_nodes(self):
        return self.graph.number_of_nodes()

    def read_graph(self, filename, filetype=".gml"):
        dataset_name = os.path.splitext(os.path.basename(filename))[0]

        if filetype == ".gml":
            g = nx.read_gml(filename)
            print("Dataset: {}".format(dataset_name))
            print("The number of nodes: {}".format(g.number_of_nodes()))
            print("The number of edges: {}".format(g.number_of_edges()))

            self.number_of_nodes = g.number_of_nodes()
            self.graph = g
            self.graph_name = dataset_name
        else:
            raise ValueError("Invalid file type!")

    def set_graph(self, graph, graph_name="unknown"):

        self.graph = graph
        self.number_of_nodes = self.graph.number_of_nodes()
        self.graph_name = graph_name

        print("Graph name: {}".format(self.graph_name))
        print("The number of nodes: {}".format(self.graph.number_of_nodes()))
        print("The number of edges: {}".format(self.graph.number_of_edges()))

    def get_graph(self):
        return self.graph

    def set_params(self, params):
        self.params = params

    def get_node_corpus(self):
        return self.corpus

    def get_topic_corpus(self):
        return self.topic_corpus

    def perform_random_walks(self, output_node_corpus_file):

        if not ('number_of_walks' and 'walk_length') in self.params.keys() or self.graph is None:
            raise ValueError("Missing parameter !")

        self.number_of_nodes = self.graph.number_of_nodes()
        self.N = self.number_of_nodes * self.params['number_of_walks']
        self.L = self.params['walk_length']

        initial_time = time.time()
        # Generate a corpus

        if self.params['random_walk'] == "deepwalk":
            if not ('dw_alpha') in self.params.keys():
                raise ValueError("A parameter is missing!")

            # Temporarily generate the edge list
            with open(os.path.join(self.temp_folder,  "graph_deepwalk.edgelist"), 'w') as f:
                for line in nx.generate_edgelist(self.graph, data=False):
                    f.write("{}\n".format(line))

            dwg = deepwalk.load_edgelist(os.path.join(self.temp_folder, "graph_deepwalk.edgelist"), undirected=True)
            self.corpus = deepwalk.build_deepwalk_corpus(G=dwg, num_paths=self.params['number_of_walks'],
                                                         path_length=self.params['walk_length'],
                                                         alpha=self.params['dw_alpha'])

        elif self.params['random_walk'] == "node2vec":

            if not ('n2v_p' and 'n2v_q') in self.params.keys():
                raise ValueError("A missing parameter exists!")

            for edge in self.graph.edges():
                self.graph[edge[0]][edge[1]]['weight'] = 1
            G = node2vec.Graph(nx_G=self.graph, p=self.params['n2v_p'], q=self.params['n2v_q'], is_directed=False)
            G.preprocess_transition_probs()
            self.corpus = G.simulate_walks(num_walks=self.params['number_of_walks'],
                                           walk_length=self.params['walk_length'])

        else:
            raise ValueError("Invalid method name!")

        self.save_corpus(output_node_corpus_file, with_title=False)

        print("The corpus was generated in {:.2f} secs.".format(time.time() - initial_time))

    def save_corpus(self, corpus_file, with_title=False, corpus=None):

        # Save the corpus
        with open(corpus_file, "w") as f:

            if with_title is True:
                f.write(u"{}\n".format(self.N))

            if corpus is None:
                for walk in self.corpus:
                    f.write(u"{}\n".format(u" ".join(v for v in walk)))
            else:
                for walk in corpus:
                    f.write(u"{}\n".format(u" ".join(v for v in walk)))

    def preprocess_corpus(self, process="equalize"):

        if process is "equalize":
            # if a walk consists of only one node due to the disconnected structure of
            # a given network, append the same node to the walk to enlarge it

            for walkId, walk in enumerate(self.corpus):
                if len(walk) != self.params['walk_length']:
                    temp_walk = walk
                    while len(temp_walk) != self.params['walk_length']:
                        temp_walk.append(temp_walk[-1])

                    self.corpus[walkId] = temp_walk

    def learn_node_embedding(self, output_node_embedding_file, workers=3):

        initial_time = time.time()

        if 'negative' not in self.params:
            self.params['hs'] = 1
            self.params['negative'] = 0
        else:
            if self.params['negative'] > 0:
                self.params['hs'] = 0
            else:
                self.params['hs'] = 1

        # Extract the node embeddings
        self.model = Word2VecWrapper(sentences=self.corpus,
                                     size=self.params["embedding_size"],
                                     window=self.params["window_size"],
                                     sg=1, hs=self.params['hs'], negative=self.params['negative'],
                                     workers=workers,
                                     min_count=0)

        # Save the node embeddings
        self.model.wv.save_word2vec_format(fname=output_node_embedding_file)
        print("The node embeddings were generated and saved in {:.2f} secs.".format(time.time() - initial_time))

    def learn_topic_embedding(self, output_topic_embedding_file):

        if 'number_of_communities' not in self.params.keys():
            raise ValueError("The number of topics was not given!")

        # Define the paths for the files generated by GibbsLDA++
        initial_time = time.time()
        self.save_corpus(corpus_file=os.path.join(self.temp_folder, "topic.corpus"),
                         with_title=False, corpus=self.topic_corpus)

        # Construct the tuples (word, topic) with each word in the corpus and its corresponding topic assignment
        combined_sentences = CombineSentences(self.corpus, self.topic_corpus)
        # Extract the topic embeddings
        self.model.train_topic(self.number_of_communities, combined_sentences)
        # Save the topic embeddings
        self.model.wv.save_word2vec_topic_format(fname=output_topic_embedding_file)
        print("The topic embeddings were generated and saved in {:.2f} secs.".format(time.time() - initial_time))

    def generate_community_corpus(self, method=None):

        if 'number_of_communities' not in self.params.keys():
            raise ValueError("the number of topics parameter is missing!")

        self.number_of_communities = self.params['number_of_communities']

        if method == "lda":
            # Run GibbsLDA++
            if not os.path.exists(GIBBSLDA_PATH):
                raise ValueError("Invalid path of GibbsLDA++!")

            temp_lda_folder = os.path.join(self.temp_folder, "lda_temp")
            if not os.path.exists(temp_lda_folder):
                os.makedirs(temp_lda_folder)

            temp_dfile_path = os.path.join(temp_lda_folder, "gibblda_temp.dfile")
            # Save the walks into the dfile
            self.save_corpus(corpus_file=temp_dfile_path, with_title=True, corpus=self.corpus)

            initial_time = time.time()
            cmd = "{} -est ".format(GIBBSLDA_PATH)
            cmd += "-alpha {} ".format(self.params['lda_alpha'])
            cmd += "-beta {} ".format(self.params['lda_beta'])
            cmd += "-ntopics {} ".format(self.params['number_of_communities'])
            cmd += "-niters {} ".format(self.params['lda_number_of_iters'])
            cmd += "-savestep {} ".format(self.params['lda_number_of_iters'] + 1)
            cmd += "-dfile {} ".format(temp_dfile_path)
            os.system(cmd)

            print("-> The LDA algorithm run in {:.2f} secs".format(time.time() - initial_time))

            # Read wordmap file
            id2node = {}
            temp_wordmap_path = os.path.join(temp_lda_folder, "wordmap.txt")
            with open(temp_wordmap_path, 'r') as f:
                f.readline()  # skip the first line
                for line in f.readlines():
                    tokens = line.strip().split()
                    id2node[int(tokens[1])] = tokens[0]

            # Read phi file
            phi = np.zeros(shape=(self.number_of_communities, self.number_of_nodes), dtype=np.float)
            temp_phi_path = os.path.join(temp_lda_folder, "model-final.phi")
            with open(temp_phi_path, 'r') as f:
                for comm, line in enumerate(f.readlines()):
                    for id, value in enumerate(line.strip().split()):
                        phi[comm, int(id2node[id])] = value

            # Read the tassign file, generate topic corpus
            temp_tassing_path = os.path.join(temp_lda_folder, "model-final.tassign")
            self.topic_corpus = []
            with smart_open(temp_tassing_path, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    self.topic_corpus.append([token.split(':')[1] for token in tokens])

            return phi

        elif method == "hmm":
            y = []
            for walk in self.corpus:
                seq = []

                for w in walk:
                    seq.append(int(w))

                y.append(seq)

            E = self.number_of_nodes
            K = self.number_of_communities
            L = self.params['walk_length']
            hmm_number_of_iters = self.params['hmm_number_of_iters']
            hmm_subset_size = self.params['hmm_subset_size']
            N = len(y)

            plates_multiplier = N / hmm_subset_size

            p0 = self.params['hmm_p0']  # a vector of size K
            t0 = self.params['hmm_t0']  # a vector of size K
            e0 = self.params['hmm_e0']

            p_param = p0*np.ones(K, dtype=np.float)
            p = bayes.Dirichlet(p_param, name='p')

            t_param = t0*np.ones(K, dtype=np.float)
            T = bayes.Dirichlet(t_param, plates=(K, ), name='T')

            e_param = e0*np.ones(E, dtype=np.float)
            E = bayes.Dirichlet(e_param, plates=(K, ), name='E')

            z = bayes.CategoricalMarkovChain(p, T, states=L, plates=(hmm_subset_size,),
                                             plates_multiplier=(plates_multiplier,), name='Z')
            x = bayes.Mixture(z, bayes.Categorical, E, name='X')

            p.initialize_from_random()
            T.initialize_from_random()
            E.initialize_from_random()


            Q = VB(x, z, E, T, p)
            """
            x.observe(y)

            Q.update(repeat=1000)
            """
            Q.ignore_bound_checks = True

            delay = 1
            forgetting_rate = 0.5

            for iter in range(hmm_number_of_iters):
                # Observe a random mini-batch
                subset = np.random.choice(a=N, size=hmm_subset_size)

                Q['X'].observe([y[inx] for inx in subset])
                # Learn intermediate variables
                Q.update('Z')
                #  Set step length
                step = (iter + delay) ** (-forgetting_rate)
                # Stochastic gradient for the global variables
                Q.gradient_step('p', 'T', 'E', scale=step)



            qp_temp = p.get_parameters()[0]
            qp = qp_temp / np.sum(qp_temp)
            qT_temp = np.asarray(T.get_parameters())[0]
            qT = np.asarray(qT_temp.T / np.sum(qT_temp, 1)).T
            qE_temp = np.asarray(E.get_parameters()[0])
            qE = np.asarray(qE_temp.T / np.sum(qE_temp, 1)).T

            likelihood = qE

            self.topic_corpus = []

            model = hmm.MultinomialHMM(n_components=self.number_of_communities, tol=0.001, n_iter=5000)
            model.startprob_ = qp
            model.emissionprob_ = qE
            model.transmat_ = qT

            initial_time = time.time()
            seq_for_hmmlearn = np.concatenate([np.asarray(seq).reshape(-1, 1).tolist() for seq in y])
            seq_lens = [self.params['walk_length'] for _ in range(N)]
            comm_conc_seq = model.predict(seq_for_hmmlearn, seq_lens)
            print("The hidden states are predicted in {} secs.".format(time.time() - initial_time))

            self.topic_corpus = []
            for i in range(N):
                self.topic_corpus.append([str(w) for w in comm_conc_seq[i*L:(i+1)*L]])

            return likelihood

        elif method == "bigclam":
            # Run AGM
            if not os.path.exists(BIGCLAM_PATH):
                raise ValueError("Invalid path of BigClam!")

            # If the temp folder for BigClam does not exits
            temp_bigclam_folder = os.path.join(self.temp_folder, "bigclam_temp")
            if not os.path.exists(temp_bigclam_folder):
                os.makedirs(temp_bigclam_folder)

            #g = nx.Graph()
            #g.add_edge(2, 3)
            #print("graph {}".format([g.copy()]))
            # Get all connected components
            cc_list = np.asarray(list(nx.connected_component_subgraphs(self.graph)))
            num_of_cc = cc_list.shape[0]
            print("graph {}".format([self.graph.copy()]))
            if num_of_cc == 1:
                cc_list = [self.graph.copy()]
            #print(cc_list)
            print("Number of connected components: {}".format(num_of_cc))
            cc_sizes = [cc.number_of_nodes() for cc in cc_list]
            # Sort the connected components
            cc_sizes_inx = np.argsort(cc_sizes)[::-1]
            cc_sizes = [cc_sizes[inx] for inx in cc_sizes_inx]
            cc_list = [cc_list[inx] for inx in cc_sizes_inx]
            # Find how many communities will be assigned for each connected component
            cum_sum_cc_sizes = np.cumsum(cc_sizes)
            # Find the community assignments of the set of the largest 'cc_inx_limit' connected components
            # in which the ratio of sizes of the smallest connected component and the size of the set is greater than
            # 1.5 times the number of communities which is desired to be assigned
            cc_inx_limit = 0
            for limit in range(num_of_cc):
                if cc_sizes[cc_inx_limit] / float(cum_sum_cc_sizes[cc_inx_limit]) >= (1.5 / self.number_of_communities):
                    cc_inx_limit += 1

            comm2node = []
            temp_bigclam_output = [[] for _ in range(cc_inx_limit)]
            temp_bigclam_edgelist = [[] for _ in range(cc_inx_limit)]
            temp_bigclam_labels = [[] for _ in range(cc_inx_limit)]
            assignment_sizes = np.zeros(shape=cc_inx_limit, dtype=np.int)
            correction_sizes = np.zeros(shape=cc_inx_limit, dtype=np.int)

            for cc_index in range(num_of_cc):
                current_ccg = cc_list[cc_index]

                if cc_index >= cc_inx_limit:
                    comm2node.append([v for v in current_ccg.nodes()])
                else:
                    assignment_sizes[cc_index] = int((float(cc_sizes[cc_index]) / cum_sum_cc_sizes[cc_inx_limit-1]) * self.params['number_of_communities'])
                    temp_bigclam_output[cc_index] = os.path.join(temp_bigclam_folder, "output{}".format(cc_index))
                    temp_bigclam_edgelist[cc_index] = os.path.join(temp_bigclam_folder, "temp{}.edgelist".format(cc_index))
                    temp_bigclam_labels[cc_index] = os.path.join(temp_bigclam_folder, "temp{}.labels".format(cc_index))

                    cc_graph_nodes = sorted([int(node) for node in current_ccg.nodes()])
                    with open(temp_bigclam_edgelist[cc_index], 'w') as f:
                        for node in cc_graph_nodes:
                            for nb in sorted([int(val) for val in nx.neighbors(current_ccg, str(node))]):
                                if int(node) < int(nb):
                                    f.write("{}\t{}\n".format(str(node), str(nb)))

                    with open(temp_bigclam_labels[cc_index], 'w') as f:
                        for node in cc_graph_nodes:
                            f.write("{}\t{}\n".format(str(node), str(node)))

                    cmd = "{} ".format(BIGCLAM_PATH)
                    cmd += "-o:{} ".format(temp_bigclam_output[cc_index])
                    cmd += "-i:{} ".format(temp_bigclam_edgelist[cc_index])
                    cmd += "-l:{} ".format(temp_bigclam_labels[cc_index])
                    cmd += "-nt:{} ".format(8)
                    cmd += "-c:{} ".format(assignment_sizes[cc_index])
                    os.system(cmd)

                    # Read the output file
                    with open(temp_bigclam_output[cc_index], 'r') as f:
                        for line in f.readlines():
                                comm2node.append(line.strip().split())
                                correction_sizes[cc_index] += 1

            total_num_of_assigned_communities = len(comm2node)
            phi = np.zeros(shape=(total_num_of_assigned_communities, self.number_of_nodes), dtype=np.float)

            self.number_of_communities = total_num_of_assigned_communities

            # Generate the phi matrix
            for k in range(total_num_of_assigned_communities):
                for node in comm2node[k]:
                    phi[k, int(node)] = 1.0

            # Be sure that every node is assigned to at least one community
            for node in range(self.number_of_nodes):
                # if a node is not assigned to any community
                if np.sum(phi[:, node]) == 0.0:
                    # Check the assignments of neighbors of the node
                    nb_comm_assign_counts = np.zeros(total_num_of_assigned_communities, dtype=np.float)
                    for nb in nx.neighbors(self.graph, str(node)):
                        nb_comm_assign_counts += phi[:, int(nb)]
                    # If the neighbors of the node is assigned to a community, assign it to the most frequent community
                    if nb_comm_assign_counts.sum() != 0.0:
                        assigned_comm_id = nb_comm_assign_counts.argmax()
                    # Otherwise assign it to a random community
                    else:
                        assigned_comm_id = np.random.choice(a=total_num_of_assigned_communities)
                    phi[assigned_comm_id, node] = 1.0

            # Normalize
            phi = np.divide(phi.T, np.sum(phi, 1)).T

            # Generate the topic corpus
            self.topic_corpus = []
            for walk in self.corpus:
                community_walk = []
                for w in walk:
                    # If the vertex has only one community assignment
                    if np.sum(phi[:, int(w)] > 0.0) == 1:
                        community_walk.append(str(np.where(phi[:, int(w)] > 0)[0][0]))
                    # otherwise, ...
                    else:
                        # if it is possible, assign it to the community which the previous node is assigned to
                        if len(community_walk) > 0 and phi[int(community_walk[-1]), int(w)] > 0.0:
                            community_walk.append(str(community_walk[-1]))
                        # if not, randomly choose a node
                        else:
                            chosen_comm = np.random.choice(a=phi.shape[0], p=phi[:, int(w)]/np.sum(phi[:, int(w)]))
                            community_walk.append(str(chosen_comm))

                self.topic_corpus.append(community_walk)

            print("---< Summary >---")
            print("+ The graph consists of {} connected component(s)".format(num_of_cc))
            for i in range(cc_inx_limit):
                print("+ The component of size {} is assigned to {}/{} communities".format(cc_sizes[i], correction_sizes[i], assignment_sizes[i]))
            print("+ Each of the remaining {} components is assigned to a unique label".format(num_of_cc-cc_inx_limit))
            print("+ The 'phi' matrix contains {} communities".format(phi.shape[0]))
            print("----o----")

            return phi

        elif method == "louvain":

            c = louvain.best_partition(self.graph)

            self.number_of_communities = len(set(c.values()))

            print("The number of detected communities: {}".format(self.number_of_communities))
            phi = np.zeros(shape=(self.number_of_communities, self.number_of_nodes), dtype=np.float)

            for node in self.graph.nodes():
                phi[int(c[node]), int(node)] = 1.0

            self.topic_corpus = []
            for walk in self.corpus:
                seq = [str(c[str(w)]) for w in walk]
                self.topic_corpus.append(seq)

            # Normalize
            phi = (phi.T / np.sum(phi, 1)).T

            return phi

        else:
            raise ValueError("Invalid community/topic detection method")
