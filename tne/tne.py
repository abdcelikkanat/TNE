import os
import time
import numpy as np
import networkx as nx
from ext.gensim_wrapper.models.word2vec import Word2VecWrapper
from six import iteritems
from gensim.corpora.dictionary import Dictionary
#from gensim.models.ldamodel import LdaModel
#from gensim.models import HdpModel
import community as louvain
import bayesian_hmm
from consts import *
#from hmmlearn import hmm


#_lda_path = os.path.join(os.path.dirname(__file__), "../ext/gibbslda/lda")
_temp_folder_path = os.path.join(BASE_FOLDER, "temp")

class TNE:

    K = None  # The number of latent communities
    walks = None  # The list of walk lists
    community_walks = None
    suffix_for_files = ""
    model = None
    phi = None
    theta = None
    id2node = None
    params = None

    def __init__(self, walks=None, params=None, suffix=""):

        if isinstance(walks, list):
            self.walks = walks

        if isinstance(params, dict):
            self.params = params

        self.sg = 1

        self.hs = 0

        if 'comm_detection_method' not in self.params:
            raise ValueError("'comm_detection_method' parameter not exists")
        self.comm_detection_method = params['comm_detection_method']

        if 'number_of_comms' not in self.params:
            raise ValueError("'number_of_comms' parameter not exists")
        self.K = params['number_of_comms']

        if 'node_embedding_size' not in self.params:
            raise ValueError("'node_embedding_size' parameter not exists")
        self.node_embedding_size = params['node_embedding_size']

        if 'comm_embedding_size' not in self.params:
            raise ValueError("'comm_embedding_size' parameter not exists")
        self.comm_embedding_size = params['comm_embedding_size']

        if 'window_size' not in self.params:
            raise ValueError("'window_size' parameter not exists")
        self.window_size = params['window_size']

        if 'num_of_workers' not in self.params:
            raise ValueError("'num_of_workers' parameter not exists")
        self.workers = params['num_of_workers']

        # Create a folder for temporary files
        self.temp_folder_path = os.path.join(_temp_folder_path, suffix)
        self._create_temp_folder(self.temp_folder_path)

        # Learn Embeddings
        self.learn_embeddings()

    def read_corpus_file(self, corpus_path):

        self.walks = []
        with open(corpus_path, 'r') as f:
            for line in f.readlines():
                walk = [str(w) for w in line.strip().split()]
                self.walks.append(walk)

        print("--> 0. The corpus file has been read, which contains {} walks".format(len(self.walks)))

    def _create_temp_folder(self, folder_path):

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def learn_embeddings(self):

        # Learn node embeddings
        initial_time = time.time()
        self.model = Word2VecWrapper(sentences=self.walks,
                                     size=self.node_embedding_size,
                                     window=self.window_size,
                                     sg=self.sg, hs=self.hs,
                                     workers=self.workers,
                                     min_count=0,
                                     alpha=0.0025,
                                     min_alpha=0.00001,
                                     )
        self.model.build_vocab(sentences=self.walks)
        print("--> 1. Node embeddings have been learned in {} secs.".format(time.time() - initial_time))

        # Learn community labels
        initial_time = time.time()
        self.phi, theta, self.id2node = self.extract_community_labels()
        print("--> 2. The community labels have been learned in {} secs.".format(time.time() - initial_time))

        # Learn community embeddings
        initial_time = time.time()
        # Construct the tuples (word, community) with each node in the corpus and its corresponding community assignment
        combined_walks = CombineSentences(node_walks=self.walks, community_walks=self.community_walks)
        #self.model.train(sentences=self.walks, total_examples=self.model.corpus_count, epochs=self.model.iter)
        self.model.train_community(self.K, combined_walks, self.comm_embedding_size, total_examples=self.model.corpus_count, epochs=self.model.iter)
        print("--> 3. Community embeddings have been learned in {} secs.".format(time.time() - initial_time))

    def extract_community_labels(self):

        detect_communities_func = getattr(self, "_" + self.comm_detection_method)
        phi, theta, id2node = detect_communities_func(self.params)
        return phi, theta, id2node

    def write_node_embeddings(self, file_path):

        # Write node embeddings
        print("--> Embeddings are being written to the file: {}".format(file_path))
        self.model.wv.save_word2vec_format(fname=file_path)

    def get_node_embeddings(self):

        node_embeddings = {}
        for word, vocab in sorted(iteritems(self.model.wv.vocab), key=lambda item: -item[1].count):
            node_embeddings[word] = [value for value in self.model.wv.syn0[vocab.index]]

        return node_embeddings

    def write_community_embeddings(self, file_path):

        # Write community embeddings
        print("--> Embeddings are being written to the file: {}".format(file_path))
        self.model.wv.save_word2vec_community_format(file_path)

    def get_community_embeddings(self):

        community_embeddings = {}
        for i in range(self.K):
            row = self.model.wv.syn0_community[i]
            community_embeddings[i] = [value for value in row]

        return community_embeddings

    def _lda(self, params):

        def __run_lda(lda_node_corpus_file, params):

            # Firstly write the walks into a file with a suitable format for the lda program
            if not os.path.exists(os.path.dirname(lda_node_corpus_file)):
                os.makedirs(os.path.dirname(lda_node_corpus_file))
            with open(lda_node_corpus_file, 'w') as f:
                f.write(u"{}\n".format(len(self.walks)))
                for walk in self.walks:
                    f.write(u"{}\n".format(u" ".join(str(w) for w in walk)))

            # Run LDA
            initial_time = time.time()
            cmd = "{} -est ".format(GIBBSLDA_PATH)
            cmd += "-alpha {} ".format(params['lda_alpha'])
            cmd += "-beta {} ".format(params['lda_beta'])
            cmd += "-ntopics {} ".format(self.K)
            cmd += "-niters {} ".format(params['lda_number_of_iters'])
            cmd += "-savestep {} ".format(params['lda_number_of_iters'] + 1)
            cmd += "-dfile {} ".format(lda_node_corpus_file)
            os.system(cmd)
            print("-> The LDA algorithm run in {:.2f} secs".format(time.time() - initial_time))

        def __read_wordmap_file(file_path):

            id2node = {}
            with open(file_path, 'r') as f:
                number_of_nodes = int(f.readline().strip())
                for line in f.readlines():
                    tokens = line.strip().split()
                    id2node[int(tokens[1])] = tokens[0]

            return number_of_nodes, id2node

        def __read_phi_file(file_path, K, num_of_nodes):

            phi = np.zeros(shape=(K, num_of_nodes), dtype=np.float)
            with open(file_path, 'r') as f:
                for community, line in enumerate(f.readlines()):
                    for nodeId, value in enumerate(line.strip().split()):
                        phi[community, nodeId] = float(value)

            return phi

        def __read_theta_file(file_path, K, num_of_walks):

            theta = np.zeros(shape=(num_of_walks, K), dtype=np.float)
            with open(file_path, 'r') as f:
                for walkId, line in enumerate(f.readlines()):
                    theta[walkId, :] = [float(value) for value in line.strip().split()]

            return theta

        def __read_tassing_file(file_path):

            community_walks = []
            with open(file_path, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    community_walks.append([str(token.split(':')[1]) for token in tokens])

            return community_walks

        # Set the corpus fie path
        lda_node_corpus_file = os.path.join(self.temp_folder_path, self.suffix_for_files, "lda_node.corpus")
        # Set the wordmap file path
        wordmap_file_path = os.path.join(self.temp_folder_path, self.suffix_for_files, "wordmap.txt")
        # Set the phi file path
        phi_file_path = os.path.join(self.temp_folder_path, self.suffix_for_files, "model-final.phi")
        # Set the theta file path
        theta_file_path = os.path.join(self.temp_folder_path, self.suffix_for_files, "model-final.theta")
        # Set the tassign file path
        tassign_file_path = os.path.join(self.temp_folder_path, self.suffix_for_files, "model-final.tassign")

        __run_lda(lda_node_corpus_file=lda_node_corpus_file, params=params)
        num_of_nodes, id2node = __read_wordmap_file(file_path=wordmap_file_path)
        phi = __read_phi_file(file_path=phi_file_path, K=self.K, num_of_nodes=num_of_nodes)
        self.community_walks = __read_tassing_file(file_path=tassign_file_path)
        theta = __read_theta_file(theta_file_path, K=self.K, num_of_walks=len(self.walks))

        return phi, theta, id2node

    def _louvain(self, params):

        if 'graph_path' not in params:
            raise ValueError("For {} algorithm, the graph path is needed!".format(self.comm_detection_method))

        graph = nx.read_gml(params['graph_path'])

        partition = louvain.best_partition(graph)
        self.K = len(set(partition.values()))

        print("--> The {} algorithm was selected for detectin communities.".format(self.comm_detection_method))
        print("--> The number of communities detected is {}.".format(self.K))

        phi = np.zeros(shape=(self.K, graph.number_of_nodes()), dtype=np.float)
        for node in range(graph.number_of_nodes()):
            phi[int(partition[str(node)]), node] = 1.0
        phi = (phi.T / np.sum(phi, 1)).T

        theta = None

        id2node = {int(node): node for node in graph.nodes()}

        # Generate community walks
        self.community_walks = []
        for walk in self.walks:
            community_walk = [str(partition[node]) for node in walk]
            self.community_walks.append(community_walk)

        return phi, theta, id2node

    def _bigclam(self, params):

        def __read_bigclam_output(file_path, k=0):
            node2comm = {}
            with open(file_path, 'r') as fin:
                for line in fin.readlines():
                    tokens = line.strip().split()
                    for token in tokens:
                        if token in node2comm:
                            node2comm[token].append(k)
                        else:
                            node2comm[token] = [k]
                    k += 1

            return node2comm, k

        if 'graph_path' not in params:
            raise ValueError("For {} algorithm, the graph path is needed!".format(self.comm_detection_method))
        graph = nx.read_gml(params['graph_path'])
        #graph.remove_edges_from(nx.selfloop_edges(graph))

        temp_folder = os.path.join(self.temp_folder_path, self.suffix_for_files)
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        # Run BigClam algorithm over connected components
        self.K = 0
        node2comm = {}
        for gccInx, gcc_nodes in enumerate(nx.connected_components(graph)):
            # Get a connected component
            gcc = graph.subgraph(gcc_nodes).copy()
            # Set the edgelist and output files
            gcc_edgelist = os.path.join(self.temp_folder_path, self.suffix_for_files, "gcc{}.edgelist".format(gccInx))
            output_path = os.path.join(self.temp_folder_path, self.suffix_for_files, "gcc{}.bigclam".format(gccInx))
            # Write the edges to the file
            nx.write_edgelist(gcc, gcc_edgelist, data=False)
            # Run the algorithm
            cmd = "{} -o:{} -i:{} -nt:{} -c:-1".format(BIGCLAM_PATH, output_path, gcc_edgelist, str(4))
            os.system(cmd)
            # Read the algorithm output
            gcc_node2comm, self.K = __read_bigclam_output(output_path, k=self.K)
            node2comm.update(gcc_node2comm)

        # If there is a node which is not assigned to a community label, assign it to a label
        for node in graph.nodes():
            if node not in node2comm:
                node2comm[node] = [self.K]
                self.K += 1

        phi = np.zeros(shape=(self.K, graph.number_of_nodes()), dtype=np.float)
        for node in range(graph.number_of_nodes()):
            for k in node2comm[str(node)]:
                phi[k, node] = 1.0
        phi = (phi.T / np.sum(phi, 1)).T

        theta = None

        id2node = {int(node): node for node in graph.nodes()}

        # Generate community walks
        self.community_walks = []
        for walk in self.walks:
            community_walk = [np.random.choice(a=node2comm[walk[0]], size=1)[0]]
            for node in walk[1:]:
                if community_walk[-1] in node2comm[node]:
                    chosen_label = community_walk[-1]
                else:
                    chosen_label = np.random.choice(a=node2comm[node], size=1)[0]
                community_walk.append(chosen_label)

            self.community_walks.append(community_walk)

        return phi, theta, id2node

    def _bayesianhmm(self, params):

        if 'graph_path' not in params:
            raise ValueError("For {} algorithm, the graph path is needed!".format(self.comm_detection_method))
        graph = nx.read_gml(params['graph_path'])

        # initialise object with overestimate of true number of latent states
        hmm = bayesian_hmm.HDPHMM(self.walks, sticky=False)
        hmm.initialise()

        n = params['bayesianhmm_number_of_steps']
        results = hmm.mcmc(n=n, burn_in=n - 1, save_every=1, ncores=3, verbose=False)

        map_index = -1
        parameters_map = results['parameters'][map_index]
        commlabel2comm = {}
        comm = 0
        for commlabel in parameters_map['p_emission'].keys():
            if commlabel != 'None':
                commlabel2comm[commlabel] = comm
                comm += 1

        self.K = len(commlabel2comm.keys())

        parameters_map = results['parameters'][map_index]
        emission_prob = parameters_map['p_emission']

        phi = np.zeros(shape=(self.K, graph.number_of_nodes()), dtype=np.float)
        for node in range(graph.number_of_nodes()):
            for k in commlabel2comm.keys():
                phi[commlabel2comm[k], node] = emission_prob[k][str(node)]
        phi = (phi.T / np.sum(phi, 1)).T

        theta = None

        id2node = {int(node): node for node in graph.nodes()}

        chains = hmm.chains
        self.community_walks = []
        for i in range(len(self.walks)):
            community_walk = []
            for w in chains[i].latent_sequence:
                community_walk.append(commlabel2comm[w])
            self.community_walks.append(community_walk)

        return phi, theta, id2node

    '''
    def _hdp(self, params):

        corpus_dictionary = Dictionary(self.walks)
        walks_bow_repr = [corpus_dictionary.doc2bow(walk) for walk in self.walks]
        hdp = HdpModel(walks_bow_repr, corpus_dictionary, T=self.K)
        self.K = hdp.m_T
        lda = hdp.suggested_lda_model()
        #lda = LdaModel(walks_bow_repr, num_topics=self.K, minimum_phi_value=0.0, minimum_probability=0.0, update_every=0)

        #num_of_nodes = lda.num_terms

        topics = lda.state.get_lambda()
        phi = topics / topics.sum(axis=1)[:, None]  # self.K x num_of_nodes matrix

        self.community_walks = []
        for walkId, walk in enumerate(self.walks):
            t = dict(lda.get_document_topics(walks_bow_repr[walkId], minimum_probability=0.0, minimum_phi_value=0.0, per_word_topics=True)[1])
            topic_walk = [t[corpus_dictionary.token2id[w]][0] for w in walk]
            self.community_walks.append(topic_walk)

        id2node = lda.id2word
        theta = 0

        return phi, theta, id2node

    def _classicalhmm(self, params):

        if 'graph_path' not in params:
            raise ValueError("For {} algorithm, the graph path is needed!".format(self.comm_detection_method))
        graph = nx.read_gml(params['graph_path'])

        id2node = {int(node): node for node in graph.nodes()}

        num_of_walks = len(self.walks)
        walk_len = len(self.walks[0])

        initial_time = time.time()
        model = hmm.MultinomialHMM(n_components=self.K, tol=0.001, n_iter=5000)
        seq_for_hmmlearn = []
        for walk in self.walks:
            seq_for_hmmlearn.extend([int(w) for w in walk])
        seq_for_hmmlearn = np.asarray(seq_for_hmmlearn).reshape(-1, 1)
        #np.concatenate([np.asarray(seq).reshape(-1, 1).tolist() for seq in self.walks])
        model.fit(seq_for_hmmlearn)
        seq_lens = [walk_len for _ in range(num_of_walks)]
        comm_conc_seq = model.predict(seq_for_hmmlearn, seq_lens)
        print("The hidden states are predicted in {} secs.".format(time.time() - initial_time))

        self.community_walks = []
        for i in range(num_of_walks):
            self.community_walks.append([str(w) for w in comm_conc_seq[i * walk_len:(i + 1) * walk_len]])

        phi = model.emissionprob_

        theta = None

        return phi, theta, id2node
        
    def _ldagensim(self, params):

        common_dictionary = Dictionary(self.walks)
        common_corpus = [common_dictionary.doc2bow(walk) for walk in self.walks]
        lda = LdaModel(common_corpus, num_topics=self.K, minimum_phi_value=0.0, minimum_probability=0.0,
                       id2word=common_dictionary)

        num_of_nodes = lda.num_terms

        phi = np.zeros(shape=(self.K, num_of_nodes), dtype=np.float)
        for topicId in range(self.K):
            for id, value in lda.get_topic_terms(topicId, num_of_nodes):
                phi[topicId, id] = value
        
        word2id = {lda.id2word[id]: id for id in lda.id2word.keys()}

        self.community_walks = []
        for walkId, walk in enumerate(self.walks):
            bow = common_corpus[walkId]
            b = lda.get_document_topics(bow, minimum_probability=0.0, minimum_phi_value=0.0, per_word_topics=True)[1]
            b = dict(b)
            #topic_walk = [b[word2id[w]][0] for w in walk]
            topic_walk = [0 for w in walk]
            #print(topic_walk)
            #print(topic_walk)
            #topic_walk = [np.random.choice(self.K, 1)[0] for tx in t]
            #topic_walk = [5 for tx in t]
            self.community_walks.append(topic_walk)

        id2node = lda.id2word
        theta = 0

        return phi, theta, id2node
        
    '''

    def write_embeddings(self, embedding_file_path, concatenate_method):

        if self.phi is None or self.id2node is None:
            raise ValueError("An error has occured in learning community labels!")

        print("--> Embeddings are being written to the file: {}".format(embedding_file_path))

        if concatenate_method == "max":

            id2comm = np.argmax(self.phi, axis=0)
            node2comm = {self.id2node[nodeId]: id2comm[nodeId] for nodeId in range(len(id2comm))}
            with open(embedding_file_path, 'w') as f:
                f.write("{} {}\n".format(len(self.model.wv.vocab), self.model.wv.syn0.shape[1] + self.model.wv.syn0_community.shape[1]))
                # store in sorted order: most frequent words at the top
                for word, vocab in sorted(iteritems(self.model.wv.vocab), key=lambda item: -item[1].count):
                    row = np.concatenate((self.model.wv.syn0[vocab.index], self.model.wv.syn0_community[node2comm[word]]))
                    f.write("{} {}\n".format(word, ' '.join(str(val) for val in row)))

        elif concatenate_method == "sum":

            id2comm = np.argmax(self.phi, axis=0)
            node2comm = {self.id2node[nodeId]: id2comm[nodeId] for nodeId in range(len(id2comm))}
            with open(embedding_file_path, 'w') as f:
                f.write("{} {}\n".format(len(self.model.wv.vocab), self.model.wv.syn0.shape[1] + self.model.wv.syn0_community.shape[1]))
                # store in sorted order: most frequent words at the top
                for word, vocab in sorted(iteritems(self.model.wv.vocab), key=lambda item: -item[1].count):
                    row = 1.0*np.asarray(self.model.wv.syn0[vocab.index]) + 1.0*np.asarray(self.model.wv.syn0_community[node2comm[word]])
                    f.write("{} {}\n".format(word, ' '.join(str(val) for val in row)))

        elif concatenate_method == "average":

            phi = self.phi / np.sum(self.phi, 0)

            comm_embs = np.zeros(shape=(len(self.model.wv.vocab), self.model.wv.syn0_community.shape[1]), dtype=np.float)
            for nodeId in range(np.shape(phi)[1]):
                for comm in range(np.shape(phi)[0]):
                    comm_emb = np.asarray(self.model.wv.syn0_community[comm])
                    comm_embs[int(self.id2node[nodeId]), :] += phi[comm, nodeId] * comm_emb

            with open(embedding_file_path, 'w') as f:
                f.write("{} {}\n".format(len(self.model.wv.vocab), self.model.wv.syn0.shape[1] + self.model.wv.syn0_community.shape[1]))
                for word, vocab in sorted(iteritems(self.model.wv.vocab), key=lambda item: -item[1].count):
                    row = np.concatenate((self.model.wv.syn0[vocab.index], comm_embs[int(word), :]))
                    f.write("{} {}\n".format(word, ' '.join(str(val) for val in row)))

        elif concatenate_method == "min":

            id2comm = np.argmin(self.phi, axis=0)
            node2comm = {self.id2node[nodeId]: id2comm[nodeId] for nodeId in range(len(id2comm))}
            with open(embedding_file_path, 'w') as f:
                f.write("{} {}\n".format(len(self.model.wv.vocab),
                                         self.model.wv.syn0.shape[1] + self.model.wv.syn0_community.shape[1]))
                # store in sorted order: most frequent words at the top
                for word, vocab in sorted(iteritems(self.model.wv.vocab), key=lambda item: -item[1].count):
                    row = np.concatenate(
                        (self.model.wv.syn0[vocab.index], self.model.wv.syn0_community[node2comm[word]]))
                    f.write("{} {}\n".format(word, ' '.join(str(val) for val in row)))

        else:

            raise ValueError("Invalid Concatenation Method Name!")


class CombineSentences(object):

    def __init__(self, node_walks, community_walks):
        assert len(node_walks) == len(community_walks), "Node and community corpus sizes must be equal!"

        self.node_walks = node_walks
        self.community_walks = community_walks

    def __iter__(self):
        for node_walk, comm_walk in zip(self.node_walks, self.community_walks):
            yield [(v, int(t)) for (v, t) in zip(node_walk, comm_walk)]

