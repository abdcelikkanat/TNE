import os
import time
import numpy as np
from utils.utils import CombineSentences
from ext.gensim_wrapper.models.word2vec import Word2VecWrapper
from six import iteritems

_lda_path = os.path.join(os.path.dirname(__file__), "../ext/gibbslda/lda")
_temp_folder_path = os.path.join(os.path.dirname(__file__), "../temp")
_community_detection_method_names = ['lda']


class TNE:

    K = None  # The number of latent communities
    walks = None  # The list of walk lists
    community_walks = None
    suffix_for_files = ""
    model = None

    def __init__(self, K, walks=None, suffix_for_files=None):

        if isinstance(walks, list):
            self.walks = walks

        if isinstance(K, int):
            self.K = K

        if isinstance(suffix_for_files, str):
            self.suffix_for_files = suffix_for_files

        # Create a folder for temporary files
        self._create_temp_folder(_temp_folder_path)

    def read_corpus_file(self, corpus_path):

        self.walks = []
        with open(corpus_path, 'r') as f:
            for line in f.readlines():
                walk = [w for w in line.strip().split()]
                self.walks.append(walk)

    def extract_community_labels(self, community_detection_method, params):

        if community_detection_method not in _community_detection_method_names:
            raise ValueError("Invalid community detection method name: {}".format(community_detection_method))

        detect_communities = getattr(self, "_" + community_detection_method)
        return detect_communities(params)

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
            cmd = "{} -est ".format(_lda_path)
            cmd += "-alpha {} ".format(params['lda_alpha'])
            cmd += "-beta {} ".format(params['lda_beta'])
            cmd += "-ntopics {} ".format(self.K)
            cmd += "-niters {} ".format(params['lda_iter_num'])
            cmd += "-savestep {} ".format(params['lda_iter_num'] + 1)
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

        def __read_phi_file(file_path, K, N):

            phi = np.zeros(shape=(K, N), dtype=np.float)
            with open(file_path, 'r') as f:
                for community, line in enumerate(f.readlines()):
                    for nodeId, value in enumerate(line.strip().split()):
                        phi[community, nodeId] = float(value)

            return phi

        def __read_tassing_file(file_path):

            community_walks = []
            with open(file_path, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    community_walks.append([int(token.split(':')[1]) for token in tokens])

            return community_walks

        # Set the corpus fie path
        lda_node_corpus_file = os.path.join(_temp_folder_path, self.suffix_for_files, "lda_node.corpus")
        # Set the wordmap file path
        wordmap_file_path = os.path.join(_temp_folder_path, self.suffix_for_files, "wordmap.txt")
        # Set the phi file path
        phi_file_path = os.path.join(_temp_folder_path, self.suffix_for_files, "model-final.phi")
        # Set the tassign file path
        tassign_file_path = os.path.join(_temp_folder_path, self.suffix_for_files, "model-final.tassign")

        __run_lda(lda_node_corpus_file=lda_node_corpus_file, params=params)
        N, id2node = __read_wordmap_file(file_path=wordmap_file_path)
        phi = __read_phi_file(file_path=phi_file_path, K=self.K, N=N)
        self.community_walks = __read_tassing_file(file_path=tassign_file_path)

        return phi, id2node

    def learn_node_embeddings(self, window_size=10, embedding_size=128, negative_samples_count=5, workers_count=1, hierarchical_softmax=0):

        initial_time = time.time()
        self.model = Word2VecWrapper(sentences=self.walks,
                                     size=embedding_size,
                                     window=window_size,
                                     sg=1, hs=hierarchical_softmax, negative=negative_samples_count,
                                     workers=workers_count,
                                     min_count=0)
        print("The node embeddings were learned in {:.2f} secs.".format(time.time() - initial_time))

    def write_node_embeddings(self, embedding_file_path):
        # Write node embeddings
        self.model.wv.save_word2vec_format(fname=embedding_file_path)

    def learn_community_embeddings(self):

        if len(self.walks) == 0:
            raise ValueError("There is no walk to learn embedding vectors!")

        if len(self.walks) != len(self.community_walks):
            raise ValueError("The number of community and node walks must be equal!")

        initial_time = time.time()
        # Construct the tuples (word, topic) with each word in the corpus and its corresponding topic assignment
        combined_walks = CombineSentences(node_walks=self.walks, community_walks=self.community_walks)
        # Extract the topic embeddings
        self.model.train_community(self.K, combined_walks)
        print("The community embeddings were generated in {:.2f} secs.".format(time.time() - initial_time))

    def write_community_embeddings(self, embedding_file_path):
        # Save the topic embeddings
        self.model.wv.save_word2vec_community_format(fname=embedding_file_path)

    def write_embeddings(self, embedding_file_path, phi, id2node):

        def _combine_embeddings(node_embs, community_embs, num_of_nodes, num_of_communities, embedding_size):
            node_comm_emb = np.zeros(shape=(num_of_nodes, num_of_communities, embedding_size), dtype=np.float)
            for

            return node2comm

        _combine_embeddings(node_embs=self.model.wv.syn0, self.model.wv.syn0_community)




        id2comm = np.argmax(phi, axis=0)
        node2comm = {id2node[nodeId]: id2comm[nodeId] for nodeId in range(len(id2comm))}
        with open(embedding_file_path, 'w') as f:
            f.write("{} {}\n".format(len(self.model.wv.vocab), self.model.wv.syn0.shape[1] + self.model.wv.syn0_community.shape[1]))
            # store in sorted order: most frequent words at the top
            for word, vocab in sorted(iteritems(self.model.wv.vocab), key=lambda item: -item[1].count):
                row = np.concatenate((self.model.wv.syn0[vocab.index], self.model.wv.syn0_community[node2comm[word]]))
                f.write("{} {}\n".format(word, ' '.join(str(val) for val in row)))

    def _create_temp_folder(self, folder_path):

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)