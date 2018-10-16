import numpy as np
from collections import OrderedDict
from gensim.utils import smart_open
from hmmlearn import hmm
from bayespy.inference import VB
import bayespy.nodes as bayes
import time
import os
from settings import *


def find_topics_for_nodes(phi_file, id2node, number_of_topics, type):

    number_of_nodes = len(id2node)

    # Phi is the node-topic distribution
    phi = np.zeros(shape=(number_of_topics, number_of_nodes), dtype=np.float)

    i = 0
    with smart_open(phi_file, 'r') as f:
        for vals in f.readlines():
            phi[i, :] = [float(v) for v in vals.strip().split()]
            i += 1

    if type == "max":
        arginx = np.argmax(phi, axis=0)
    if type == "min":
        arginx = np.argmin(phi, axis=0)

    node2topic = {}
    for i in range(arginx.shape[0]):
        node2topic.update({id2node[i]: arginx[i]})

    return node2topic


def generate_id2node(wordmap_file):
    id2node = {}
    with smart_open(wordmap_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            id, node = int(tokens[1]), tokens[0]
            id2node.update({id: node})

    return id2node


def convert_node2topic(tassign_file):
    with smart_open(tassign_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            yield [token.split(':')[1] for token in tokens]


def concatenate_embeddings(node_embedding_file, topic_embedding_file, phi, output_filename, method):

    # Set the number of topics and the number of nodes
    number_of_topics, number_of_nodes = phi.shape

    # Read the node embeddings
    node_embeddings = OrderedDict()  # OrderedDict is used to keep the order of nodes in the embedding vector
    with smart_open(node_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            node_embeddings[tokens[0]] = [val for val in tokens[1:]]

    # Read the topic embeddings
    topic_embeddings = {}
    with smart_open(topic_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            topic_embeddings[tokens[0]] = [val for val in tokens[1:]]

    node_embedding_len = len(list(node_embeddings.values())[0])
    topic_embedding_len = len(list(topic_embeddings.values())[0])

    # Concatenate the embeddings
    concatenated_embeddings = {}

    if method is "max" or method is "min":
        # Find the corresponding topics for each node
        node2topic = get_node2comm_assignments(phi, method=method)

        for node in node_embeddings:
            concatenated_embeddings[node] = node_embeddings[node] + topic_embeddings[str(node2topic[node])]

    if method is "wmean":
        topic_prior = np.ones(number_of_topics, dtype=np.float) * (1.0/float(number_of_topics))

        posterior = np.multiply(phi.T, topic_prior)
        posterior = np.divide(posterior.T, np.sum(posterior, 1)).T

        for node in node_embeddings:
            topic_vector = np.zeros(shape=topic_embedding_len, dtype=np.float)
            for topic in range(number_of_topics):
                topvect = np.asarray([float(value) for value in topic_embeddings[str(topic)]])
                topic_vector += posterior[int(node), topic] * topvect

            concatenated_embeddings[node] = node_embeddings[node] + ["{:.6f}".format(v) for v in topic_vector]

    concatenated_embedding_length = node_embedding_len + topic_embedding_len
    with smart_open(output_filename, 'w') as f:
        f.write(u"{} {}\n".format(number_of_nodes, concatenated_embedding_length))
        for node in node_embeddings:
            f.write(u"{} {}\n".format(str(node), " ".join(concatenated_embeddings[node])))


def concatenate_embeddings_wmean(node_embedding_file, topic_embedding_file, phi_file, id2node, output_filename):

    # Read the node embeddings
    node_embeddings = OrderedDict()
    with smart_open(node_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            node_embeddings.update({tokens[0]: [val for val in tokens[1:]]})

    # Read the topic embeddings
    topic_embeddings = {}
    topic_num = 0
    with smart_open(topic_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            topic_embeddings.update({tokens[0]: [float(val) for val in tokens[1:]]})
            topic_num += 1

    # Phi is the topic-node distribution
    phi = np.zeros(shape=(topic_num, len(node_embeddings.keys())), dtype=np.float)
    t = 0
    with smart_open(phi_file, 'r') as f:
        for vals in f.readlines():
            phi[t, :] = [float(v) for v in vals.strip().split()]
            t += 1

    # Concatenate the embeddings
    concatenated_embeddings = {}
    number_of_nodes = len(node_embeddings.keys())
    d = len(topic_embeddings['0'])
    for idx in range(number_of_nodes):
        wmean_topic_vec = np.zeros(shape=d, dtype=np.float)
        for t in range(topic_num):
            wmean_topic_vec += np.multiply(topic_embeddings[str(t)], phi[t, idx])

        concatenated_embeddings.update({id2node[idx]: node_embeddings[id2node[idx]] + wmean_topic_vec.tolist()})

    concatenated_embedding_length = len(list(concatenated_embeddings.values())[0])
    with smart_open(output_filename, 'w') as f:
        f.write(u"{} {}\n".format(number_of_nodes, concatenated_embedding_length))
        for node in node_embeddings:
            f.write(u"{} {}\n".format(node, " ".join(str(v) for v in concatenated_embeddings[node])))


def getCommProb(tassing_file_path, lda_num_of_communities):
    # it is assumed that probability of a walk p(w) or probability of a document p(d) is uniform.

    community2counts = np.asarray([0.0 for _ in range(lda_num_of_communities)])

    with smart_open(tassing_file_path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            for token in tokens:
                community_label = token.split(':')[1]
                community2counts[int(community_label)] += 1.0

    return community2counts / np.sum(community2counts)


def getPhi(phi_file_path):
    phi = []  # num_of_communities X num_of_nodes

    with smart_open(phi_file_path, 'r') as f:
        for values in f.readlines():
            phi.append([float(v) for v in values.strip().split()])

    return np.asarray(phi)


def prob_of_vertex_given_community(tassing_file_path, phi_file_path, lda_num_of_communities):

    community_prob = getCommProb(tassing_file_path, lda_num_of_communities)
    phi = getPhi(phi_file_path)

    unnormalized = np.multiply(phi.T, community_prob)
    prob = np.divide(unnormalized, np.sum(unnormalized, 0))
    return prob  # num_of_nodes X num_of_communities


def concatenate_embeddings_wmean2(node_embedding_file, topic_embedding_file, tassing_file, phi_file,
                                  lda_num_of_communities, id2node, output_filename):

    # Read the node embeddings
    node_embeddings = OrderedDict()
    with smart_open(node_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            node_embeddings.update({tokens[0]: [val for val in tokens[1:]]})

    # Read the topic embeddings
    topic_embeddings = {}
    with smart_open(topic_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            topic_embeddings.update({tokens[0]: [float(val) for val in tokens[1:]]})

    # Get the probability of vertex given community
    prob = prob_of_vertex_given_community(tassing_file_path=tassing_file, phi_file_path=phi_file,
                                          lda_num_of_communities=lda_num_of_communities)

    # Concatenate the embeddings
    concatenated_embeddings = {}
    number_of_nodes = len(node_embeddings.keys())
    topic_emb_dim = len(topic_embeddings['0'])
    for idx in range(number_of_nodes):
        wmean_topic_vec = np.zeros(shape=topic_emb_dim, dtype=np.float)
        for t in range(lda_num_of_communities):
            wmean_topic_vec += np.multiply(topic_embeddings[str(t)], prob[idx, t])

        concatenated_embeddings.update({id2node[idx]: node_embeddings[id2node[idx]] + wmean_topic_vec.tolist()})

    # Write the concatenated embeddings to a file
    concatenated_embedding_length = len(list(concatenated_embeddings.values())[0])
    with smart_open(output_filename, 'w') as f:
        f.write(u"{} {}\n".format(number_of_nodes, concatenated_embedding_length))
        for node in node_embeddings:
            f.write(u"{} {}\n".format(node, " ".join(str(v) for v in concatenated_embeddings[node])))


def get_node2comm_assignments(phi, method):

    if method == "max":
        arginx = np.argmax(phi, axis=0)
    if method == "min":
        arginx = np.argmin(phi, axis=0)

    node2comm = {}
    for node in range(phi.shape[1]):
        node2comm[str(node)] = arginx[node]

    return node2comm



def get_node_distr_over_comm(g, walks, method=None, params={}):

    if method == "HMM_param":

        seqs = []
        lens = []
        for walk in walks:
            s = [[int(w)] for w in walk]
            seqs.extend(s)
            lens.append(len(s))

        model = hmm.MultinomialHMM(n_components=params['number_of_topics'], tol=0.001, n_iter=5000)
        model.fit(seqs, lens)

        #posteriors = model.predict_proba(np.asarray([[i] for i in range(self.g.number_of_nodes())]))
        #comms = np.argmax(posteriors, 1)

        likelihood = model.emissionprob_

        """
        comms = np.argmax(likelihood, 0)

        node2comm = {}
        for id in range(len(comms)):
            node2comm[str(id)] = comms[id]

        return node2comm
        """


    elif method == "Nonparam_HMM":

        seqs = []
        lens = []
        for walk in walks:
            s = [int(w) for w in walk]
            seqs.append(s)
            lens.append(len(s))

        seqs = np.vstack(seqs)

        K = params['number_of_topics']  # the number of hidden states
        O = g.number_of_nodes()  # the size of observation set
        L = len(seqs[0])  # the length of each sequence
        N = len(seqs)  # the number of sequences

        p0 = params['prior_p0']  # a vector of size K
        t0 = params['prior_t0']  # a vector of size K
        e0 = params['prior_e0']  # a vector of size K

        p = bayes.Dirichlet(p0*np.ones(K), name='p')

        T = bayes.Dirichlet(t0*np.ones(K), plates=(K,), name='T')

        E = bayes.Dirichlet(e0*np.ones(O), plates=(K,), name='E')

        Z = bayes.CategoricalMarkovChain(p, T, states=L, name='Z', plates=(N, ))

        # Emission/observation distribution
        X = bayes.Mixture(Z, bayes.Categorical, E, name='X')

        p.initialize_from_random()
        T.initialize_from_random()
        E.initialize_from_random()

        Q = VB(X, Z, p, T, E)

        Q['X'].observe(seqs)
        Q.update(repeat=1000)

        likelihood = Q['E'].random()
        """
        comms = np.argmax(likelihood, 0)

        node2comm = {}
        for id in range(len(comms)):
            node2comm[str(id)] = comms[id]

        return node2comm
        """

        return likelihood

    elif method == "LDA":

        # Run GibbsLDA++
        if not os.path.exists(GIBBSLDA_PATH):
            raise ValueError("Invalid path of GibbsLDA++!")

        temp_lda_folder = os.path.join(TEMP_FOLDER, "lda_temp")
        if not os.path.exists(temp_lda_folder):
            os.makedirs(temp_lda_folder)

        temp_dfile_path = os.path.join(temp_lda_folder, "gibblda_temp.dfile")
        # Save the walks into the dfile
        n = len(walks)
        with open(temp_dfile_path, 'w') as f:
            f.write("{}\n".format(n))
            for walk in walks:
                f.write("{}\n".format(" ".join(str(w) for w in walk)))

        initial_time = time.time()
        cmd = "{} -est ".format(GIBBSLDA_PATH)
        cmd += "-alpha {} ".format(params['lda_alpha'])
        cmd += "-beta {} ".format(params['lda_beta'])
        cmd += "-ntopics {} ".format(params['number_of_topics'])
        cmd += "-niters {} ".format(params['lda_number_of_iters'])
        cmd += "-savestep {} ".format(params['lda_number_of_iters'] + 1)
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
        num_of_nodes = len(id2node)
        phi = np.zeros(shape=(params['number_of_topics'], num_of_nodes), dtype=np.float)
        temp_phi_path = os.path.join(temp_lda_folder, "model-final.phi")
        with open(temp_phi_path, 'r') as f:
            for comm, line in enumerate(f.readlines()):
                for id, value in enumerate(line.strip().split()):
                    phi[comm, int(id2node[id])] = value

        # Read the tassign file, generate topic corpus
        temp_tassing_path = os.path.join(temp_lda_folder, "model-final.tassign")
        comm_corpus = []
        with smart_open(temp_tassing_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                comm_corpus.append([token.split(':')[1] for token in tokens])

        """
        max_topics = np.argmax(phi, axis=0)

        node2comm = {}
        for nodeId in id2node:
            node2comm[id2node[nodeId]] = max_topics[int(nodeId)]

        return node2comm
        """

        return phi, comm_corpus
    else:
        raise ValueError("Wrong parameter name!")


class CombineSentences(object):

    def __init__(self, node_corpus, comm_corpus):
        self.node_corpus = node_corpus
        self.comm_corpus = comm_corpus

        assert len(node_corpus) == len(comm_corpus), "Node and topic corpus sizes must be equal!"

    def __iter__(self):
        for node_walk, comm_walk in zip(self.node_corpus, self.comm_corpus):
            yield [(v, int(t)) for (v, t) in zip(node_walk, comm_walk)]