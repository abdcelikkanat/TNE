import sys
sys.path.append('../')
sys.path.append('./TNE')
sys.path.append('../../')

from os.path import basename, splitext, join
from utils.utils import *
from tne.tne import TNE
import time
from settings import *


dataset_folder = os.path.join(BASE_FOLDER, '../datasets')
dataset_file = "Mich67_updated.gml" #"political_blog.gml" #"citeseer_undirected_gcc.gml" #"Amherst41_updated.gml"


# Set all parameters #
params = {}
params['comm_detection_method'] = "hmm"
params['random_walk'] = "node2vec"
# Common parameters
params['number_of_walks'] = 80
params['walk_length'] = 10
params['window_size'] = 10
params['embedding_size'] = 128
params['number_of_communities'] = 30
# Parameters for LDA
params['lda_number_of_iters'] = 1000
params['lda_alpha'] = 0.2
params['lda_beta'] = 0.1
# Parameters for Deepwalk
params['dw_alpha'] = 0
# Parameters for Node2vec
params['n2v_p'] = 4.0
params['n2v_q'] = 1.0
# Parameters for SkipGram
params['hs'] = 0
params['negative'] = 5
# Parameters for hmm
params['hmm_p0'] = 0.3
params['hmm_t0'] = 0.2
params['hmm_e0'] = 0.1
params['hmm_number_of_iters'] = 2000
params['hmm_subset_size'] = 100

# Define the file paths
nx_graph_path = os.path.join(dataset_folder, dataset_file)

file_desc = "{}_{}_n{}_l{}_w{}_k{}_{}".format(splitext(basename(dataset_file))[0],
                                              params['comm_detection_method'],
                                              params['number_of_walks'],
                                              params['walk_length'],
                                              params['window_size'],
                                              params['number_of_communities'],
                                              params['random_walk'])

if params['random_walk'] == 'node2vec':
    file_desc += "_p={}_q={}".format(params['n2v_p'], params['n2v_q'])


# temp folder
TEMP_FOLDER = os.path.join(OUTPUT_FOLDER, "temp", file_desc)
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# output folder
EMBEDDING_FOLDER = os.path.join(OUTPUT_FOLDER, "embeddings", file_desc)
if not os.path.exists(EMBEDDING_FOLDER):
    os.makedirs(EMBEDDING_FOLDER)


# Determine the paths of embedding files
node_embedding_file = join(EMBEDDING_FOLDER, "{}_node.embedding".format(file_desc))
community_embedding_file = join(EMBEDDING_FOLDER, "{}_community.embedding".format(file_desc))

concatenated_embedding_file = dict()
concatenated_embedding_file['max'] = join(EMBEDDING_FOLDER, "{}_final_max.embedding".format(file_desc))
concatenated_embedding_file['wmean'] = join(EMBEDDING_FOLDER, "{}_final_wmean.embedding".format(file_desc))
concatenated_embedding_file['min'] = join(EMBEDDING_FOLDER, "{}_final_min.embedding".format(file_desc))

# The path for the corpus
corpus_path_for_node = join(TEMP_FOLDER, "{}_node_corpus.corpus".format(file_desc))

tne = TNE(nx_graph_path, TEMP_FOLDER, params)
tne.perform_random_walks(output_node_corpus_file=corpus_path_for_node)
tne.preprocess_corpus(process="equalize")
phi = tne.generate_community_corpus(method=params['comm_detection_method'])

tne.learn_node_embedding(output_node_embedding_file=node_embedding_file)
tne.learn_topic_embedding(output_topic_embedding_file=community_embedding_file)


# Compute the corresponding topics for each node
for embedding_strategy in ["max", "min", "wmean"]:
    initial_time = time.time()
    # Concatenate the embeddings
    concatenate_embeddings(node_embedding_file=node_embedding_file,
                           topic_embedding_file=community_embedding_file,
                           phi=phi,
                           method=embedding_strategy,
                           output_filename=concatenated_embedding_file[embedding_strategy])
    print("-> The {} embeddings were generated and saved in {:.2f} secs | {}".format(
        embedding_strategy, (time.time()-initial_time), concatenated_embedding_file[embedding_strategy]))

