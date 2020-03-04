import os
import sys
sys.path.insert(0, "./")
sys.path.insert(1, "../")
sys.path.insert(2, "./examples/")
from consts import *
from utils.tools import read_corpus_file
from tne.tne import TNE


# Set all parameters #
params = {}
params['comm_detection_method'] = "bigclam"
params['number_of_comms'] = 10
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


corpus_paths = [os.path.join(BASE_FOLDER, "examples", "corpus", "karate.corpus"), ]
dataset_paths = [os.path.join(BASE_FOLDER, "examples", "datasets", "karate.gml"), ]


for cp, dp in zip(corpus_paths, dataset_paths):
    corpus_path = cp
    dataset_path = dp

    # Get the filename without the extension
    corpus_filename = os.path.splitext(os.path.basename(corpus_path))[0]
    # Set the network path
    params['graph_path'] = dataset_path
    suffix = "{}_{}_k={}_nodeEmbSize={}_commEmbSize={}".format(corpus_filename, params['comm_detection_method'],
                                                               params['number_of_comms'], params['node_embedding_size'],
                                                               params['comm_embedding_size'])
    # Read the walks
    walks = read_corpus_file(corpus_path)
    # Call the main class
    tne = TNE(walks=walks, params=params, suffix=suffix)
    # Define the path for embedding files
    emb_folder = os.path.join(BASE_FOLDER, "examples", "embeddings", suffix)
    os.makedirs(emb_folder, exist_ok=True)
    # Save all embeddings
    tne.write_node_embeddings(os.path.join(emb_folder, "node.embedding"))
    tne.write_community_embeddings(os.path.join(emb_folder, "community.embedding"))
    tne.write_embeddings(embedding_file_path=os.path.join(emb_folder, "min.embedding"), concatenate_method="min")
    tne.write_embeddings(embedding_file_path=os.path.join(emb_folder, "average.embedding"), concatenate_method="average")
    tne.write_embeddings(embedding_file_path=os.path.join(emb_folder, "max.embedding"), concatenate_method="max")
