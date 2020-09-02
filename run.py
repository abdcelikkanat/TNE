import networkx as nx
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils.tools import read_corpus_file
from tne.tne import TNE
from consts import *


def process(args):

    # Set all parameters #
    params = {}
    params['comm_detection_method'] = args.comm_method
    params['number_of_comms'] = args.K
    # Common parameters
    params['node_embedding_size'] = args.node_emb_size
    params['comm_embedding_size'] = args.comm_emb_size
    # Parameters for SkipGram
    params['window_size'] = args.window_size
    params['num_of_workers'] = args.workers
    # Parameters for LDA
    params['lda_alpha'] = 50 / float(params['number_of_comms']) if args.lda_alpha == 0 else args.lda_alpha  # Default is 50 / K
    params['lda_beta'] = args.lda_beta  # Default is 0.1
    params['lda_number_of_iters'] = args.lda_iter_num  # Default is 1000
    # Parameters for BayesianHMM
    params['bayesianhmm_number_of_steps'] = args.hmm_steps
    # Set the graph path, it might be required by some methods.
    params['graph_path'] = args.graph_path

    # Read the walks
    walks = read_corpus_file(args.corpus)
    # Call the main class
    tne = TNE(walks=walks, params=params, suffix=args.suffix)
    # Save the embedding file
    #tne.write_embeddings(embedding_file_path=args.emb, concatenate_method="average")
    tne.write_node_embeddings(file_path=args.emb)

def parse_arguments():
    parser = ArgumentParser(description="TNE: Topical Node Embeddings",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--corpus', type=str, required=True, help='The path of corpus file.')

    parser.add_argument('--emb', type=str, required=True, help='The path of embedding file.')

    parser.add_argument('--comm_method', type=str, required=True, choices=COMMMUNITY_DETECTION_METHODS,
                                         help='The community detection method.')
    parser.add_argument('--K', type=int, required=False, default=100,
                                         help='The number of latent communities.')
    parser.add_argument('--graph_path', type=str, required=False,
                                        help='The path for the graph in gml format.')
    parser.add_argument('--node_emb_size', type=int, required=False, default=96,
                                         help='The embedding size.')
    parser.add_argument('--comm_emb_size', type=int, required=False, default=32,
                                         help='Community embedding size.')
    parser.add_argument('--window_size', type=int, required=False, default=10,
                                         help='The window size.')
    parser.add_argument('--negative_samples', type=int, required=False, default=5,
                                         help='The number of negative samples.')
    parser.add_argument('--workers', type=int, required=False, default=1,
                                         help='The number of workers.')
    parser.add_argument('--lda_alpha', type=float, required=False, default=0,
                                         help='The value of the parameter alpha of LDA.')
    parser.add_argument('--lda_beta', type=float, required=False, default=0.1,
                                         help='The value of the parameter beta of LDA.')
    parser.add_argument('--lda_iter_num', type=int, required=False, default=1000,
                                         help='The number of iterations for LDA algorithm, GibssLDA++.')
    parser.add_argument('--hmm_steps', type=int, required=False, default=20,
                        help='The number of steps for Bayesian HMM model.')
    parser.add_argument('--suffix', type=str, required=False, default="",
                                         help='The suffix for file names.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    process(args)

