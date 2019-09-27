import os
from os.path import basename, splitext, join
#from tne.tne2 import TNE
from utils.utils import *
import time
import networkx as nx
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from random_walks import *
from tne import *

# def process_old(args):
#
#     nx_graph_path = args.graph_path
#     outputs_folder = args.output_folder
#
#     # Set the parameters
#     params = {}
#     # The method name
#     params['random_walk'] = args.random_walk
#     # Common parameters
#     params['number_of_walks'] = args.n
#     params['walk_length'] = args.l
#     params['embedding_size'] = args.d
#     params['number_of_communities'] = args.k
#     params['community_detection_method'] = args.community_detection_method
#     # Parameters for Deepwalk
#     params['dw_alpha'] = args.dw_alpha
#     # Parameters for Node2vec
#     params['n2v_p'] = args.n2v_p
#     params['n2v_q'] = args.n2v_q
#     # Parameters for Skip-Gram
#     params['hs'] = args.hs
#     params['negative'] = args.negative
#     params['window_size'] = args.w
#     # Parameters for LDA
#     params['lda_alpha'] = float(50.0 / params['number_of_communities']) if args.lda_alpha == -1.0 else args.lda_alpha
#     params['lda_beta'] = args.lda_beta
#     params['lda_number_of_iters'] = args.lda_iter_num
#     params['concat_method'] = args.concat_method
#     # Parameters for HMM
#     params['hmm_p0'] = args.hmm_p0
#     params['hmm_t0'] = args.hmm_t0
#     params['hmm_e0'] = args.hmm_e0
#     params['hmm_number_of_iters'] = args.hmm_number_of_iters
#     params['hmm_subset_size'] = args.hmm_subset_size
#
#     graph_name = splitext(basename(nx_graph_path))[0]
#
#     file_desc = "{}_{}_n{}_l{}_w{}_k{}_{}".format(graph_name,
#                                                   params['community_detection_method'],
#                                                   params['number_of_walks'],
#                                                   params['walk_length'],
#                                                   params['window_size'],
#                                                   params['number_of_communities'],
#                                                   params['random_walk'])
#
#     if params['random_walk'] == 'node2vec':
#         file_desc += "_p={}_q={}".format(params['n2v_p'], params['n2v_q'])
#
#     # temp folder
#     temp_folder = os.path.join(outputs_folder, "temp", file_desc)
#     if not os.path.exists(temp_folder):
#         os.makedirs(temp_folder)
#
#     # output folder
#     embedding_folder = os.path.join(outputs_folder, "embeddings", file_desc)
#     if not os.path.exists(embedding_folder):
#         os.makedirs(embedding_folder)
#
#     # Determine the paths of embedding files
#     node_embedding_file = join(embedding_folder, "{}_node.embedding".format(file_desc))
#     community_embedding_file = join(embedding_folder, "{}_community.embedding".format(file_desc))
#
#     concatenated_embedding_file = dict()
#     concatenated_embedding_file['max'] = join(embedding_folder, "{}_final_max.embedding".format(file_desc))
#     concatenated_embedding_file['wmean'] = join(embedding_folder, "{}_final_wmean.embedding".format(file_desc))
#     concatenated_embedding_file['min'] = join(embedding_folder, "{}_final_min.embedding".format(file_desc))
#
#     # The path for the corpus
#     corpus_path_for_node = join(temp_folder, "{}_node.corpus".format(file_desc))
#
#     tne = TNE(nx_graph_path, temp_folder, params)
#     tne.perform_random_walks(output_node_corpus_file=corpus_path_for_node)
#     tne.preprocess_corpus(process="equalize")
#     phi = tne.generate_community_corpus(method=params['community_detection_method'])
#
#     tne.learn_node_embedding(output_node_embedding_file=node_embedding_file)
#     tne.learn_topic_embedding(output_topic_embedding_file=community_embedding_file)
#
#     # Compute the corresponding communities for each node
#     for embedding_strategy in ["max", "min", "wmean"]:
#         if params['concat_method'] == embedding_strategy or params['concat_method'] == "all":
#             initial_time = time.time()
#             # Concatenate the embeddings
#             concatenate_embeddings(node_embedding_file=node_embedding_file,
#                                    topic_embedding_file=community_embedding_file,
#                                    phi=phi,
#                                    method=embedding_strategy,
#                                    output_filename=concatenated_embedding_file[embedding_strategy])
#             print("-> The {} embeddings were generated and saved in {:.2f} secs | {}".format(
#                 embedding_strategy, (time.time() - initial_time), concatenated_embedding_file[embedding_strategy]))


def process(args):

    if args.task == "generate_walks":

        ext = os.path.splitext(args.graph)[1]
        if ext == '.gml':
            g = nx.read_gml(args.graph)
        elif ext == ".edgelist":
            g = nx.read_edgelist(args.graph_path, create_using=nx.Graph())
        else:
            raise ValueError("Invalid graph extension! {}".format(ext))

        opts = {'dw_alpha': args.dw_alpha, 'n2v_p': args.n2v_p, 'n2v_q': args.n2v_q}
        rw = RandomWalks(g=g, strategy_name=args.strategy, N=args.N, L=args.L, opts=opts)
        rw.write_walks(args.output)

    if args.task == "learn_embeddings":

        suffix = ""
        if args.suffix != "":
            suffix = "_" + args.suffix

        tne = TNE(K=args.K, suffix_for_files="deneme")
        tne.read_corpus_file(corpus_path=args.corpus)
        params = {'lda_alpha': args.lda_alpha, 'lda_beta': args.lda_beta, 'lda_iter_num': args.lda_iter_num}
        phi, theta, id2node = tne.extract_community_labels(community_detection_method=args.comm_method, params=params)
        tne.learn_node_embeddings(window_size=args.window_size, embedding_size=args.embedding_size,
                                  negative_samples_count=args.negative_samples, workers_count=args.workers)
        tne.write_node_embeddings("./node_prev_{}.embedding".format(suffix))
        tne.learn_community_embeddings(args.community_embed_size)
        tne.write_node_embeddings("./node{}.embedding".format(suffix))
        tne.write_community_embeddings("./community{}.embedding".format(suffix))
        tne.write_embeddings(embedding_file_path=args.output, phi=phi, theta=theta, id2node=id2node,
                             concatenate_method=args.concatenate_method)


def parse_arguments():
    parser = ArgumentParser(description="TNE: A Latent Model for Representation Learning on Networks",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(help='The task name', dest="task")

    generate_walks_parser = subparsers.add_parser('generate_walks')
    generate_walks_parser.add_argument('--graph', type=str, required=True,
                                        help='The path of graph file')
    generate_walks_parser.add_argument('--strategy', type=str, required=True,
                                       help='Random walk strategy', choices=['deepwalk', 'node2vec'])
    generate_walks_parser.add_argument('--N', type=int, required=True,
                                         help='The number of walks')
    generate_walks_parser.add_argument('--L', type=int, required=True,
                                         help='The length of walks')
    generate_walks_parser.add_argument('--output', type=str, required=True,
                                        help='The path of the corpus file')
    generate_walks_parser.add_argument('--dw_alpha', type=float, required=False, default=0.0,
                                       help='The value of the parameter alpha of deepwalk')
    generate_walks_parser.add_argument('--n2v_p', type=float, required=False, default=1.0,
                                       help='The value of the parameter p of node2vec')
    generate_walks_parser.add_argument('--n2v_q', type=float, required=False, default=1.0,
                                       help='The value of the parameter q of node2vec')

    learn_embeddings_parser = subparsers.add_parser('learn_embeddings')
    learn_embeddings_parser.add_argument('--corpus', type=str, required=True,
                                         help='The path of the corpus file')
    learn_embeddings_parser.add_argument('--comm_method', type=str, required=True, choices=['lda', 'hmm', 'bigclam', 'louvain'],
                                         help='The community detection method')
    learn_embeddings_parser.add_argument('--K', type=int, required=True,
                                         help='The number of latent communities')
    learn_embeddings_parser.add_argument('--embedding_size', type=int, required=False, default=128,
                                         help='The embedding size')
    learn_embeddings_parser.add_argument('--community_embed_size', type=int, required=False, default=128,
                                         help='Community embedding size')
    learn_embeddings_parser.add_argument('--window_size', type=int, required=False, default=10,
                                         help='The window size')
    learn_embeddings_parser.add_argument('--negative_samples', type=int, required=False, default=5,
                                         help='The number of negative samples')
    learn_embeddings_parser.add_argument('--workers', type=int, required=False, default=1,
                                         help='The number of workers')
    learn_embeddings_parser.add_argument('--output', type=str, required=True,
                                         help='The path of the embedding file')
    learn_embeddings_parser.add_argument('--lda_alpha', type=float, required=False, default=0.1,
                                         help='The value of the parameter alpha of LDA')
    learn_embeddings_parser.add_argument('--lda_beta', type=float, required=False, default=0.1,
                                         help='The value of the parameter beta of LDA')
    learn_embeddings_parser.add_argument('--lda_iter_num', type=int, required=False, default=1000,
                                         help='The number of iterations for LDA algorithm, GibssLDA++')
    learn_embeddings_parser.add_argument('--suffix', type=str, required=False, default="",
                                         help='The suffix for file names')
    learn_embeddings_parser.add_argument('--concatenate_method', type=str, required=False, default="max",
                                         help='The suffix for file names')

    '''
    parser.add_argument('--graph_path', type=str, required=True,
                        help='The path for networkx graph')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='The path of the output folder')
    parser.add_argument('--random_walk', choices=['deepwalk', 'node2vec'], required=True,
                        help='The name of the method used for performing random walks')
    parser.add_argument('--n', type=int, required=True,
                        help='The number of walks')
    parser.add_argument('--l', type=int, required=True,
                        help='The length of each walk')
    parser.add_argument('--d', type=int, default=128,
                        help='The size of the embedding vector')
    parser.add_argument('--k', type=int, required=True,
                        help='The number of communities')
    parser.add_argument('--community_detection_method', choices=['lda', 'hmm', 'bigclam', 'louvain'],
                        required=True, help="The community detection method")
    parser.add_argument('--dw_alpha', type=float, default=0.0,
                        help='The parameter for Deepwalk')
    parser.add_argument('--n2v_p', type=float, default=1.0,
                        help='The parameter for node2vec')
    parser.add_argument('--n2v_q', type=float, default=1.0,
                        help='The parameter for node2vec')
    parser.add_argument('--w', type=int, default=10,
                        help='The window size')
    parser.add_argument('--hs', type=int, default=0,
                        help='1 for the hierachical softmax, otherwise 0 for negative sampling')
    parser.add_argument('--negative', type=int, default=5,
                        help='It specifies how many noise words are used')
    parser.add_argument('--lda_alpha', type=float, default=-1.0,
                        help='A hyperparameter of LDA')
    parser.add_argument('--lda_beta', type=float, default=0.1,
                        help='A hyperparameter of LDA')
    parser.add_argument('--lda_iter_num', type=int, default=1000,
                        help='The number of iterations for GibbsLDA++')
    parser.add_argument('--hmm_p0', type=float, default=0.1,
                        help='p0 for hmm')
    parser.add_argument('--hmm_t0', type=float, default=0.2,
                        help='t0 for hmm')
    parser.add_argument('--hmm_e0', type=float, default=0.3,
                        help='e0 for hmm')
    parser.add_argument('--hmm_number_of_iters', type=int, default=2000,
                        help='The number of iterations for HMM')
    parser.add_argument('--hmm_subset_size', type=int, default=100,
                        help = 'The subset size for HMM')
    parser.add_argument('--concat_method', type=str, default='all',
                        help='Specifies the method for concatenating node and community embeddings')
    '''
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    process(args)

