import os
import ipdb
import numpy as np
from Utils.network_reconstruction_tensor import Network_Reconstructor
from Utils.NNetwork import NNetwork, Wtd_NNetwork
import networkx as nx
import tracemalloc
import argparse


def read_networks_as_graph(path, if_DNA_string=True, node_range=None):
    if if_DNA_string:
        edgelist = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
        edgelist = edgelist[:, 1:]
        edgelist = edgelist.tolist()
        if 'sbm' not in path:
            ### Delete unwanted characters
            for edge in edgelist:
                edge[0] = edge[0][3:-1]
                edge[1] = edge[1][2:-3]
        else:
            # delete unwanted quatation from string.
            for edge in edgelist:
                edge[0] = edge[0].replace('"', '')
                edge[1] = edge[1].replace('"', '')
    else:
        edgelist = np.genfromtxt(path, delimiter=',', dtype=str)
        edgelist = edgelist.tolist()

    # G = nx.Graph(edgelist)
    G = NNetwork()
    G.add_edges(edges=edgelist)

    if node_range is not None:
        node_subset = []
        for i in node_range:
            node_subset.append('V' + str(i))
        G1 = nx.Graph(edgelist)
        G2 = G1.subgraph(nodes=node_subset)
        G = NNetwork()
        G.add_edges(edges=G2.edges)
        G.add_nodes(nodes=G1.nodes)

    print('number of nodes=', len(G.vertices))
    print('number of edges=', len(G.edges))
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default = 'Data/synthetic/', help='folder for data')
    parser.add_argument('--result_dir', default = 'synthetic_test/', help='folder for output results')
    args = parser.parse_args()

    # create the folder if not exist
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    ### set motif arm lengths
    k1 = 0
    # k2 = 20
    k2 = 10
    n_components = 25

    ### Create list of file names
    onlyfiles = ['sbm_edgelist.txt', 'sbm2_edgelist.txt']

    for DNA in onlyfiles:
        directory = args.data_dir
        path = os.path.join(directory, DNA)
        network_name = DNA.replace('.txt', '')
        network_name = network_name.replace('_edgelist', '')
        print('Currently learning dictionary patches from DNA: ' + str(network_name))

        ### Initialize reconstructor
        initialize_reconstructor = True

        num_MCMC = 1000
        sample_size = 1

        if initialize_reconstructor:
            ### read in networks
            G = read_networks_as_graph(path,
                                       if_DNA_string=True,
                                       node_range=None)  # np.arange(13513, 14267+1)


            reconstructor = Network_Reconstructor(G=G,  # networkx simple graph
                                                  n_components=n_components,  # num of dictionaries
                                                  MCMC_iterations=num_MCMC,  # MCMC steps (macro, grow with size of ntwk)
                                                  loc_avg_depth=1,  # keep it at 1
                                                  sample_size=sample_size,  # number of patches in a single batch
                                                  batch_size=20,  # number of columns used to train dictionary
                                                  # within a single batch step (keep it)
                                                  sub_iterations=100,  # number of iterations of the
                                                  # sub-batch learning (keep it)
                                                  k1=k1, k2=k2,  # left and right arm lengths
                                                  alpha=1,  # parameter for sparse coding, higher for stronger smoothing
                                                  is_glauber_dict=False,  # keep true to use Glauber chain for dict. learning
                                                  is_glauber_recons=False,  # keep false to use Pivot chain for recons.
                                                  ONMF_subsample=True)  # whether use i.i.d. subsampling for each batch

            reconstructor.result_dict.update({'Network name': network_name})
            reconstructor.result_dict.update({'# of nodes': len(G.vertices)})

       

        p2df = os.path.join(args.result_dir, f'df_{network_name}_{num_MCMC}_MCMC_pivot')
        p2feature = os.path.join(args.result_dir, f'feature_{network_name}_{num_MCMC}_MCMC_pivot')
        reconstructor.gen_MCMC_sample_in_df(p2df, p2feature)

