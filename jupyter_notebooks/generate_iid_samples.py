import ipdb
import numpy as np
from Utils.network_reconstruction_tensor import Network_Reconstructor
from Utils.NNetwork import NNetwork, Wtd_NNetwork
import networkx as nx
import tracemalloc


def read_networks_as_graph(path, if_DNA_string=True, node_range=None):
    if if_DNA_string:
        edgelist = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
        edgelist = edgelist[:, 1:]
        edgelist = edgelist.tolist()
        if 'ChIA_Drop' not in path:
            ### Delete unwanted characters
            for edge in edgelist:
                edge[0] = edge[0][3:-1]
                edge[1] = edge[1][2:-3]
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

def see_results(DNA):
    onlyfiles = ['edge_list_of_chr2R_GSM3347525NR.txt',
                 'edge_list_of_chr3L_GSM3347525NR.txt',
                 'edge_list_of_chr3R_GSM3347525NR.txt',
                 'edge_list_of_chr2L_GSM3347525NR.txt',
                 'edge_list_of_chr4_GSM3347525NR.txt']

    directory = "Network_dictionary/"
    path = directory + 'full_result_' + DNA + '.npy'
    full_results = np.load(path, allow_pickle=True).item()
    print('Reconstruction accuracy for DNA' + str(DNA) + ' = ' + '%f' % full_results.get('reconstruction accuracy=') )


def main():
    ### set motif arm lengths
    k1 = 0
    k2 = 20
    n_components = 25

    ### Create list of file names
    # onlyfiles.remove('desktop.ini')
    onlyfiles = ['edge_list_of_chr2R_GSM3347525NR.txt',
                 'edge_list_of_chr3L_GSM3347525NR.txt',
                 'edge_list_of_chr3R_GSM3347525NR.txt',
                 'edge_list_of_chr2L_GSM3347525NR.txt',
                 'edge_list_of_chr4_GSM3347525NR.txt']

    onlyfiles = ['edge_list_of_chr2R_GSM3347525NR.txt',
                 'edge_list_of_chr3L_GSM3347525NR.txt',
                 'edge_list_of_chr3R_GSM3347525NR.txt',
                 'edge_list_of_chr2L_GSM3347525NR.txt']

    # onlyfiles = ['edge_list_of_chr2L_GSM3347525NR.txt']

    # onlyfiles = ['edge_list_of_chr4_GSM3347525NR.txt']
    onlyfiles = [# 'edge_list_of_chr4_drosophila_ChIA_Drop_0.1_PASS.txt',
                 # 'edge_list_of_chrX_drosophila_ChIA_Drop_0.1_PASS.txt',
                 'edge_list_of_chr2L_drosophila_ChIA_Drop_0.1_PASS.txt',
                 'edge_list_of_chr3L_drosophila_ChIA_Drop_0.1_PASS.txt',
                 'edge_list_of_chr2R_drosophila_ChIA_Drop_0.1_PASS.txt',
                 'edge_list_of_chr3R_drosophila_ChIA_Drop_0.1_PASS.txt']

    # onlyfiles = ['edge_list_of_chr4_drosophila_ChIA_Drop_0.1_PASS.txt',]

    for DNA in onlyfiles:
        directory = "Data/DNA/"
        path = directory + DNA
        network_name = DNA.replace('.txt', '')
        network_name = network_name.replace('edge_list_of_', '')
        print('Currently learning dictionary patches from DNA: ' + str(network_name))

        ### Initialize reconstructor
        initialize_reconstructor = True

        num_MCMC = 20000

        if initialize_reconstructor:
            ### read in networks
            G = read_networks_as_graph(path,
                                       if_DNA_string=True,
                                       node_range=None)  # np.arange(13513, 14267+1)


            reconstructor = Network_Reconstructor(G=G,  # networkx simple graph
                                                  n_components=n_components,  # num of dictionaries
                                                  MCMC_iterations=num_MCMC,  # MCMC steps (macro, grow with size of ntwk)
                                                  loc_avg_depth=1,  # keep it at 1
                                                  sample_size=1000,  # number of patches in a single batch
                                                  batch_size=20,  # number of columns used to train dictionary
                                                  # within a single batch step (keep it)
                                                  sub_iterations=100,  # number of iterations of the
                                                  # sub-batch learning (keep it)
                                                  k1=k1, k2=k2,  # left and right arm lengths
                                                  alpha=1,  # parameter for sparse coding, higher for stronger smoothing
                                                  is_glauber_dict=True,  # keep true to use Glauber chain for dict. learning
                                                  is_glauber_recons=False,  # keep false to use Pivot chain for recons.
                                                  ONMF_subsample=True)  # whether use i.i.d. subsampling for each batch

            reconstructor.result_dict.update({'Network name': network_name})
            reconstructor.result_dict.update({'# of nodes': len(G.vertices)})

        p2df = f'/data/shared/jianhao/online_cvxNDL_data/new_data_with_node_id/df_{network_name}_{num_MCMC}'
        p2feature = f'/data/shared/jianhao/online_cvxNDL_data/new_data_with_node_id/feature_{network_name}_{num_MCMC}'
        reconstructor.gen_iid_samples_in_df(p2df, p2feature)
        # # if_learn_fresh = False
        # if_learn_fresh = True

        # if if_learn_fresh:
        #     reconstructor.train_dict()
        # else:
        #     reconstructor.result_dict = np.load('Network_dictionary/full_result_' + str(network_name) + '.npy',
        #                                         allow_pickle=True).item()
        #     reconstructor.W = reconstructor.result_dict.get('Dictionary learned')

        # np.save("Network_dictionary/full_result_" + str(network_name), reconstructor.result_dict)

        # if_save_fig = False

        # if if_save_fig:
        #     ### save dictionaytrain_dict figures
        #     title = str(k1 + k2 + 1) + "by" + str(
        #         k1 + k2 + 1) + "Network dictionary patches"
        #     save_filename = "dict_plot_" + "_" + str(k1) + str(k2) + '_' + str(n_components)
        #     # reconstructor.display_dict(title, save_filename)
        #     reconstructor.display_dict(title='Dictionary learned from DNA ' + str(network_name),
        #                                save_filename='DNA_dict' + '_' + str(network_name) + str(k2))

        # print('Finished dictionary learning from network ' + str(DNA))

        # if_recons = True

        # if if_recons:
        #     iter = np.floor(len(G.vertices) * np.log(len(G.vertices))/2)
        #     G_recons = reconstructor.reconstruct_network(recons_iter=iter,
        #                                                  if_construct_WtdNtwk=True,
        #                                                  if_save_history=True,
        #                                                  use_checkpoint_refreshing=True,
        #                                                  ckpt_epoch=10000)
        #     reconstructor.result_dict.update({'reconstruction iterations': iter})
        #     print(len(G_recons.vertices))
        #     path_recons = 'Network_dictionary/' + 'recons_25_self_' + str(k2) + '.npy'

        #     # G_recons = read_networks_as_graph(path, if_DNA_string=False)
        #     recons_accuracy = reconstructor.compute_recons_accuracy(G_recons)

        # np.save("Network_dictionary/full_result_" + str(network_name), reconstructor.result_dict)

        # ### Display final results
        # # see_results(network_name)


if __name__ == '__main__':
    main()
