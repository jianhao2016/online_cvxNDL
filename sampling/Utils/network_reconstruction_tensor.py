# from utils.onmf.onmf import Online_NMF
from Utils.ontf import Online_NTF
from Utils.NNetwork import NNetwork, Wtd_NNetwork
import numpy as np
import itertools
from time import time
from sklearn.decomposition import SparseCoder
import matplotlib.pyplot as plt
import networkx as nx
import os
import psutil
from time import sleep
import sys
import ipdb

DEBUG = False


class Network_Reconstructor():
    def __init__(self,
                 G,
                 n_components=100,
                 MCMC_iterations=500,
                 sub_iterations=100,
                 loc_avg_depth=1,
                 sample_size=1000,
                 batch_size=10,
                 k1=1,
                 k2=2,
                 patches_file='',
                 alpha=None,
                 is_glauber_dict=True,
                 is_glauber_recons=True,
                 ONMF_subsample=True,
                 if_wtd_network=False,
                 if_tensor_ntwk=False):
        '''
        batch_size = number of patches used for training dictionaries per ONMF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.G = G  ### Full netowrk -- could have positive or negagtive edge weights
        if if_tensor_ntwk:
            self.G.set_clrd_edges_signs()
            ### Each edge with weight w is assigned with tensor weight [+(w), -(w)] stored in the field colored_edge_weight

        self.n_components = n_components
        self.MCMC_iterations = MCMC_iterations
        self.sub_iterations = sub_iterations
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.loc_avg_depth = loc_avg_depth
        self.k1 = k1
        self.k2 = k2
        self.patches_file = patches_file
        self.if_tensor_ntwk = if_tensor_ntwk  # if True, input data is a 3d array
        self.W = np.zeros(shape=((k1 + k2 + 1) ** 2, n_components))
        if if_tensor_ntwk:
            self.W = np.zeros(shape=(G.color_dim*(k1 + k2 + 1) ** 2, n_components))

        self.code = np.zeros(shape=(n_components, sample_size))
        self.code_recons = np.zeros(shape=(n_components, sample_size))
        self.alpha = alpha
        self.is_glauber_dict = is_glauber_dict  ### if false, use pivot chain for dictionary learning
        self.is_glauber_recons = is_glauber_recons  ### if false, use pivot chain for reconstruction
        self.edges_deleted = []
        self.ONMF_subsample = ONMF_subsample
        self.result_dict = {}
        self.if_wtd_network = if_wtd_network

    def list_intersection(self, lst1, lst2):
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    def path_adj(self, k1, k2):
        # generates adjacency matrix for the path motif of k1 left nodes and k2 right nodes
        if k1 == 0 or k2 == 0:
            k3 = max(k1, k2)
            A = np.eye(k3 + 1, k=1, dtype=int)
        else:
            A = np.eye(k1 + k2 + 1, k=1, dtype=int)
            A[k1, k1 + 1] = 0
            A[0, k1 + 1] = 1
        return A

    def indices(self, a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def find_parent(self, B, i):
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # Find the index of the unique parent of i in B
        j = self.indices(B[:, i], lambda x: x == 1)  # indices of all neighbors of i in B
        # (!!! Also finds self-loop)
        return min(j)

    def tree_sample(self, B, x):
        # A = N by N matrix giving edge weights on networks
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # samples a tree B from a given pivot x as the first node
        N = self.G
        k = np.shape(B)[0]
        emb = np.array([x])  # initialize path embedding

        if sum(sum(B)) == 0:  # B is a set of isolated nodes
            y = np.random.randint(N.num_nodes(), size=(1, k - 1))
            y = y[0]  # juts to make it an array
            emb = np.hstack((emb, y))
        else:
            for i in np.arange(1, k):
                j = self.find_parent(B, i)
                nbs_j = np.asarray(list(N.neighbors(emb[j])))
                if len(nbs_j) > 0:
                    y = np.random.choice(nbs_j)
                else:
                    y = emb[j]
                    print('tree_sample_failed:isolated')
                emb = np.hstack((emb, y))
        # print('emb', emb)
        return emb

    def glauber_gen_update(self, B, emb):
        N = self.G
        k = np.shape(B)[0]

        if k == 1:

            emb[0] = self.walk(emb[0], 1)
            # print('Glauber chain updated via RW')
        else:
            j = np.random.choice(np.arange(0, k))  # choose a random node to update
            nbh_in = self.indices(B[:, j], lambda x: x == 1)  # indices of nbs of j in B
            nbh_out = self.indices(B[j, :], lambda x: x == 1)  # indices of nbs of j in B

            # build distribution for resampling emb[j] and resample emb[j]
            time_a = time()
            cmn_nbs = N.nodes(is_set=True)
            time_1 = time()
            time_neighbor = 0

            if not self.if_wtd_network:
                for r in nbh_in:
                    time_neighb = time()
                    nbs_r = N.neighbors(emb[r])
                    end_neighb = time()
                    time_neighbor += end_neighb - time_neighb
                    if len(cmn_nbs) == 0:
                        cmn_nbs = nbs_r
                    else:
                        cmn_nbs = cmn_nbs & nbs_r

                for r in nbh_out:
                    nbs_r = N.neighbors(emb[r])
                    if len(cmn_nbs) == 0:
                        cmn_nbs = nbs_r
                    else:
                        cmn_nbs = cmn_nbs & nbs_r

                cmn_nbs = list(cmn_nbs)
                if len(cmn_nbs) > 0:
                    y = np.random.choice(np.asarray(cmn_nbs))
                    emb[j] = y
                else:
                    emb[j] = np.random.choice(N.nodes())
                    print('Glauber move rejected')  # Won't happen once a valid embedding is established

            else: ### Now need to use edge weights for Glauber chain update as well
                # build distribution for resampling emb[j] and resample emb[j]
                cmn_nbs = [i for i in N.nodes()]
                for r in nbh_in:
                    nbs_r = [i for i in N.neighbors(emb[r])]
                    cmn_nbs = list(set(cmn_nbs) & set(nbs_r))
                for r in nbh_out:
                    nbs_r = [i for i in N.neighbors(emb[r])]
                    cmn_nbs = list(set(cmn_nbs) & set(nbs_r))
                if len(cmn_nbs) > 0:
                    ### Compute distribution on cmn_nbs
                    dist = np.ones(len(cmn_nbs))
                    for v in np.arange(len(cmn_nbs)):
                        for r in nbh_in:
                            dist[v] = dist[v] * abs(N.get_edge_weight(emb[r], cmn_nbs[v]))
                        for r in nbh_out:
                            dist[v] = dist[v] * abs(N.get_edge_weight(cmn_nbs[v], emb[r]))
                            ### As of now (05/15/2020) Wtd_NNetwork class has weighted edges without orientation,
                            ### so there is no distinction between in- and out-neighbors
                            ### Use abs since edge weights could be negative
                    dist = dist / np.sum(dist)
                    y = np.random.choice(np.asarray(cmn_nbs), p=dist)
                    emb[j] = y
                else:
                    emb[j] = np.random.choice(np.asarray([i for i in self.G.nodes]))
                    print('Glauber move rejected')  # Won't happen once valid embedding is established

        return emb

    def Pivot_update(self, emb):
        # G = underlying simple graph
        # emb = current embedding of a path in the network
        # k1 = length of left side chain from pivot
        # updates the current embedding using pivot rule

        k1 = self.k1
        k2 = self.k2
        x0 = emb[0]  # current location of pivot
        x0 = self.RW_update(x0)  # new location of the pivot
        B = self.path_adj(k1, k2)
        #  emb_new = self.Path_sample_gen_position(x0, k1, k2)  # new path embedding
        emb_new = self.tree_sample(B, x0)  # new path embedding
        return emb_new

    def RW_update(self, x):
        # G = simple graph
        # x = RW is currently at site x
        # stationary distribution = uniform

        N = self.G
        nbs_x = np.asarray(list(N.neighbors(x)))  # array of neighbors of x in G
        #  dist_x = np.where(dist_x > 0, 1, 0)  # make 1 if positive, otherwise 0
        # (!!! The above line seem to cause disagreement b/w Pivot and Glauber chains in WAN data for
        # k1=0 and k2=1 case and other inner edge CHD cases -- 9/30/19)

        if len(nbs_x) > 0:  # this holds if the current location x of pivot is not isolated
            y = np.random.choice(nbs_x)  # choose a uniform element in nbs_x

            # Use MH-rule to accept or reject the move
            # Use another coin flip (not mess with the kernel) to keep the computation local and fast
            nbs_y = np.asarray(list(N.neighbors(y)))
            prop_accept = min(1, len(nbs_x) / len(nbs_y))

            if np.random.rand() > prop_accept:
                y = x  # move to y rejected

        else:  # if the current location is isolated, uniformly choose a new location
            y = np.random.choice(np.asarray(N.nodes()))
        return y

    def chd_gen_mx(self, B, emb, iterations=1, is_glauber=True, verbose=0):
        # computes a B-patch of the input network G using Glauber chain to evolve embedding of B in to G
        # iterations = number of iteration
        # underlying graph = specified by A
        # B = adjacency matrix of rooted tree motif
        start = time()

        N = self.G
        emb2 = emb
        k = B.shape[0]
        #  x0 = np.random.choice(np.arange(0, N))  # random initial location of RW
        #  emb2 = self.tree_sample(B, x0)  # initial sampling of path embedding

        hom_mx2 = np.zeros([k, k])
        if self.if_tensor_ntwk:
            hom_mx2 = np.zeros([k, k, N.color_dim])

        for i in range(iterations):
            start_iter = time()
            if is_glauber:
                emb2 = self.glauber_gen_update(B, emb2)
            else:
                emb2 = self.Pivot_update(emb2)
            end_update = time()
            # start = time.time()

            if not self.if_tensor_ntwk:
                # full adjacency matrix or induced weight matrix over the path motif
                a2 = np.zeros([k, k])
                start_loop = time()
                for q in np.arange(k):
                    for r in np.arange(k):
                        if not self.if_wtd_network or N.has_edge(emb2[q], emb2[r])==0:
                            a2[q, r] = int(N.has_edge(emb2[q], emb2[r]))
                        else:
                            a2[q, r] = N.get_edge_weight(emb2[q], emb2[r])

                hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)

            else: # full induced weight tensor over the path motif (each slice of colored edge gives a weight matrix)
                a2 = np.zeros([k, k, N.color_dim])
                start_loop = time()
                for q in np.arange(k):
                    for r in np.arange(k):
                        if N.has_edge(emb2[q], emb2[r])==0:
                            a2[q, r, :] = np.zeros(N.color_dim)
                        else:
                            a2[q, r, :] = N.get_colored_edge_weight(emb2[q], emb2[r])
                            # print('np.sum(a2[q, r, :])', np.sum(a2[q, r, :]))
                hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)

            if (verbose):
                print([int(i) for i in emb2])

        return hom_mx2, emb2

    def get_patches_glauber(self, B, emb):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0] # len of motif. k-chain.
        X = np.zeros((k ** 2, 1)) # flatten matrix of dimension k * k to k^2 * 1.
        if self.if_tensor_ntwk:
            X = np.zeros((k ** 2, self.G.color_dim, 1))

        for i in np.arange(self.sample_size):
            Y, emb = self.chd_gen_mx(B, emb, iterations=1, is_glauber=self.is_glauber_dict)  # k by k matrix
            if not self.if_tensor_ntwk:
                Y = Y.reshape(k ** 2, -1)
            else:
                Y = Y.reshape(k ** 2, self.G.color_dim, -1)

            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=-1)  # x is class ndarray
        #  now X.shape = (k**2, sample_size) or (k**2, color_dim, sample_size)
        # print(X)
        return X, emb

    def get_single_patch_glauber(self, B, emb):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        Y, emb = self.chd_gen_mx(B, emb, iterations=1, is_glauber=self.is_glauber_recons)  # k by k matrix
        if not self.if_tensor_ntwk:
            X = Y.reshape(k ** 2, -1)
        else:
            X = Y.reshape(k ** 2, self.G.get_edge_color_dim(), -1)

        return X, emb

    def glauber_walk(self, x0, length, iters=1, verbose=0):

        N = self.G
        B = self.path_adj(0, length)
        # x0 = 2
        # x0 = np.random.choice(np.asarray([i for i in G]))
        emb = self.tree_sample(B, x0)
        k = B.shape[0]

        emb, _ = self.chd_gen_mx(B, emb, iterations=iters, verbose=0)

        return [int(i) for i in emb]

    def walk(self, node, iters=10):
        for i in range(iters):
            node = np.random.choice(self.G.neighbors(node))

        return node

    def train_dict(self):
        # emb = initial embedding of the motif into the network
        print('training dictionaries from patches...')
        '''
        Trains dictionary based on patches.
        '''
        ipdb.set_trace()

        G = self.G
        B = self.path_adj(self.k1, self.k2)
        x0 = np.random.choice(np.asarray([i for i in G.vertices]))
        emb = self.tree_sample(B, x0)
        W = self.W
        print('W.shape', W.shape)
        errors = []
        code = self.code
        for t in np.arange(self.MCMC_iterations):
            X, emb = self.get_patches_glauber(B, emb)
            # print('X.shape', X.shape)  ## X.shape = (k**2, sample_size)
            if not self.if_tensor_ntwk:
                X = np.expand_dims(X, axis=1)     ### X.shape = (k**2, 1, sample_size)
            if t == 0:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      alpha=self.alpha,
                                      mode=2,
                                      learn_joint_dict=True,
                                      subsample=self.ONMF_subsample)  # max number of possible patches
                self.W, self.At, self.Bt, self.Ct, self.H = self.ntf.train_dict()
                self.H = code
            else:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_dict=self.W,
                                      ini_A=self.At,
                                      ini_B=self.Bt,
                                      ini_C=self.Ct,
                                      alpha=self.alpha,
                                      history=self.ntf.history,
                                      subsample=self.ONMF_subsample,
                                      mode=2,
                                      learn_joint_dict=True)
               # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                self.W, self.At, self.Bt, self.Ct, self.H = self.ntf.train_dict()
                code += self.H
                error = np.trace(self.W @ self.At @ self.W.T) - 2 * np.trace(self.W @ self.Bt) + np.trace(self.Ct)
                print('error', error)
                errors.append(error)
            #  progress status
            # if 100 * t / self.MCMC_iterations % 1 == 0:
            #    print(t / self.MCMC_iterations * 100)
            print('Current iteration %i out of %i' % (t, self.MCMC_iterations))
        self.code = code
        self.result_dict.update({'Dictionary learned':self.W})
        self.result_dict.update({'Motif size':self.k2 + 1})
        self.result_dict.update({'Code learned': self.code})
        # print(self.W)

    # Jianhao 09/07/2021. Generate MCMC samples.
    def gen_MCMC_sample_in_df(self, p2df, p2feature):
        import pandas as pd
        import pickle
        from tqdm import tqdm

        G = self.G
        B = self.path_adj(self.k1, self.k2)
        # Create history graph for edges has been visited.
        G_overlap_count = Wtd_NNetwork()
        G_overlap_count.add_nodes(nodes=[v for v in G.vertices])

        # get length of the motif.
        k = B.shape[0]
        x0 = np.random.choice(np.asarray([i for i in G.vertices]))
        emb = self.tree_sample(B, x0)

        # # count how many has been visited.
        # visited_graph = []
        
        # Start collecting samples from iid tree_sample
        X_mcmc = []
        for t in tqdm(np.arange(self.MCMC_iterations)):
            # sample initial node and create tree from it.
            # emb is a list of vertices in the form of ['V1', 'V2', ..., 'Vk']
            # MCMC sampling
            X_patch, emb_patch = self.get_patches_glauber(B, emb)
            hom_mx = X_patch[:, -1]
            emb = emb_patch

            # count how many edges has been visited after one emb generated.
            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                edge = [str(a), str(b)]

                if G_overlap_count.has_edge(a, b) == True:
                    # print(G_recons.edges)
                    # print('ind', ind)
                    new_edge_weight = G_overlap_count.get_edge_weight(a, b) + 1
                else:
                    new_edge_weight = 1

                G_overlap_count.add_edge(edge, weight = new_edge_weight, increment_weights = False)

            common_edges = G.intersection(G_overlap_count)
            coverage_accuracy = len(common_edges) / len(G.get_edges())
            if t % 100 == 0:
                print(f'>> iteration {t}, coverage in iid sampling: {coverage_accuracy * 100:.2f}%')



            # append flatten matrix to input data X.
            if t == 0:
                # first iteration
                X_mcmc = hom_mx.reshape(k * k, 1)
                node_mtx = emb.reshape(-1, 1)
            else:
                X_mcmc = np.append(X_mcmc, hom_mx.reshape(k * k, 1), axis = -1)
                node_mtx = np.append(node_mtx, emb.reshape(-1, 1), axis = -1)

        # X_mcmc should be num_sample * num_features
        print(f'>> iteration {t}, coverage in iid sampling: {coverage_accuracy * 100:.2f}%')
        X = X_mcmc.T
        node_mtx = node_mtx.T

        Y = -1 * np.ones((X.shape[0], 1))
        feat = [f'dim_{i}' for i in range(X.shape[1])]

        df_mcmc = pd.DataFrame(data = np.hstack((X, Y)), columns = feat + ['label'])
        df_mcmc.to_pickle(p2df)

        df_node_mtx = pd.DataFrame(data = node_mtx, columns = [f'node_{x}' for x in range(node_mtx.shape[1])])
        df_node_mtx.to_pickle(f'{p2df}_sample_node_matrix')

        with open(p2feature, 'wb') as f:
            pickle.dump(feat, f)
    # done jianhao 09/07/2021

    # jianhao 03/01/2021. Add function to generate independent samples.
    def gen_iid_samples_in_df(self, p2df, p2feature):
        import pandas as pd
        import pickle
        from tqdm import tqdm

        G = self.G
        B = self.path_adj(self.k1, self.k2)
        # Create history graph for edges has been visited.
        G_overlap_count = Wtd_NNetwork()
        G_overlap_count.add_nodes(nodes=[v for v in G.vertices])

        # get length of the motif.
        k = B.shape[0]

        # # count how many has been visited.
        # visited_graph = []
        
        # Start collecting samples from iid tree_sample
        X_iid = []
        for t in tqdm(np.arange(self.MCMC_iterations)):
            # sample initial node and create tree from it.
            # emb is a list of vertices in the form of ['V1', 'V2', ..., 'Vk']
            emb = []
            # rejection sampling
            while len(emb) < k:
                x = np.random.choice(np.asarray([i for i in G.vertices]))
                emb = np.array([x])  # initialize path embedding
    
                if sum(sum(B)) == 0:  # B is a set of isolated nodes
                    y = np.random.randint(G.num_nodes(), size=(1, k - 1))
                    y = y[0]  # juts to make it an array
                    emb = np.hstack((emb, y))
                else:
                    for i in np.arange(1, k):
                        j = self.find_parent(B, i)
                        nbs_j = np.asarray(list(G.neighbors(emb[j])))
                        if len(nbs_j) > 0:
                            y = np.random.choice(nbs_j)
                            emb = np.hstack((emb, y))
                        else:
                            print('tree_sample_failed:isolated, restart sampling')
                            # break from adding nodes to emb. len(emb) < k. trigger restart.
                            break

            # count how many edges has been visited after one emb generated.
            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                edge = [str(a), str(b)]

                if G_overlap_count.has_edge(a, b) == True:
                    # print(G_recons.edges)
                    # print('ind', ind)
                    new_edge_weight = G_overlap_count.get_edge_weight(a, b) + 1
                else:
                    new_edge_weight = 1

                G_overlap_count.add_edge(edge, weight = new_edge_weight, increment_weights = False)

            common_edges = G.intersection(G_overlap_count)
            coverage_accuracy = len(common_edges) / len(G.get_edges())
            if t % 100 == 0:
                print(f'>> iteration {t}, coverage in iid sampling: {coverage_accuracy * 100:.2f}%')



            # initialize homomorphism 
            hom_mx = np.zeros([k, k])
            # full adjacency matrix or induced weight matrix over the path motif
            a2 = np.zeros([k, k])

            for q in np.arange(k):
                for r in np.arange(k):
                    if not self.if_wtd_network or G.has_edge(emb[q], emb[r])==0:
                        a2[q, r] = int(G.has_edge(emb[q], emb[r]))
                    else:
                        a2[q, r] = G.get_edge_weight(emb[q], emb[r])

            # done with homomorphism
            i = 0
            hom_mx = ((hom_mx * i) + a2) / (i + 1)
        
            # append flatten matrix to input data X.
            if t == 0:
                # first iteration
                X_iid = hom_mx.reshape(k * k, 1)
                node_mtx = emb.reshape(-1, 1)
            else:
                X_iid = np.append(X_iid, hom_mx.reshape(k * k, 1), axis = -1)
                node_mtx = np.append(node_mtx, emb.reshape(-1, 1), axis = -1)

        # X_iid should be num_sample * num_features
        print(f'>> iteration {t}, coverage in iid sampling: {coverage_accuracy * 100:.2f}%')
        X = X_iid.T
        node_mtx = node_mtx.T

        Y = -1 * np.ones((X.shape[0], 1))
        feat = [f'dim_{i}' for i in range(X.shape[1])]

        df_iid = pd.DataFrame(data = np.hstack((X, Y)), columns = feat + ['label'])
        df_iid.to_pickle(p2df)

        df_node_mtx = pd.DataFrame(data = node_mtx, columns = [f'node_{x}' for x in range(node_mtx.shape[1])])
        df_node_mtx.to_pickle(f'{p2df}_sample_node_matrix')

        with open(p2feature, 'wb') as f:
            pickle.dump(feat, f)
    # done jianhao 03/01/2021

    def display_dict(self, title, save_filename):
        #  display learned dictionary
        W = self.W
        n_components = W.shape[1]
        rows = np.round(np.sqrt(n_components))
        rows = rows.astype(int)
        if rows ** 2 == n_components:
            cols = rows
        else:
            cols = rows + 1

        k = self.k1 + self.k2 + 1

        importance = np.sum(self.code, axis=1) / sum(sum(self.code))
        idx = np.argsort(importance)
        idx = np.flip(idx)

        if not self.if_tensor_ntwk:
            # fig, axs = plt.subplots(nrows=5, ncols=9, figsize=(7, 5),
            fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5, 5),
                                    subplot_kw={'xticks': [], 'yticks': []})
            k = self.k1 + self.k2 + 1  # number of nodes in the motif F
            for ax, j in zip(axs.flat, range(n_components)):
                ax.imshow(self.W.T[idx[j]].reshape(self.k, self.k), cmap="gray_r", interpolation='nearest')
                ax.set_xlabel('%1.2f' % importance[idx[j]], fontsize=13)  # get the largest first
                ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                # use gray_r to make black = 1 and white = 0

            plt.suptitle(title)
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
            fig.savefig('Network_dictionary/' + save_filename)
        else:
            W = W.reshape(k ** 2, self.G.color_dim, self.n_components)
            for c in range(self.G.color_dim):
                fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5, 5),
                                        subplot_kw={'xticks': [], 'yticks': []})

                for ax, j in zip(axs.flat, range(n_components)):
                    # importance = sum(code[j, :])/sum(sum(code))
                    # ax.set_xlabel('%1.2f' % importance, fontsize=15)
                    # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
                    ax.imshow(W[:,c,:].T[j].reshape(k, k), cmap="gray_r", interpolation='nearest')
                    # use gray_r to make black = 1 and white = 0

                plt.suptitle(title)
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
                fig.savefig('Network_dictionary/' + save_filename + '_color_' + str(c))

        # plt.show()

    def reconstruct_network(self,
                            recons_iter=100,
                            if_save_history=True,
                            if_construct_WtdNtwk=True,
                            use_checkpoint_refreshing=False,
                            ckpt_epoch=1000):
        print('reconstructing given network...')
        '''
        NNetwork version of the reconstruction algorithm (custom Neighborhood Network package for scalable Glauber chain sampling)
        Using large "ckpt_epoch" improves reconstruction accuracy but uses more memory
        '''

        G = self.G
        self.G_recons = Wtd_NNetwork()
        self.G_overlap_count = Wtd_NNetwork()
        self.G_recons.add_nodes(nodes=[v for v in G.vertices])
        self.G_overlap_count.add_nodes(nodes=[v for v in G.vertices])

        B = self.path_adj(self.k1, self.k2)
        k = self.k1 + self.k2 + 1  # size of the network patch
        x0 = np.random.choice(np.asarray([i for i in G.vertices]))
        emb = self.tree_sample(B, x0)
        emb_history = emb.copy()
        code_history = np.zeros(2*self.n_components)

        ### Extend the learned dictionary for the flip-symmetry of the path embedding
        atom_size, num_atoms = self.W.shape
        W_ext = np.empty((atom_size, 2 * num_atoms))
        W_ext[:, 0:num_atoms] = self.W[:, 0:num_atoms]
        W_ext[:, num_atoms:(2 * num_atoms)] = np.flipud(self.W[:, 0:num_atoms])

        ### Set up paths and folders
        default_folder = 'Temp_save_graphs'
        default_name_recons = 'temp_wtd_edgelist_recons'
        path_recons = default_folder + '/' + default_name_recons + '.txt'

        t0 = time()
        c = 0

        for t in np.arange(recons_iter):
            patch, emb = self.get_single_patch_glauber(B, emb)
            coder = SparseCoder(dictionary=W_ext.T,  ### Use extended dictioanry
                                transform_n_nonzero_coefs=None,
                                transform_alpha=0,
                                transform_algorithm='lasso_lars',
                                positive_code=True)
            # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
            # This only occurs when sparse coding a single array
            code = coder.transform(patch.T)

            if if_save_history:
                emb_history = np.vstack((emb_history, emb))
                code_history = np.vstack((code_history, code))

            patch_recons = np.dot(W_ext, code.T).T
            patch_recons = patch_recons.reshape(k, k)

            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                edge = [str(a), str(b)]

                if self.G_overlap_count.has_edge(a, b) == True:
                    # print(G_recons.edges)
                    # print('ind', ind)
                    j = self.G_overlap_count.get_edge_weight(a, b)
                else:
                    j = 0

                if self.G_recons.has_edge(a, b) == True:
                    new_edge_weight = (j * self.G_recons.get_edge_weight(a, b) + patch_recons[x[0], x[1]]) / (j + 1)
                else:
                    new_edge_weight = patch_recons[x[0], x[1]]

                if if_construct_WtdNtwk:
                    if new_edge_weight > 0:
                        self.G_recons.add_edge(edge, weight=new_edge_weight, increment_weights=False)
                    # print('G.num_edges', len(G_recons.edges))
                    self.G_overlap_count.add_edge(edge, weight=j + 1, increment_weights=False)

            has_saved_checkpoint = False
            # progress status, saving reconstruction checkpoint, and memory refreshing
            if t % 1000 == 0:
                self.result_dict.update({'homomorphisms_history': emb_history})
                self.result_dict.update({'code_history': code_history})
                print('iteration %i out of %i' % (t, recons_iter))

                # print('num edges in G_count', len(self.G_overlap_count.get_edges()))
                # print('num edges in G_recons', len(self.G_recons.get_edges()))

                if use_checkpoint_refreshing and t % ckpt_epoch == 0:
                    ### print out current memory usage
                    pid = os.getpid()
                    py = psutil.Process(pid)
                    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
                    print('memory use:', memoryUse)

                    ### Threshold and simplify the current reconstruction graph
                    G_recons_simplified = self.G_recons.threshold2simple(threshold=0.5)
                    G_recons_combined = Wtd_NNetwork()
                    G_recons_combined.add_edges(edges=G_recons_simplified.get_edges(),
                                                edge_weight=1,
                                                increment_weights=True)

                    ### Load and combine with the saved edges and reconstruction counts
                    if has_saved_checkpoint:
                        G_recons_combined.load_add_wtd_edges(path=path_recons, increment_weights=True)

                    ### Save current graphs
                    G_recons_combined.save_wtd_edgelist(default_folder=default_folder,
                                                        default_name=default_name_recons)

                    has_saved_checkpoint = True

                    ### Clear up the edges of the current graphs

                    # self.G_overlap_count.clear_edges()
                    # self.G_recons.clear_edges()
                    self.G_recons = Wtd_NNetwork()
                    self.G_overlap_count = Wtd_NNetwork()
                    self.G_recons.add_nodes(nodes=[v for v in G.vertices])
                    self.G_overlap_count.add_nodes(nodes=[v for v in G.vertices])

                # print('num edges in G_recons', len(self.G_recons.get_edges()))

        ### Finalize the simplified reconstruction graph
        G_recons_final = self.G_recons.threshold2simple(threshold=0.5)
        if use_checkpoint_refreshing:
            G_recons_combined = Wtd_NNetwork()
            if not self.if_wtd_network:
                G_recons_combined.add_edges(edges=G_recons_final.get_edges(),
                                            edge_weight=1,
                                            increment_weights=True)
                G_recons_combined.load_add_wtd_edges(path=path_recons, increment_weights=True)
                G_recons_final = G_recons_combined.threshold2simple(threshold=0.5)
            else:
                G_recons_combined.add_wtd_edges(edges=self.G_recons.get_wtd_edgelist(),
                                                increment_weights=True)
                G_recons_combined.load_add_wtd_edges(path=path_recons, increment_weights=True)
                G_recons_final = G_recons_combined

        self.result_dict.update({'Edges reconstructed': G_recons_final.get_edges()})

        print('Reconstructed in %.2f seconds' % (time() - t0))
        # print('result_dict', self.result_dict)
        if if_save_history:
            self.result_dict.update({'homomorphisms_history': emb_history})
            self.result_dict.update({'code_history': code_history})

        return G_recons_final


    def network_completion(self, foldername, filename, threshold=0.5, recons_iter=100):
        print('reconstructing given network...')
        '''
        Networkx version of the network completion algorithm
        Scale the reconstructed matrix B by np.max(A) and compare with the original network. 
        '''

        G = self.G
        G_recons = G
        G_overlap_count = nx.DiGraph()
        G_overlap_count.add_nodes_from([v for v in G])
        B = self.path_adj(self.k1, self.k2)
        k = self.k1 + self.k2 + 1  # size of the network patch
        x0 = np.random.choice(np.asarray([i for i in G]))
        emb = self.tree_sample(B, x0)
        t0 = time()
        c = 0

        for t in np.arange(recons_iter):
            patch, emb = self.get_single_patch_glauber(B, emb)
            coder = SparseCoder(dictionary=self.W.T,
                                transform_n_nonzero_coefs=None,
                                transform_alpha=0,
                                transform_algorithm='lasso_lars',
                                positive_code=True)
            # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
            # This only occurs when sparse coding a single array
            code = coder.transform(patch.T)
            patch_recons = np.dot(self.W, code.T).T
            patch_recons = patch_recons.reshape(k,k)

            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                ind1 = int(G_overlap_count.has_edge(a,b) == True)
                if ind1 == 1:
                    # print(G_recons.edges)
                    # print('ind', ind)
                    j = G_overlap_count[a][b]['weight']
                    new_edge_weight = (j * G_recons[a][b]['weight'] + patch_recons[x[0], x[1]]) / (j + 1)
                else:
                    j = 0
                    new_edge_weight = patch_recons[x[0], x[1]]

                ind2 = int(G_recons.has_edge(a, b) == True)
                if ind2 == 0:
                    G_recons.add_edge(a, b, weight=new_edge_weight)
                    G_overlap_count.add_edge(a, b, weight=j + 1)
            ### Only repaint upper-triangular


            # progress status
            # print('iteration %i out of %i' % (t, recons_iter))
            if 1000 * t / recons_iter % 1 == 0:
                print(t / recons_iter * 100)

        ### Round the continuum-valued Recons matrix into 0-1 matrix.
        G_recons_simple = nx.Graph()
        # edge_list = [edge for edge in G_recons.edges]
        for edge in G_recons.edges:
            [a, b] = edge
            conti_edge_weight = G_recons[a][b]['weight']
            binary_edge_weight = np.where(conti_edge_weight>threshold, 1, 0)
            if binary_edge_weight > 0:
                G_recons_simple.add_edge(a, b)

        self.G_recons = G_recons_simple
        ### Save reconstruction
        path_recons = 'Network_dictionary/'+ str(foldername) + '/' + str(filename)
        nx.write_edgelist(G_recons,
                          path=path_recons,
                          data=False,
                          delimiter=",")
        print('Reconstruction Saved')
        print('Reconstructed in %.2f seconds' % (time() - t0))
        return G_recons_simple




    def compute_recons_accuracy(self, G_recons):
        ### Compute reconstruction error
        G = self.G
        G_recons.add_nodes(G.vertices)
        common_edges = G.intersection(G_recons)
        recons_accuracy = len(common_edges) / len(G.get_edges())

        print('# edges of original ntwk=', len(G.get_edges()))
        print('# edges of reconstructed ntwk=', len(G_recons.get_edges()))
        print('reconstruction accuracy=', recons_accuracy)

        self.result_dict.update({'# edges of original ntwk': len(G.get_edges())})
        self.result_dict.update({'# edges of reconstructed ntwk=': len(G_recons.get_edges())})
        self.result_dict.update({'reconstruction accuracy=': recons_accuracy})
        return recons_accuracy

    def compute_A_recons(self, G_recons):
        ### Compute reconstruction error
        G_recons.add_nodes_from(self.G.vertices)
        A_recons = nx.to_numpy_matrix(G_recons, nodelist=self.G.vertices)
        ### Having "nodelist=G.nodes" is CRUCIAL!!!
        ### Need to use the same node ordering between A and G for A_recons and G_recons.
        return A_recons
