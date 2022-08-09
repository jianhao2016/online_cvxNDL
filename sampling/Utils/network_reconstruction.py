from Utils.onmf import Online_NMF
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
                 is_stack=False,
                 alpha=None,
                 is_glauber_dict=True,
                 is_glauber_recons=True,
                 ONMF_subsample=True):
        '''
        batch_size = number of patches used for training dictionaries per ONMF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.G = G
        self.n_components = n_components
        self.MCMC_iterations = MCMC_iterations
        self.sub_iterations = sub_iterations
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.loc_avg_depth = loc_avg_depth
        self.k1 = k1
        self.k2 = k2
        self.patches_file = patches_file
        self.is_stack = is_stack  # if True, input data is a 3d array
        self.W = np.zeros(shape=((k1 + k2 + 1) ** 2, n_components))
        self.code = np.zeros(shape=(n_components, sample_size))
        self.code_recons = np.zeros(shape=(n_components, sample_size))
        self.alpha = alpha
        self.is_glauber_dict = is_glauber_dict  ### if false, use pivot chain for dictionary learning
        self.is_glauber_recons = is_glauber_recons  ### if false, use pivot chain for reconstruction
        self.edges_deleted = []
        self.ONMF_subsample = ONMF_subsample
        self.result_dict = {}

    '''
    def read_networks(self, path):
        G = nx.read_edgelist(path, delimiter=',')
        A = nx.to_numpy_matrix(G)
        A = np.squeeze(np.asarray(A))
        print(A.shape)
        # A = A / np.max(A)
        return A

    def np2nx(self, x):
        ### Gives bijection from np array node ordering to G.node()
        G = self.G
        a = np.asarray([v for v in G])
        return a[x]

    def nx2np(self, y):
    ### Gives bijection from G.node() to nx array node ordering
        G = self.G
        a = np.asarray([v for v in G])
        return np.min(np.where(a == y))
    '''

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
        start = time()

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

            for r in nbh_in:
                time_neighb = time()
                nbs_r = N.neighbors(emb[r])
                end_neighb = time()
                time_neighbor += end_neighb - time_neighb
                if len(cmn_nbs) == 0:
                    cmn_nbs = nbs_r

                else:
                    cmn_nbs = cmn_nbs & nbs_r

            time_2 = time()

            for r in nbh_out:
                nbs_r = N.neighbors(emb[r])
                if len(cmn_nbs) == 0:
                    cmn_nbs = nbs_r
                else:
                    cmn_nbs = cmn_nbs & nbs_r
            time_3 = time()
            cmn_nbs = list(cmn_nbs)
            if len(cmn_nbs) > 0:

                y = np.random.choice(np.asarray(cmn_nbs))

                emb[j] = y
            else:
                emb[j] = np.random.choice(N.nodes())

                print('Glauber move rejected')  # Won't happen once valid embedding is established

        end = time()
        """
        print("Update time: " + str(end - start))
        print("List to set: " + str(time_1 - time_a))
        print("First loop: " + str(time_2 - time_1))
        print("Neighbor time: " + str(time_neighbor))
        print("Second loop: " + str(time_3 - time_2))
        print("")
        """
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
        # computes B-patches of the input network G using Glauber chain to evolve embedding of B in to G
        # iterations = number of iteration
        # underlying graph = specified by A
        # B = adjacency matrix of rooted tree motif
        start = time()

        N = self.G

        emb2 = emb
        k = B.shape[0]
        #  x0 = np.random.choice(np.arange(0, N))  # random initial location of RW
        #  emb2 = self.tree_sample(B, x0)  # initial sampling of path embedding
        hom2 = np.array([])
        hom_mx2 = np.zeros([k, k])

        for i in range(iterations):
            start_iter = time()
            if is_glauber:
                emb2 = self.glauber_gen_update(B, emb2)
            else:
                emb2 = self.Pivot_update(emb2)
            end_update = time()
            # start = time.time()

            # full adjacency matrix over the path motif
            a2 = np.zeros([k, k])
            start_loop = time()
            for q in np.arange(k):
                for r in np.arange(k):
                    a2[q, r] = int(N.has_edge(emb2[q], emb2[r]))

            end_loop = time()
            hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)
            # end = time.time()
            # print("Loop:")
            # print(end - start)
            '''
            #  progress status
            if 100 * i / iterations % 1 == 0:
                print(i / iterations * 100)
            '''

            if (verbose):
                print([int(i) for i in emb2])

            end_iter = time()
            """
            if i < 5:
                print("iteration time: " + str(end_iter-start_iter))
                print("update time: " + str(end_update - start_iter))
                print("loop time: " + str(end_loop - start_loop))
            """
        end = time()
        """ 
        print("chd_gen_mx: " + str(end-start))
        """

        return hom_mx2, emb2

    def get_patches_glauber(self, B, emb):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        X = np.zeros((k ** 2, 1))
        for i in np.arange(self.sample_size):
            Y, emb = self.chd_gen_mx(B, emb, iterations=1, is_glauber=self.is_glauber_dict)  # k by k matrix
            Y = Y.reshape(k ** 2, -1)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=1)  # x is class ndarray
        #  now X.shape = (k**2, sample_size)
        # print(X)
        return X, emb

    def get_single_patch_glauber(self, B, emb):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        Y, emb = self.chd_gen_mx(B, emb, iterations=1, is_glauber=self.is_glauber_recons)  # k by k matrix
        X = Y.reshape(k ** 2, -1)

        #  now X.shape = (k**2, sample_size)
        # print(X)
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

        G = self.G
        B = self.path_adj(self.k1, self.k2)
        x0 = np.random.choice(np.asarray([i for i in G.vertices]))
        emb = self.tree_sample(B, x0)
        W = self.W
        print('W.shape', W.shape)
        At = []
        Bt = []
        Ct = []
        errors = []
        code = self.code
        for t in np.arange(self.MCMC_iterations):
            X, emb = self.get_patches_glauber(B, emb)
            # print('X.shape', X.shape)
            if t == 0:
                self.nmf = Online_NMF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      alpha=self.alpha,
                                      subsample=self.ONMF_subsample)  # max number of possible patches
                self.W, self.At, self.Bt, self.Ct, self.H = self.nmf.train_dict()
                self.H = code
            else:
                self.nmf = Online_NMF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_dict=self.W,
                                      ini_A=self.At,
                                      ini_B=self.Bt,
                                      ini_C=self.Ct,
                                      alpha=self.alpha,
                                      history=self.nmf.history,
                                      subsample=self.ONMF_subsample)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                self.W, self.At, self.Bt, self.Ct, self.H = self.nmf.train_dict()
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
        self.result_dict.update({'MCMC iteration for dictionary learning':self.MCMC_iterations})
        self.result_dict.update({'Motif size':self.k2 + 1})
        self.result_dict.update({'Code learned': self.code})
        # print(self.W)

    def display_dict(self, title, save_filename):
        #  display learned dictionary
        W = self.W
        code = self.code  # row sum of code matrix will give importance of each dictionary patch
        n_components = W.shape[1]
        rows = np.round(np.sqrt(n_components))
        rows = rows.astype(int)
        if rows ** 2 == n_components:
            cols = rows
        else:
            cols = rows + 1

        # fig, axs = plt.subplots(nrows=5, ncols=9, figsize=(7, 5),
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5, 5),
                                subplot_kw={'xticks': [], 'yticks': []})
        k = self.k1 + self.k2 + 1  # number of nodes in the motif F
        for ax, j in zip(axs.flat, range(n_components)):
            # importance = sum(code[j, :])/sum(sum(code))
            # ax.set_xlabel('%1.2f' % importance, fontsize=15)
            # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
            ax.imshow(W.T[j].reshape(k, k), cmap="gray_r", interpolation='nearest')
            # use gray_r to make black = 1 and white = 0

        plt.suptitle(title)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig.savefig('../Network_dictionary/Facebook/' + save_filename + "NEW.png")
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
        code_history = np.zeros(self.n_components)

        ### Set up paths and folders
        default_folder = 'Temp_save_graphs'
        default_name_recons = 'temp_wtd_edgelist_recons'
        path_recons = default_folder + '/' + default_name_recons + '.txt'

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

            if if_save_history:
                emb_history = np.vstack((emb_history, emb))
                code_history = np.vstack((code_history, code))

            patch_recons = np.dot(self.W, code.T).T
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
        G_recons_simplified = self.G_recons.threshold2simple(threshold=0.5)
        if use_checkpoint_refreshing:
            G_recons_combined = Wtd_NNetwork()
            G_recons_combined.add_edges(edges=G_recons_simplified.get_edges(),
                                        edge_weight=1,
                                        increment_weights=True)
            G_recons_combined.load_add_wtd_edges(path=path_recons, increment_weights=True)
            G_recons_simplified = G_recons_combined.threshold2simple(threshold=0.5)

        self.result_dict.update({'Edges reconstructed': G_recons_simplified.get_edges()})

        print('Reconstructed in %.2f seconds' % (time() - t0))
        # print('result_dict', self.result_dict)
        if if_save_history:
            self.result_dict.update({'homomorphisms_history': emb_history})
            self.result_dict.update({'code_history': code_history})

        return G_recons_simplified

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