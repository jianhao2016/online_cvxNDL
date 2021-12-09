#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script keeps all tmp functions.
"""
import numpy as np

def update_X_hat(W_hat, X_hat, x_new_sample):
    n_hat, k_cluster = W_hat.shape
    D_t = np.matmul(X_hat, W_hat)
    dist_to_centroids = lambda x: np.array([np.linalg.norm(x - D_t[:, j]) for j in range(k_cluster)])
    # find the assignment of X_hat and new incoming sample
    assigned_cluster = np.argmin(list(map(dist_to_centroids, X_hat.reshape(n_hat, -1))), axis  = 1)
    new_sample_cluster = np.argmin(dist_to_centroids(x_new_sample))

    # select the cluster with same label as new sample, update that cluster based on distance
    query_centorid = D_t[:, new_sample_cluster]
    test_dist = lambda x: np.linalg.norm( x - query_centorid)
    points_index_in_update_cluster = np.where(assigned_cluster == new_sample_cluster)
    cluster_to_update = X_hat[:, points_index_in_update_cluster]
    min_point_index = np.argmin([test_dist(x) for x in cluster_to_update.T])

    # if the distance of new sample point is further, then update.
    if test_dist(x_new_sample) > test_dist(X_hat[:, min_point_index]):
        X_hat[:, min_point_index] = x_new_sample
    return X_hat

def online_dict_learning(X, y_true, lmda, D_0, T, k_cluster, eps):
    '''
    algo 1 in the paper
    D_0: R^(m * k)
    X: R^(n * m)
    '''
    n_dim, m_dim = X.shape
    A_t = np.zeros((k_cluster, k_cluster))
    B_t = np.zeros((m_dim, k_cluster))
    D_t = D_0
    
    t_start = time.time()
    print(lmda, _NF, eps)
    for t in range(T):
        # t_start_online = time.time()
        if t % 50 == 0:
            tmp_assignment = get_clustering_assignment_1(X, D_t)
            tmp_acc, tmp_AMI = evaluation_clustering(tmp_assignment, y_true)
            print('1)iteration {}, distance acc = {:.4f}, AMI = {:.4f}'.format(t, tmp_acc, tmp_AMI))

            tmp_assignment = get_clustering_assignment_2(X, D_t, k_cluster, lmda)
            tmp_acc, tmp_AMI = evaluation_clustering(tmp_assignment, y_true)
            print('2)iteration {}, kmeans of weights acc = {:.4f}, AMI = {:.4f}'.format(t, tmp_acc, tmp_AMI))

            print('-' * 7)

        sample_idx = np.random.randint(0, n_dim)
        x_sample = X[sample_idx, :]

        lars_lasso = LassoLars(alpha = lmda)
        lars_lasso.fit(D_t, x_sample)
        alpha_t = lars_lasso.coef_

        A_t += np.matmul(alpha_t.reshape(k_cluster, 1), alpha_t.reshape(1, k_cluster))
        B_t += np.matmul(x_sample.reshape(m_dim, 1), alpha_t.reshape(1, k_cluster))

        D_t = dict_update(D_t, A_t, B_t, eps = eps)
        # print('===== Iteration in online dictionary learning cost {:.04f}s'.format(time.time() - t_start_online))
    print('Dcitionary update done! Time elapse {:.04f}s'.format(time.time() - t_start))
    return D_t


_NF = 1
def dict_update(D_t, A_t, B_t, eps):
    '''
    D_t: R^(m * k)
    A_t: R^(k * k)
    B_t: R^(m * k)
    '''
    m_dim, k_cluster = D_t.shape
    D_new = D_t.copy()

    # t_start = time.time()
    while True:
        D_old = D_new.copy()
        for j in range(k_cluster):
            grad = (B_t[:, j] - np.matmul(D_new, A_t[:, j]))
            u_j =  1/(np.linalg.norm(A_t[:, j]) + 1e-5) * grad + D_new[:, j]
            D_new[:, j] = (u_j / max(np.linalg.norm(u_j), 1)) * _NF
        if (np.linalg.norm(D_new - D_old) < eps):
            break
    # print('Iteration in dictionary update cost {:.04f}s'.format(time.time() - t_start))

    return D_new

