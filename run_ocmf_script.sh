#! /bin/sh
#
# run_ocmf_script.sh
# Copyright (C) 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.
#

# NUMITER=10000
# # NUMITER=1
# K_CLUSTER=25
# LMDA=0
# DTYPE="iid_ndl"
# # EXPR_NAME="chr3R_drosophila_ChIA_Drop_0.1_PASS_10000"
# CANDIDATE=15
# MIN_SIZE=10
# VERSION=Rr
# PCA=-1
# 
# # chromo=chrX
# chromo=chr2R
# for chromo in chr2R chr2L chr3R chr3L
# do
#     EXPR_NAME=${chromo}_drosophila_ChIA_Drop_0.1_PASS_20000_MCMC_pivot
#     python -W ignore iid_sample_online_cvxNDL.py --numIter ${NUMITER} --NF 1 \
#         --lmda ${LMDA} \
#         --k_cluster ${K_CLUSTER} \
#         --dtype ${DTYPE} \
#         --candidate_size ${CANDIDATE} \
#         --expr_name ${EXPR_NAME} \
#         --size_min ${MIN_SIZE} \
#         --version ${VERSION} \
#         --pca ${PCA}
# done

NUMITER=1000
# NUMITER=1
K_CLUSTER=3
LMDA=0
DTYPE="sbm"
# EXPR_NAME="chr3R_drosophila_ChIA_Drop_0.1_PASS_10000"
CANDIDATE=5
MIN_SIZE=3
VERSION=Rr
PCA=-1

EXPR_NAME=sbm2_1000_MCMC_pivot
python -W ignore iid_sample_online_cvxNDL.py --numIter ${NUMITER} --NF 1 \
    --lmda ${LMDA} \
    --k_cluster ${K_CLUSTER} \
    --dtype ${DTYPE} \
    --candidate_size ${CANDIDATE} \
    --expr_name ${EXPR_NAME} \
    --size_min ${MIN_SIZE} \
    --version ${VERSION} \
    --pca ${PCA}
