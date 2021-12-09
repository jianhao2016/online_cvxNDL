#! /bin/sh
#
# run_ocmf_script.sh
# Copyright (C) 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.
#

NUMITER=10
K_CLUSTER=3
LMDA=0
DTYPE="iid_ndl"
EXPR_NAME="chr4_drosophila_ChIA_Drop_0.1_PASS_5000"
CANDIDATE=10
MIN_SIZE=5
VERSION=Rr
PCA=-1

python -W ignore iid_sample_online_cvxNDL.py --numIter ${NUMITER} --NF 1 \
    --lmda ${LMDA} \
    --k_cluster ${K_CLUSTER} \
    --dtype ${DTYPE} \
    --candidate_size ${CANDIDATE} \
    --expr_name ${EXPR_NAME} \
    --size_min ${MIN_SIZE} \
    --version ${VERSION} \
    --pca ${PCA}

