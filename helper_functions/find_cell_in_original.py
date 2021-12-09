#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script is used to find orginal cells for x_hat.
"""

import pandas as pd
from functools import reduce

p2original = '/data/shared/jianhao/10xGenomics_scRNA/pandasDF/pandas_dataframe_bcell_cd34cell_clusters_-1'
p2X = '/home/jianhao2/sc_project/extract500_cell_results/X_df'
p2xhat = '/home/jianhao2/sc_project/extract500_cell_results/x_hat_df'

original_df = pd.read_pickle(p2original)
X_df = pd.read_pickle(p2X)
xhat_df = pd.read_pickle(p2xhat)

selected_rows = reduce(lambda a, b: a & b, [X_df[x].isin(xhat_df[x]) for x in X_df.columns])

selected_X_df = X_df[selected_rows]
seleced_idx = selected_X_df.index

selected_X = original_df.iloc[seleced_idx]

p2selected_cells = '/home/jianhao2/sc_project/extract500_cell_results/selected_cell_results'
selected_X.to_pickle(p2selected_cells)
