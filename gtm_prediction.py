import ugtm
import numpy as np
import time
import argparse
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
import math
import gtm_config


def gtm_classification(config, test_data_matrix, test_id_list, data, labels, data_ids):
    prediction = ugtm.advancedGTC(
        train=data,
        labels=labels,
        test=test_data_matrix,
        doPCA=config.pca_preprocess,
        n_components=config.pca_n_components,
        n_neighbors=config.n_neighbors,
        representation=config.representation,
        missing=config.missing,
        missing_strategy=config.missing_strategy,
        random_state=config.random_state,
        k=config.grid_size,
        m=config.rbf_grid_size,
        predict_mode=config.predict_mode,
        prior=config.gtm_prior,
        regul=config.regul,
        s=config.rbf_width_factor
    )
    prediction["optimizedModel"].plot_html(
        ids=data_ids,
        plot_arrows=True,
        title="GTM",
        labels=labels,
        discrete=config.discrete_labels,
        output=config.output + "_trainedMap",
        cname=config.color_map,
        pointsize=config.pointsize,
        alpha=config.alpha,
        prior=config.gtm_prior,
        do_interpolate=config.interpolate,
    )
    ugtm.printClassPredictions(prediction, output=config.output)
    prediction["optimizedModel"].plot_html_projection(
        labels=labels,
        projections=prediction["indiv_projections"],
        ids=test_id_list,
        plot_arrows=True,
        title="GTM projection",
        discrete=config.discrete_labels,
        cname=config.color_map,
        pointsize=config.pointsize,
        output=config.output,
        alpha=config.alpha,
        prior=config.gtm_prior,
        do_interpolate=config.interpolate,
    )

def predict(config, classify_id):
    data = config.data
    labels = config.labels
    ids = config.ids
    sample_index = None
    for i in xrange(len(ids)):
        if ids[i] == classify_id:
            sample_index = i
    if not sample_index:
        print("Unable to find: " + str(classify_id) + " in dataset")
        exit 
    test_data = np.array(data[sample_index]).reshape(1, -1)
    filtered_data = np.delete(data, sample_index, axis=0)
    filtered_labels = np.delete(labels, sample_index, axis=0)
    filtered_ids = np.delete(ids, sample_index, axis=0)
    test_ids = np.array([classify_id]).reshape(1, -1)
    gtm_classification(
        config, test_data, test_ids, filtered_data, filtered_labels, filtered_ids
    )