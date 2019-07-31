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
import util


def gtm_classification(config, predict_data):
    print(np.size(predict_data.filtered_data))
    print(np.size(predict_data.filtered_labels))
    print(config.pca_preprocess)

    prediction = ugtm.advancedGTC(
        train=predict_data.filtered_data,
        labels=predict_data.filtered_labels,
        test=predict_data.test_data,
        doPCA=config.pca_preprocess,
        n_components=config.pca_n_components,
        n_neighbors=config.n_neighbors,
        representation=config.representation,
        missing=config.missing,
        missing_strategy=config.missing_strategy,
        random_state=config.random_state,
        k=config.k,
        m=config.m,
        predict_mode=config.predict_mode,
        prior=config.gtm_prior,
        regul=config.regul,
        s=config.rbf_width_factor
    )
    prediction["optimizedModel"].plot_html(
        ids=predict_data.filtered_ids,
        plot_arrows=True,
        title="GTM",
        labels=predict_data.filtered_labels,
        discrete=config.discrete_labels,
        output=config.output + "_trainedMap",
        cname=config.color_map,
        pointsize=config.pointsize,
        alpha=config.alpha,
        prior=config.gtm_prior,
        do_interpolate=config.interpolate
    )
    ugtm.printClassPredictions(prediction, output=config.output)
    prediction["optimizedModel"].plot_html_projection(
        labels=predict_data.filtered_labels,
        projections=prediction["indiv_projections"],
        ids=predict_data.test_ids,
        plot_arrows=True,
        title="GTM projection",
        discrete=config.discrete_labels,
        cname=config.classify_color,
        pointsize=config.pointsize,
        output=config.output,
        alpha=config.alpha,
        prior=config.gtm_prior,
        do_interpolate=config.interpolate
    )


def predict(config, classify_id):
    predict_data = util.extract_sample(
        config.data, config.labels, config.ids, classify_id
    )
    gtm_classification(config, predict_data)
