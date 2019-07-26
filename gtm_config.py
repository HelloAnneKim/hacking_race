import json
import os
import numpy as np


class Json_config:
    def __init__(self, json_file):
        with open(json_file) as json_data_file:
            self.config_data = json.load(json_data_file)
        self.data = np.genfromtxt(
            os.path.expanduser(self.config_data["data_file"]),
            delimiter=",",
            dtype=np.float64,
        )
        self.ids = np.genfromtxt(
            os.path.expanduser(self.config_data["id_file"]), delimiter="\t", dtype=str
        )
        self.discrete_labels = True
        label_path = os.path.expanduser(self.config_data["label_file"])
        if self.config_data.get("continuous_labels"):
            self.discrete_labels = False
            self.labels = np.genfromtxt(label_path, delimiter="\t", dtype=float)
        else:
            print("Discrete labels")
            self.labels = np.genfromtxt(label_path, delimiter="\t", dtype=str)
        self.pca_preprocess = bool(self.config_data.get("pca_preprocess"))
        self.missing = True
        self.missing_strategy = "median"
        if self.config_data.get("missing_data_strategy"):
            # mean, median, most_frequent
            self.missing_strategy = self.config_data.get("missing_data_strategy")
        self.predict_mode = "bayes"
        if self.config_data.get("predict_mode"):
            # bayes, knn
            # t-sne and PCA only support knn
            self.predictmode = self.config_data.get("predict_mode")
        self.gtm_prior = "equiprobable"
        if self.config_data.get("gtm_prior"):
            self.gtm_prior = self.config_data.get("gtm_prior")
        self.pca_n_components = 10
        if self.config_data.get("pca_n_components"):
            self.gtm_prior = self.config_data.get("pca_n_components")
        self.regul = 0.1
        self.rbf_width_factor = 0.3
        self.svm_margin = 1.0
        self.svm_epsilon = 1.0
        self.pointsize = 1.0
        self.alpha = 0.5
        self.svm_gamma = 1.0
        if self.config_data.get("grid_size") :
            self.grid_size = self.config_data.get("grid_size")
        else :
            self.grid_size = 0
        if self.config_data.get("rbf_grid_size"):
            self.rbf_grid_size = self.config_data.get("rbf_grid_size")
        else:
            self.rbf_grid_size = 0
        self.n_neighbors = 1
        self.random_state = 8
        self.representation = "modes"
        self.color_map = "Spectral_r"
        self.admixtures = 3
        self.interpolate = False
