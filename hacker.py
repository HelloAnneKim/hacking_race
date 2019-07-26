import ugtm
import numpy as np
import time
import argparse
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import manifold
import math
import gtm_config
import gtm_prediction

# argument parsing
parser = argparse.ArgumentParser(
    description="Generate and assess GTM maps " "for classification " "or regression."
)

parser.add_argument(
    "--config-file", help="PATH to JSON config file", dest="config_file"
)
parser.add_argument(
    "--usetest",
    help="use S or swiss roll or iris test data",
    dest="usetest",
    choices=["s", "swiss", "iris"],
)
parser.add_argument(
    "--classify-id",
    help="id of sample to give a report on ancestry (for PCA or GTM)",
    dest="classify_id",
)

parser.add_argument(
    "--model",
    help="GTM model, kernel GTM model, SVM, PCA or "
    "comparison between: "
    "GTM, kGTM, LLE and tSNE for simple visualization, "
    "GTM and SVM for regression or "
    "classification (--crossvalidate); "
    "benchmarked parameters for GTM are "
    "regularization and rbf_width_factor "
    "for given grid_size and rbf_grid_size",
    dest="model",
    choices=["GTM", "kGTM", "SVM", "PCA", "t-SNE", "SVMrbf", "compare"],
)
parser.add_argument("--output", help="output name", dest="output")
"""
parser.add_argument('--crossvalidate',
                    help='show best regul (regularization coefficient) '
                         'and s (RBF width factor) '
                         'for classification or regression, '
                         'with default grid size parameter '
                         'k = sqrt(5*sqrt(Nfeatures))+2) '
                         'and RBF grid size parameter m = sqrt(k); '
                         'you can also set the 4 parameters '
                         'and run only one model with '
                         '--rbf_width_factor, --regularization, '
                         '--grid_size and --rbf_grid_size',
                    action='store_true')
"""
parser.add_argument("--verbose", help="verbose mode", action="store_true")

args = parser.parse_args()
config = gtm_config.Json_config(args.config_file)
print("")
print(args)
print("")
config.output = args.output
config.model = args.model


# process some of the arguments; make sure data is preprocessed if model is PCA
if args.model == "PCA":
    args.pca = True

type_of_experiment = "visualization"


if args.classify_id:
    classify_id = args.classify_id
    gtm_prediction.predict(config, classify_id)
    exit


# TYPE OF EXPERIMENT: 3: VISUALIZATION: CAN BE t-SNE, GTM, PCA
###################################################
###################################################
########### VISUALIZATION #########################
###################################################
###################################################


def distance(point1, point2):
    return distance.euclidean(point1, point2)


def score(data, test_sample):
    return {}
    
def centroids(pca_data, labels):
    pop_size = {}
    centroids = {}
    dimensions = len(pca_data[0])
    for label in labels:
        if label not in pop_size:
            pop_size[label] = 0
            centroids[label] = [0 for _d in xrange(dimensions)]
        pop_size[label] += 1
    for i in xrange(len(pca_data)):
        label = labels[i]
        centroids[label] = np.add(
            centroids[label], [x / pop_size[label] for x in pca_data[i]]
        )
    return centroids


def ordered_samples(pca_data, classify_id):
    return {}

if type_of_experiment == "manipulation":
    classify_id = args.classify_id
    print("TO BE IMPLEMENTED")
    

if type_of_experiment == "visualization":
    data = config.data
    labels = config.lables
    ids = config.ids
    # set default parameters
    k = int(math.sqrt(5 * math.sqrt(data.shape[0]))) + 2
    m = int(math.sqrt(k))
    regul = 0.1
    s = 0.3
    niter = 1000
    maxdim = 100
    if args.model != "GTM":
        pca_data = ugtm.pcaPreprocess(
            data=data,
            doPCA=args.pca,
            n_components=config.pca_n_components,
            missing=config.missing,
            missing_strategy=config.missing_strategy,
            random_state=config.random_state,
        )
        centroids = centroids(pca_data, labels)
        print(centroids)
        k = int(math.sqrt(5 * math.sqrt(pca_data.shape[0]))) + 2
        m = int(math.sqrt(k))

    # set parameters if provided in options
    if config.regularization:
        regul = args.regularization
    if config.rbf_width_factor:
        s = args.rbf_width_factor
    if config.grid_size:
        k = args.grid_size
    if config.rbf_grid_size:
        m = args.rbf_grid_size

    # PCA visualization
    if args.model == "PCA":
        # if discrete:
        #    uniqClasses, labels = np.unique(labels, return_inverse=True)
        ugtm.plot_html(
            labels=labels,
            coordinates=pca_data,
            ids=ids,
            title="",
            output=args.output,
            cname=config.color_map,
            pointsize=config.pointsize,
            alpha=config.alpha,
            discrete=discrete,
        )
        ugtm.plot(
            labels=labels,
            coordinates=pca_data,
            discrete=discrete,
            output=config.output,
            cname=config.color_map,
            pointsize=config.pointsize,
            alpha=config.alpha,
            title="",
        )
        np.savetxt(config.output + ".csv", pca_data, delimiter=",")
        exit

    # t-SNE visualization
    elif args.model == "t-SNE":
        #     if discrete:
        #         uniqClasses, labels = np.unique(labels, return_inverse=True)
        tsne = manifold.TSNE(n_components=2, init="pca", random_state=args.random_state)
        data_r = tsne.fit_transform(data)
        ugtm.plot_html(
            labels=labels,
            coordinates=data_r,
            ids=ids,
            discrete=discrete,
            output=args.output,
            cname=args.cname,
            pointsize=args.pointsize,
            alpha=args.alpha,
            title="",
        )
        ugtm.plot(
            labels=labels,
            coordinates=data_r,
            discrete=discrete,
            output=args.output,
            cname=args.cname,
            pointsize=args.pointsize,
            alpha=args.alpha,
            title="",
        )
        np.savetxt(args.output + ".csv", data_r, delimiter=",")
        exit

    # GTM visualization
    elif args.model == "GTM":
        start = time.time()
        gtm = ugtm.runGTM(
            data=data,
            k=k,
            m=m,
            s=s,
            regul=regul,
            niter=niter,
            doPCA=args.pca,
            n_components=args.n_components,
            missing=args.missing,
            missing_strategy=args.missing_strategy,
            random_state=args.random_state,
            verbose=args.verbose,
        )
        print("k:%s, m:%s, regul:%s, s:%s" % (k, m, regul, s))
        end = time.time()
        elapsed = end - start
        print("time taken for GTM: ", elapsed)
        np.savetxt(args.output + "_means.csv", gtm.matMeans, delimiter=",")
        gtm.plot_multipanel(
            labels=labels,
            output=args.output + "_multipanel",
            discrete=discrete,
            cname=args.cname,
            pointsize=args.pointsize,
            alpha=args.alpha,
            prior=args.prior,
            do_interpolate=args.interpolate,
        )
        gtm.plot_html(
            labels=labels,
            ids=ids,
            discrete=discrete,
            output=args.output,
            cname=args.cname,
            pointsize=args.pointsize,
            alpha=args.alpha,
            title="",
            prior=args.prior,
            do_interpolate=args.interpolate,
        )
        gtm.plot(
            labels=labels,
            output=args.output,
            discrete=discrete,
            pointsize=args.pointsize,
            alpha=args.alpha,
            cname=args.cname,
        )
        exit

    else:
        print("Sorry. Model not recognized.")
        exit
else:
    print(
        "Sorry. Could not guess what you wanted. "
        "Remember to define --model "
        "and (--data and --labels) or --model and --usetest."
    )
    exit
