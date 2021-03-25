import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from helper_functions import get_num_samples
from helper_functions import cluster_dist
from helper_functions import infer_k
from helper_functions import get_data_size
from helper_functions import normalize
from thinsage_subsample import stratified_subsample_balanced
from thinsage_subsample import weighted_subsample


def distances_to_probabilities(distances):
    # convert to probabilities
    max_dist = max(distances)
    min_dist = min(distances)
    probs = [(max_dist + min_dist) - dist for dist in distances]
    return normalize(probs)


# pure implementation no regard for distance
def multiclass_subsample(data, size, k=None):
    # convert to list
    if isinstance(data, pd.DataFrame):
        data = data.values.tolist()

    # get k clusters
    k = infer_k(data) if k is None else k

    # fit the model, get the clusters
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    cluster_labels = kmeans.labels_

    # subsample from each cluster
    subsample, sub_labels = stratified_subsample_balanced(data=data, class_list=cluster_labels, size=size)
    return subsample, sub_labels


# take distance into account
def multiclass_subsample_prob(data, size, k=None):
    """
    TO-DO: check for duplicates
    :param data:
    :param size:
    :param k:
    :return:
    """
    # convert to list
    if isinstance(data, pd.DataFrame):
        data = data.values.tolist()

    num_samples = get_num_samples(size=size, data_size=get_data_size(data))

    # get k clusters
    k = infer_k(data) if k is None else k

    # fit the model, get the clusters
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    cluster_labels = kmeans.labels_

    # list of distances from each point to their assigned cluster center
    distances = [cluster_dist(data[i], kmeans.cluster_centers_[cluster_labels[i]]) for i in range(len(data))]

    # convert to probabilities
    probs = distances_to_probabilities(distances)

    # subsample with weights
    return weighted_subsample(data, num_samples, probs, cluster_labels, k)
