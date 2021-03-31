import pandas as pd
from sklearn.cluster import KMeans
from helper_functions import get_num_samples
from helper_functions import cluster_dist
from helper_functions import infer_k
from helper_functions import get_data_size
from helper_functions import normalize
from thinsage_subsample import stratified_subsample_balanced
from thinsage_subsample import weighted_subsample


def distances_to_probabilities(distances):
    """
    used in clustering, converts list of distances to inversely proportional probabilities.
    closer the point to the center, greater the probability.
    :param distances: array-like
    :return: list of probabilities corresponding to each distance.
    """
    # convert to probabilities
    max_dist = max(distances)
    min_dist = min(distances)
    probs = [(max_dist + min_dist) - dist for dist in distances]
    return normalize(probs)


# pure implementation no regard for distance
def multiclass_subsample(data, size, k=None, inflate_k=0):
    """
    takes sample based on inferred grouping.
    clusters points, then takes random samples from each cluster.
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent
                 percentage of samples to be taken.
    :param k: number of clusters that best fit the data if known.
                if unknown/undefined, k will be inferred.
    :param inflate_k: optional integer. if k is undefined k will be inferred,
                        then inflate_k will be added to the inferred k value.
    :return: tuple: subsample: containing subsampled y values and
                    sub_labels: corresponding x-labels containing either x_labels subsampled or integers corresponding to
                    the sampled y values
    """
    # convert to list
    if isinstance(data, pd.DataFrame):
        data = data.values.tolist()

    if not isinstance(inflate_k, int):
        raise Exception(f'expected value of type int for inflate_k but got {type(inflate_k)}.')

    # get k clusters
    k = infer_k(data) + inflate_k if k is None else k

    # fit the model, get the clusters
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    cluster_labels = kmeans.labels_

    # subsample from each cluster
    subsample, sub_labels = stratified_subsample_balanced(data=data, class_list=cluster_labels, size=size)
    return subsample, sub_labels


# take distance into account
def multiclass_subsample_prob(data, size, k=None, inflate_k=0):
    """
    takes sample based on inferred grouping.
    clusters points, then takes probabilistic samples
    from each cluster based on distance to cluster center.
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent
                 percentage of samples to be taken.
    :param k: number of clusters that best fit the data if known.
                if unknown/undefined, k will be inferred.
    :param inflate_k: optional integer. if k is undefined k will be inferred,
                        then inflate_k will be added to the inferred k value.
    :return: tuple: subsample: containing subsampled y values and
                    sub_labels: corresponding x-labels containing either x_labels subsampled or integers corresponding to
                    the sampled y values
    """
    # convert to list
    if isinstance(data, pd.DataFrame):
        data = data.values.tolist()

    if not isinstance(inflate_k, int):
        raise Exception(f'expected value of type int for inflate_k but got {type(inflate_k)}.')

    num_samples = get_num_samples(size=size, data_size=get_data_size(data))

    # get k clusters
    k = infer_k(data) + inflate_k if k is None else k

    # fit the model, get the clusters
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    cluster_labels = kmeans.labels_

    # list of distances from each point to their assigned cluster center
    distances = [cluster_dist(data[i], kmeans.cluster_centers_[cluster_labels[i]]) for i in range(len(data))]

    # convert to probabilities
    probs = distances_to_probabilities(distances)

    # subsample with weights
    return weighted_subsample(data, num_samples, probs, cluster_labels, k)
