import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from helper_functions import Helper
from thinsage_subsample import BasicSubsample


class MultiFeature:
    def __init__(self):
        self.subsample = BasicSubsample()
        self.help = Helper()

    def __weighted_subsample__(self, data, labels, num_samples, probs, k):
        """
        :param data:
        :param num_samples:
        :param probs:
        :param labels:
        :param k:
        :return:
        """

        # workaround for 1-D restriction of numpy.random.choice
        class Temp:
            def __init__(self, val):
                self.val = val

        subsample = []
        for i in range(k):
            cluster = []
            prob_list = []
            for j in range(len(data)):
                if labels[j] == i:
                    cluster.append(data[j])
                    prob_list.append(probs[j])
            # convert list of lists to list of objects for numpy.random.choice
            cluster = [Temp(cluster[i]) for i in range(len(cluster))]
            num = num_samples - len(subsample) if i == k - 1 else round(num_samples / k)
            subsample += list(np.random.choice(a=cluster, size=num, replace=False, p=normalize(prob_list)))

        # convert back to list of lists
        subsample = [i.val for i in subsample]
        return subsample

    def multiclass_subsample(self, data, size, k=None, inflate_k=0, prob=True):
        """
        takes sample based on inferred grouping.
        clusters points, then takes equal number samples per group
        prob=True means sample from each cluster based on distance to cluster center.
        prob=False means sample is randomly chosen from cluster
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
        return self.__multiclass_subsample_prob__(data=data, size=size, k=k, inflate_k=inflate_k) if prob else \
            self.__multiclass_subsample_rand__(data=data, size=size, k=k, inflate_k=inflate_k)

    def __multiclass_subsample_rand__(self, data, size, k=None, inflate_k=0):
        """
        takes sample based on inferred grouping.
        clusters points, then takes random samples
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

        # get k clusters
        k = self.help.__infer_k__(data) + inflate_k if k is None else k + inflate_k

        # fit the model, get the clusters
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        cluster_labels = kmeans.labels_

        # subsample from each cluster
        subsample, sub_labels = self.subsample.stratified_subsample_balanced(data=data, labels=cluster_labels, size=size)
        return subsample, sub_labels

    def __multiclass_subsample_prob__(self, data, size, k=None, inflate_k=0):
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

        num_samples = self.help.__get_num_samples__(size=size, data_size=self.help.__get_data_size__(data))

        # get k clusters
        k = self.help.__infer_k__(data) + inflate_k if k is None else k + inflate_k

        # fit the model, get the clusters
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        cluster_labels = kmeans.labels_

        # list of distances from each point to their assigned cluster center
        distances = [np.linalg.norm(data[i] - kmeans.cluster_centers_[cluster_labels[i]]) for i in range(len(data))]

        # convert to probabilities
        probs = self.help.__convert_to_probabilities__(distances)

        # subsample with weights
        return self.__weighted_subsample__(data, num_samples, probs, cluster_labels, k)