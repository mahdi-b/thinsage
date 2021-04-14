### helper_functions.py

* get_num_samples(size, data_size):

    helper function to check parameter 'size'. returns num_samples or -1 if invalid</br>
    :param size: user given input parameter</br>
    :param data_size: size of the original dataset</br>
    :return: num_samples: the number of samples to select from the data</br>
  
* get_data_size(data):

    helper function to check data parameter and return length/size of data</br>
    :param data: data set given by the user</br>
    :return: integer representing length or shape[0] of data</br>
  
* avg(_list):

    takes avg of list of values</br>
    :param _list: given list</br>
    :return: float avg</br>
  
* normalize(_list):

    normalize a list of numbers</br>
    :param _list: list of numbers</br>
    :return: list normalized</br>
  
* cluster_dist(point, cluster_center):

    given two points returns Euclidean distance between the points</br>
    :param point: point in n-dimensional space</br>
    :param cluster_center: point in n-dimensional space</br>
    :return: float distance between the two</br>
  
* infer_k(data):

    iterates from 2-10 testing how many clusters best fit the given data</br>
    (uses silhouette index to guage)</br>
    :param data: given data</br>
    :return: best fit k</br>
  
### thinsage_subasample.py

* get_final_balanced_list(data, class_data, class_list, num_samples, num_classes):
    
    helper function specifically for stratified subsample, (should not be callable by user)</br>
    take equal proportions from each class and combine into one list</br>
    while keeping track of the assigned classes</br>
    :param data: list of points representing data</br>
    :param class_data: data dictionary with key: class value: data val</br>
    :param class_list: corresponding class_label for data</br>
    :param num_samples: number of samples to take</br>
    :param num_classes: number of distinct class labels</br>
    :return: combined_final, class_final: tuple of lists representing data, and labels respectively</br>
    
* get_final_imbalanced_list(data, class_data, class_list, proportions):
    
    imbalanced version of get_final_balanced_list(), again should not be callable by user</br>
    take samples equal to proportions from each class and combine into one list
    while keeping track of the assigned classes
    :param data: list of points representing data</br>
    :param class_data: data dictionary with key: class value: data val</br>
    :param class_list: corresponding class_label for data</br>
    :param proportions: dictionary with key: class value: number samples for class key
    :return: combined_final, class_final: tuple of lists representing data, and labels respectively</br>
    
* random_subsample(data, size=0.25, axis=0):

    TO-DO: check for 2d in data and numpy array type</br>
    takes in a list/collection and returns random sub-samples.</br>
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.</br>
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent</br>
                 percentage of samples to be taken.</br>
    :param axis: If data is 2 dimensional, axis=0 will take rows as sub-samples, axis=1 will take columns</br>
    :return: list/collection of type Object, of size equivalent to size.</br>

* weighted_subsample(data, num_samples, probs, labels, k):

    helper used for multiclass_prob(), not callable by user</br>
    :param data: list like data</br>
    :param num_samples: integer, number of samples to return</br>
    :param probs: list like probabilities</br>
    :param labels: cluster labels</br>
    :param k: number of clusters</br>
    :return: list like subsample chosen based off probabilities</br>
    
* stratified_subsample(data, class_list, size, balanced=True):

    maybe use this for user ease of access. </br>
    function is purely to choose between stratified_subsample_balanced() stratified_subsample_imbalanced()</br>
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.</br>
    :param class_list: list of class identifiers, classList[i] = x implies data[i] belongs to class x</br>
                       where x can be of type int, string, or object</br>
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent</br>
                 percentage of samples to be taken.</br>
    :param balanced: boolean, If True, original proportions of class data is ignored and even proportions</br>
                     will be returned (50:50 for 2 class data)</br>
                     If false, original proportions will be kept. Prioritize proportions over desired size</br>
    :return: randomized list containing balanced number samples from each class in the given data set</br>
             matching class_list to indicate classes of the sub-samples.</br>
             
* stratified_subsample_balanced(data, class_list, size):

    TO-DO: check for numpy array data type</br>
    takes subsample based on class labels. balanced means an equal amount from each class will be sampled regardless of original ratio.
    data will be list/collection of type Object where each item belongs to a class as specified by class_list,</br>
    data[i] will belong to class class_list[i].</br>
    np.unique(class_list) should indicate number of different classes.</br>
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.</br>
    :param class_list: list of class identifiers, classList[i] = x implies data[i] belongs to class x</br>
                       where x can be of type int, string, or object</br>
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent</br>
                 percentage of samples to be taken.</br>
    :return: randomized list containing balanced number samples from each class in the given data set</br>
             matching class_list to indicate classes of the sub-samples.</br>
             
* stratified_subsample_imbalanced(data, class_list, size, exact=True):

    TO-DO: check for numpy array data type</br>
    user ease function to decide between stratified_subsample_imbalanced_exact() and </br>
    stratified_subsample_imbalanced_prob(). imbalanced means an equal *proportion* from each class will be</br> 
    sampled based on original ratios.
    data will be list/collection of type Object where each item belongs to a class as specified by class_list,</br>
    data[i] will belong to class class_list[i].</br>
    np.unique(class_list) should indicate number of different classes.</br>
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.</br>
    :param class_list: list of class identifiers, classList[i] = x implies data[i] belongs to class x</br>
                       where x can be of type int, string, or object</br>
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent</br>
                 percentage of samples to be taken.</br>
    :param exact: bool, samples will be exactly according to probabilities 
    :return: randomized list containing balanced number samples from each class in the given data set</br>
             matching class_list to indicate classes of the sub-samples.</br>
             
* stratified_subsample_imbalanced_prob(data, class_list, size):

    TO-DO: check for numpy array data type</br>
    takes subsample based on class labels. imbalanced means an equal *proportion* from each class will be</br>
      sampled based on original ratios.
    data will be list/collection of type Object where each item belongs to a class as specified by class_list,</br>
    data[i] will belong to class class_list[i].</br>
    np.unique(class_list) should indicate number of different classes.</br>
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.</br>
    :param class_list: list of class identifiers, classList[i] = x implies data[i] belongs to class x</br>
                       where x can be of type int, string, or object</br>
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent</br>
                 percentage of samples to be taken.</br>
    :return: randomized list containing balanced number samples from each class in the given data set</br>
             matching class_list to indicate classes of the sub-samples.</br>
             
* stratified_subsample_imbalanced_exact(data, class_list, size):

     data will be list/collection of type Object where each item belongs to a class as specified by class_list,</br>
     data[i] will belong to class class_list[i].</br>
     np.unique(class_list) should indicate number of different classes.</br>
     :param data: list/collection of type Object. Contains the samples to be sub-sampled from.</br>
     :param class_list: list of class identifiers, classList[i] = x implies data[i] belongs to class x</br>
                        where x can be of type int, string, or object</br>
     :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent</br>
                  percentage of samples to be taken.</br>
     :return: randomized list containing "imbalanced" (keep original proportions) number samples from each</br>
              class in the given data set matching class_list to indicate classes of the sub-samples.</br>
              
### thinsage_timeseries.py

* timeseries_random(data, size, x_labels=None):

    subsample from a timeseries dataset. samples are chosen randomly but will be returned in original respective order.</br>
    :param data: array-like containing the y values for each record.</br>
    :param size: the desired size of the subsample to be returned. either int representing the number or float</br>
                    representing the fraction of the original data size</br>
    :param x_labels: optional array like. if indices are not just simple integers, timeseries_random will return the</br>
                corresponding x_labels with the y samples chosen</br>
    :return: tuple: ret_list: containing subsampled y values and</br>
                labels: corresponding x-labels containing either x_labels subsampled or integers corresponding to</br>
                    the sampled y values</br>
                    
* timeseries_interval(data, k=0, size=0, x_labels=None):

    subsample from a time series data set. sample is taken every k records if k is defined.</br>
        if size is defined k will be calculated. both k and size can not be defined.</br>
    :param data: array-like containing the y values for each record.</br>
    :param k: interval step size</br>
    :param size: the desired size of the subsample to be returned. either int representing the number or float</br>
                    representing the fraction of the original data size</br>
    :param x_labels: optional array like. if indices are not just simple integers, timeseries_random will return the</br>
                corresponding x_labels with the y samples chosen</br>
    :return: tuple: ret_list: containing subsampled y values and</br>
                labels: corresponding x-labels containing either x_labels subsampled or integers corresponding to</br>
                    the sampled y values</br>
                    
* timeseries_bucket_random(data, num_buckets, per_bucket=1, x_labels=None):

    splits time series data into sections or "buckets", then takes equal samples from each bucket.</br>
    :param data: array-like containing the y values for each record.</br>
    :param num_buckets: number of sections to split data into. if per_bucket = 1, num_bucket samples will be returned</br>
    :param per_bucket: number of samples to take per section or "bucket"</br>
    :param x_labels: optional array like. if indices are not just simple integers, timeseries_random will return the</br>
            corresponding x_labels with the y samples chosen</br>
    :return: tuple: ret_list: containing subsampled y values and</br>
                labels: corresponding x-labels containing either x_labels subsampled or integers corresponding to</br>
                    the sampled y values</br>
                    
* timeseries_sliding_window(data, w_size, f=avg, delta=.1, x_labels=None):

    creates "window" of the last w_size records or data points. takes a sample iff the difference</br>
        between the next sample and previous window samples is greater than some delta.</br>
    :param data: array-like containing the y values for each record.</br>
    :param w_size: int, window size</br>
    :param f: function, (avg by default). other examples: max, min, median</br>
    :param delta: float fraction to define the how different the change must be to take a sample</br>
    :param x_labels: optional array like. if indices are not just simple integers, timeseries_random will return the</br>
                corresponding x_labels with the y samples chosen</br>
    :return: tuple: ret_list: containing subsampled y values and</br>
                labels: corresponding x-labels containing either x_labels subsampled or integers corresponding to</br>
                    the sampled y values</br>
                    
* timeseries_LTTB(data, threshold):

    Return a downsampled version of data.</br>
    Parameters</br>
    *** original code found here: https://github.com/devoxi/lttb-py/blob/master/lttb/lttb.py</br>
    slight edit to take another format of data</br>
    :param data: list or, list of lists/tuples</br>
        data can be formated this way: [[x,y], [x,y], [x,y], ...]</br>
                                    or: [(x,y), (x,y), (x,y), ...]</br>
                                    or: [y1, y2, y3, y4 ...]</br>
    :param threshold: int</br>
        threshold must be >= 2 and <= to the len of data</br>
    :return: data, but downsampled using threshold</br>
    
### thinsage_multiclass.py:

* distances_to_probabilities(distances):

    used in clustering, converts list of distances to inversely proportional probabilities.</br>
    closer the point to the center, greater the probability.</br>
    :param distances: array-like</br>
    :return: list of probabilities corresponding to each distance.</br>
    
* multiclass_subsample(data, size, k=None, inflate_k=0):

    takes sample based on inferred grouping.</br>
    clusters points, then takes random samples from each cluster.</br>
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.</br>
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent</br>
                 percentage of samples to be taken.</br>
    :param k: number of clusters that best fit the data if known.</br>
                if unknown/undefined, k will be inferred.</br>
    :param inflate_k: optional integer. if k is undefined k will be inferred,</br>
                        then inflate_k will be added to the inferred k value.</br>
    :return: tuple: subsample: containing subsampled y values and</br>
                    sub_labels: corresponding x-labels containing either x_labels subsampled or integers corresponding to</br>
                    the sampled y values</br>
                    
* multiclass_subsample_prob(data, size, k=None, inflate_k=0):

    takes sample based on inferred grouping.</br>
    clusters points, then takes probabilistic samples</br>
    from each cluster based on distance to cluster center.</br>
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.</br>
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent</br>
                 percentage of samples to be taken.</br>
    :param k: number of clusters that best fit the data if known.</br>
                if unknown/undefined, k will be inferred.</br>
    :param inflate_k: optional integer. if k is undefined k will be inferred,</br>
                        then inflate_k will be added to the inferred k value.</br>
    :return: tuple: subsample: containing subsampled y values and</br>
                    sub_labels: corresponding x-labels containing either x_labels subsampled or integers corresponding to</br>
                    the sampled y values</br>
