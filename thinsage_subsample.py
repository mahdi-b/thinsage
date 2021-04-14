import numpy as np
import scipy as sp
from helper_functions import get_num_samples
from helper_functions import get_data_size
from helper_functions import normalize


def get_final_balanced_list(data, class_data, class_list, num_samples, num_classes):
    """
    helper function specifically for stratified subsample, (should not be callable by user)</br>
    take equal proportions from each class and combine into one list</br>
    while keeping track of the assigned classes</br>
    :param data: list of points representing data</br>
    :param class_data: data dictionary with key: class value: data val</br>
    :param class_list: corresponding class_label for data</br>
    :param num_samples: number of samples to take</br>
    :param num_classes: number of distinct class labels</br>
    :return: combined_final, class_final: tuple of lists representing
                data, and labels respectively
    """
    combined_list = []
    sub_class_list = []
    for i, key in enumerate(class_data.keys()):
        if i == (num_classes - 1):
            num_per_class = num_samples
        else:
            num_per_class = int(round(num_samples / (num_classes - i)))
            num_samples -= num_per_class
        random_perm = np.random.permutation(class_data[key])
        for j in range(num_per_class):
            combined_list.append(data[random_perm[j]])
            sub_class_list.append(class_list[random_perm[j]])

    # shuffle the combined list so that the returned list is NOT sorted by class
    final_perm = np.random.permutation(len(combined_list))
    combined_final = []
    class_final = []
    for i in range(len(final_perm)):
        combined_final.append(combined_list[final_perm[i]])
        class_final.append(sub_class_list[final_perm[i]])
    return combined_final, class_final


def get_final_imbalanced_list(data, class_data, class_list, proportions):
    # take samples equal to proportions from each class and combine into one list
    # while keeping track of the assigned classes
    combined_list = []
    sub_class_list = []
    for key in class_data.keys():
        random_perm = np.random.permutation(class_data[key])
        for i in range(proportions[key]):
            combined_list.append(data[random_perm[i]])
            sub_class_list.append(class_list[random_perm[i]])

    # shuffle the combined list so that the returned list is NOT sorted by class
    final_perm = np.random.permutation(len(combined_list))
    combined_final = [combined_list[final_perm[i]] for i in range(len(final_perm))]
    class_final = [sub_class_list[final_perm[i]] for i in range(len(final_perm))]

    return combined_final, class_final


def random_subsample(data, size=0.25, axis=0):
    """
    TO-DO: check for 2d in data and numpy array type
    takes in a list/collection and returns random sub-samples.
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent
                 percentage of samples to be taken.
    :param axis: If data is 2 dimensional, axis=0 will take rows as sub-samples, axis=1 will take columns
    :return: list/collection of type Object, of size equivalent to size.
    """

    # check data parameter
    data_size = get_data_size(data)

    # check size parameter
    num_samples = get_num_samples(size, data_size)

    random_perm = np.random.permutation(data_size)
    ret_list = [data[random_perm[i]] for i in range(num_samples)]
    return ret_list


def weighted_subsample(data, num_samples, probs, labels, k):
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


def stratified_subsample(data, class_list, size, balanced=True):
    """
    maybe use this for user ease of access
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.
    :param class_list: list of class identifiers, classList[i] = x implies data[i] belongs to class x
                       where x can be of type int, string, or object
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent
                 percentage of samples to be taken.
    :param balanced: boolean, If True, original proportions of class data is ignored and even proportions
                     will be returned (50:50 for 2 class data)
                     If false, original proportions will be kept. Prioritize proportions over desired size
    :return: randomized list containing balanced number samples from each class in the given data set
             matching class_list to indicate classes of the sub-samples.
    """
    return stratified_subsample_balanced(data=data, class_list=class_list, size=size) if balanced else \
        stratified_subsample_imbalanced(data=data, class_list=class_list, size=size)


def stratified_subsample_balanced(data, class_list, size):
    """
    TO-DO: check for numpy array data type
    data will be list/collection of type Object where each item belongs to a class as specified by class_list,
    data[i] will belong to class class_list[i].
    np.unique(class_list) should indicate number of different classes.
    :param data: list/collection of type Object. Contains the samples to be sub-sampled from.
    :param class_list: list of class identifiers, classList[i] = x implies data[i] belongs to class x
                       where x can be of type int, string, or object
    :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent
                 percentage of samples to be taken.
    :return: randomized list containing balanced number samples from each class in the given data set
             matching class_list to indicate classes of the sub-samples.
    """
    num_classes = len(np.unique(class_list))

    data_size = get_data_size(data)

    # check size parameter and get number of samples to return
    num_samples = get_num_samples(size, data_size)
    if num_samples == -1:
        raise Exception(f"Invalid value was given for parameter \'size\'")

    # make dictionary key is class and value is list of indices in data
    # where item is of class key
    class_data = {}
    for i in range(data_size):
        if class_list[i] not in class_data.keys():
            class_data[class_list[i]] = [i]
        else:
            class_data[class_list[i]].append(i)

    return get_final_balanced_list(data, class_data, class_list, num_samples, num_classes)




def stratified_subsample_imbalanced(data, class_list, size, exact=True):
    """
    TO-DO: check for numpy array data type</br>
    user ease function to decide between stratified_subsample_imbalanced_exact() and
    stratified_subsample_imbalanced_prob().
    imbalanced means an equal *proportion* from each class will be</br>
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
    """
    return stratified_subsample_imbalanced_exact(data=data, class_list=class_list, size=size) if exact\
        else stratified_subsample_imbalanced_prob(data, class_list, size)


def stratified_subsample_imbalanced_prob(data, class_list, size):
    """
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
    """
    data_size = get_data_size(data)

    if data_size != len(class_list):
        raise Exception(f'classes and data should be same size')

    # check size parameter and get number of samples desired
    num_samples = get_num_samples(size, data_size)
    if num_samples == -1:
        raise Exception(f"Invalid value was given for parameter \'size\'")

    proportions = {}
    for this_class in np.unique(class_list):
        proportions[this_class] = 0

    for this_class in class_list:
        proportions[this_class] += 1

    prop_list = [float(val / data_size) for val in proportions.values()]

    multi = sp.random.multinomial(num_samples, prop_list)
    for i, key in enumerate(proportions.keys()):
        proportions[key] = multi[i]

    # make dictionary key is class and value is list of indices in data
    # where item is of class key
    class_data = {}
    for i in range(data_size):
        if class_list[i] not in class_data.keys():
            class_data[class_list[i]] = [i]
        else:
            class_data[class_list[i]].append(i)

    return get_final_imbalanced_list(data, class_data, class_list, proportions)


def stratified_subsample_imbalanced_exact(data, class_list, size):
    """
       data will be list/collection of type Object where each item belongs to a class as specified by class_list,
       data[i] will belong to class class_list[i].
       np.unique(class_list) should indicate number of different classes.
       :param data: list/collection of type Object. Contains the samples to be sub-sampled from.
       :param class_list: list of class identifiers, classList[i] = x implies data[i] belongs to class x
                          where x can be of type int, string, or object
       :param size: Integer to determine how many samples desired, or float between 0 and 1 to represent
                    percentage of samples to be taken.
       :return: randomized list containing "imbalanced" (keep original proportions) number samples from each
                class in the given data set matching class_list to indicate classes of the sub-samples.
    """
    num_classes = len(np.unique(class_list))
    data_size = get_data_size(data)

    if data_size != len(class_list):
        raise Exception(f'classes and data should be same size')

    # check size parameter and get number of samples desired
    num_samples = get_num_samples(size, data_size)
    if num_samples == -1:
        raise Exception(f"Invalid value was given for parameter \'size\'")

    # calculate original proportions of classes
    proportions = {}
    for this_class in np.unique(class_list):
        proportions[this_class] = 0

    for this_class in class_list:
        proportions[this_class] += 1

    for this_class in np.unique(class_list):
        # proportions[this_class] = round(float(proportions[this_class] / data_size) * num_samples)
        proportions[this_class] = float(proportions[this_class] / data_size)

    # alter proportions to instead hold number of samples per class desired
    # (convert float to whole numbers)
    count = 0
    for i, key in enumerate(proportions.keys()):
        # proportions[key] = round(proportions[key] * num_samples)
        if i == num_classes - 1:
            proportions[key] = num_samples - count
        else:
            add = round(proportions[key] * num_samples)
            proportions[key] = add
            count += add

    # make dictionary key is class and value is list of indices in data
    # where item is of class key
    class_data = {}
    for i in range(data_size):
        if class_list[i] not in class_data.keys():
            class_data[class_list[i]] = [i]
        else:
            class_data[class_list[i]].append(i)

    return get_final_imbalanced_list(data, class_data, class_list, proportions)
