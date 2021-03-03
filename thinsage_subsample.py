import numpy as np
import random


def check_size_parameter(size, data_size):
    """
    helper function to check parameter 'size'. returns num_samples or -1 if invalid
    :param size: user given input parameter
    :param data_size: size of the original dataset
    :return: num_samples: the number of samples to select from the data
    """
    num_samples = -1
    if isinstance(size, float):
        if size <= 0 or size >= 1:
            raise Exception(f"Invalid value was given for parameter \'size\'")
        else:
            num_samples = np.floor((size * data_size))
    elif isinstance(size, int):
        if size >= data_size or size <= 0:
            raise Exception(f"Invalid value was given for parameter \'size\'")
        else:
            num_samples = size
    return num_samples


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
    if isinstance(data, list):
        data_size = len(data)
    else:
        data_size = data.shape[0]
    if data_size <= 0:
        raise Exception(f'An empty collection or list was given for the parameter\'data\'.')

    # check size parameter
    num_samples = check_size_parameter(size, data_size)

    # subsampling with random
    ret_list = []
    random_perm = np.random.permutation(data_size)
    for i in range(num_samples):
        ret_list.append(data[random_perm[i]])
    return ret_list


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
    return stratified_subsample_balanced(data, class_list, size) if balanced else \
        stratified_subsample_imbalanced(data, class_list, size)


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

    # validate type of 'data'
    if isinstance(data, list):
        data_size = len(data)
    else:
        data_size = data.shape[0]
    if data_size <= 0:
        raise Exception(f'An empty collection or list was given for the parameter\'data\'.')

    # check size parameter and get number of samples to return
    num_samples = check_size_parameter(size, data_size)
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

    # take equal proportions from each class and combine into one list
    # while keeping track of the assigned classes
    combined_list = []
    sub_class_list = []
    for i, key in enumerate(class_data.keys()):
        if i == (num_classes - 1):
            num_per_class = num_samples
        else:
            num_per_class = int(round(num_samples / (num_classes-i)))
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


def stratified_subsample_imbalanced(data, class_list, size):
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

    # check validity of data and get data size
    if isinstance(data, list):
        data_size = len(data)
    else:
        data_size = data.shape[0]

    if data_size != len(class_list):
        raise Exception(f'')

    # check size parameter and get number of samples desired
    num_samples = check_size_parameter(size, data_size)
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
        proportions[key] = round(proportions[key] * num_samples)
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
    combined_final = []
    class_final = []
    for i in range(len(final_perm)):
        combined_final.append(combined_list[final_perm[i]])
        class_final.append(sub_class_list[final_perm[i]])
    return combined_final, class_final
