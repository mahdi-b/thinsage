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
    for i in range(num_classes):
        if i == (num_classes - 1):
            num_per_class = num_samples
        else:
            num_per_class = int(round(num_samples / (num_classes-i)))
            num_samples -= num_per_class
        random_perm = np.random.permutation(class_data[class_list[i]])
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

# def stratified_subsample_imbalanced(data, class_list, size):
    # num_classes = np.unique(class_list)
    #
    # if isinstance(data, list):
    #     data_size = len(data)
    # else:
    #     data_size = data.shape[0]
    #
    # num_samples = check_size_parameter(size, data_size)
    # if num_samples == -1:
    #     raise Exception(f"Invalid value was given for parameter \'size\'")

