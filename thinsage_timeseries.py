import numpy as np
import scipy as sp
import random

from thinsage_subsample import check_size_parameter

"""
sampling from time series.
we'll assume time series is given as list/array where indices are time interval. 
optional to include matching list of labels for the x axis
if data is missing, np.nan or None object is expected in order to keep interval consistent
(np.nan for float/int data or None for Obj) ?
"""


def timeseries_random(data, size, x_labels=None):
    # check data parameter
    if isinstance(data, list):
        data_size = len(data)
    else:
        data_size = data.shape[0]
    if data_size <= 0:
        raise Exception(f'An empty collection or list was given for the parameter\'data\'.')

    # check size and get num samples to return
    num_samples = check_size_parameter(size=size, data_size=data_size)

    # subsampling with random

    indices = random.sample(range(data_size), num_samples)
    indices.sort()
    ret_list = [data[indices[i]] for i in range(num_samples)]
    if x_labels is None:
        labels = list(indices)
    else:
        labels = [x_labels[indices[i]] for i in range(num_samples)]
    return ret_list, labels


def timeseries_interval(data, k=0, size=0, x_labels=None):
    # takes either k or size
    # check data parameter
    if isinstance(data, list):
        data_size = len(data)
    else:
        data_size = data.shape[0]
    if data_size <= 0:
        raise Exception(f'An empty collection or list was given for the parameter\'data\'.')

    if not isinstance(k, int):
        raise Exception(f'Invalid type \'{type(k)}\' was given, expected \'int\'.')

    # default value
    if k == 0 and size == 0:
        size = .25
    elif k > 0 and size > 0:
        raise Exception(f'Expected either one or none of the two parameters; \'k\' and \'size\', not both.')

    if k < 1:
        num_samples = check_size_parameter(size=size, data_size=data_size)
        k = int(np.floor(data_size / num_samples))

    indices = range(0, data_size, k)

    ret_list = [data[indices[i]] for i in range(len(indices))]
    if x_labels is None:
        ret_labels = list(indices)
    else:
        ret_labels = [x_labels[indices[i]] for i in range(len(indices))]

    return ret_list, ret_labels


def timeseries_bucket_random(data, num_buckets, per_bucket=1, x_labels=None):
    # check data parameter
    if isinstance(data, list):
        data_size = len(data)
    else:
        data_size = data.shape[0]
    if data_size <= 0:
        raise Exception(f'An empty collection or list was given for the parameter\'data\'.')

    if num_buckets * per_bucket >= data_size:
        raise Exception(f'\'num_buckets\' parameter must be less than size of \'data\'.')

    buckets = []

    count = 0
    for i in range(num_buckets):
        if i == num_buckets - 1:
            bucket_size = data_size - count
        bucket_size = round((data_size - count) / (num_buckets - i))
        buckets.append(list(range(count, count + bucket_size)))
        count += bucket_size

    indices = []
    for bucket in buckets:
        sample = random.sample(range(bucket[0], bucket[len(bucket) - 1] + 1), per_bucket)
        sample.sort()
        indices += sample

    ret_list = [data[indices[i]] for i in range(len(indices))]
    if x_labels is None:
        ret_labels = list(indices)
    else:
        ret_labels = [x_labels[indices[i]] for i in range(len(indices))]

    return ret_list, ret_labels


def timeseries_sliding_window(data, w_size, lambd='avg', delta=.1, x_labels=None):
    # check data parameter
    # delta will be ratio/percentage of abs(max - min)
    if isinstance(data, list):
        data_size = len(data)
    else:
        data_size = data.shape[0]
    if data_size <= 0:
        raise Exception(f'An empty collection or list was given for the parameter\'data\'.')

    if delta <= 0 or not isinstance(delta, float) or delta >= 1:
        raise Exception('Invalid value was given for parameter \'delta\'.')

    delta_num = np.abs(max(data) - min(data)) * delta

    # include first val?
    indices = [0]
    window = [data[0]]
    for i in range(1, data_size):
        if lambd == 'avg':
            val = np.average(window)
        elif lambd == 'min':
            val = min(window)
        elif lambd == 'max':
            val = max(window)

        if np.abs(val - data[i]) >= delta_num:
            indices.append(i)

        if len(window) >= w_size:
            window.remove(window[0])

        window.append(data[i])

    ret_list = [data[indices[i]] for i in range(len(indices))]
    if x_labels is None:
        ret_labels = list(indices)
    else:
        ret_labels = [x_labels[indices[i]] for i in range(len(indices))]

    return ret_list, ret_labels
