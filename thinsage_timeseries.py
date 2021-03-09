import numpy as np
import math
import random

from thinsage_subsample import check_size_parameter

"""
:TO-DO: nan values ?
sampling from time series.
we'll assume time series is given as list/array where indices are time interval. 
optional to include matching list of labels for the x axis
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


def timeseries_LTTB(data, threshold):
    """
    Return a downsampled version of data.
    Parameters
    ----------
    data: list of lists/tuples
        data must be formated this way: [[x,y], [x,y], [x,y], ...]
                                    or: [(x,y), (x,y), (x,y), ...]
    threshold: int
        threshold must be >= 2 and <= to the len of data
    Returns
    -------
    data, but downsampled using threshold

    """
    convert = False
    if type(data[0]) != tuple or type(data[0]) != list:
        convert = True
        data = [(x, y) for x, y in enumerate(data)]

    # Check if data and threshold are valid
    if not isinstance(data, list):
        raise Exception("data is not a list")
    if not isinstance(threshold, int) or threshold <= 2 or threshold >= len(data):
        raise Exception("threshold not well defined")
    for i in data:
        if not isinstance(i, (list, tuple)) or len(i) != 2:
            raise Exception("datapoints are not lists or tuples")

    # Bucket size. Leave room for start and end data points
    every = (len(data) - 2) / (threshold - 2)

    a = 0  # Initially a is the first point in the triangle
    next_a = 0
    max_area_point = (0, 0)

    sampled = [data[0]]  # Always add the first point

    for i in range(0, threshold - 2):
        # Calculate point average for next bucket (containing c)
        avg_x = 0
        avg_y = 0
        avg_range_start = int(math.floor((i + 1) * every) + 1)
        avg_range_end = int(math.floor((i + 2) * every) + 1)
        # typo
        avg_range_end = avg_range_end if avg_range_end < len(data) else len(data)

        avg_range_length = avg_range_end - avg_range_start

        # typo
        while avg_range_start < avg_range_end:
            avg_x += data[avg_range_start][0]
            avg_y += data[avg_range_start][1]
            avg_range_start += 1

        avg_x /= avg_range_length
        avg_y /= avg_range_length

        # Get the range for this bucket
        range_offs = int(math.floor((i + 0) * every) + 1)
        range_to = int(math.floor((i + 1) * every) + 1)

        # Point a
        point_ax = data[a][0]
        point_ay = data[a][1]

        max_area = -1

        while range_offs < range_to:
            # Calculate triangle area over three buckets
            area = math.fabs(
                (point_ax - avg_x)
                * (data[range_offs][1] - point_ay)
                - (point_ax - data[range_offs][0])
                * (avg_y - point_ay)
            ) * 0.5

            if area > max_area:
                max_area = area
                max_area_point = data[range_offs]
                next_a = range_offs  # Next a is this b
            range_offs += 1

        sampled.append(max_area_point)  # Pick this point from the bucket
        a = next_a  # This a is the next a (chosen b)

    sampled.append(data[len(data) - 1])  # Always add last

    if convert:
        ret = [sample[1] for sample in sampled]
        labels = [sample[0] for sample in sampled]
        return ret, labels

    return sampled
