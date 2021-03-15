def get_num_samples(size, data_size):
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
            num_samples = round((size * data_size))
    elif isinstance(size, int):
        if size >= data_size or size <= 0:
            raise Exception(f"Invalid value was given for parameter \'size\'")
        else:
            num_samples = size
    return num_samples


def get_data_size(data):
    if isinstance(data, list):
        data_size = len(data)
    else:
        data_size = data.shape[0]
    if data_size == 0:
        raise Exception(f'An empty collection or list was given for the parameter\'data\'.')
    return data_size


def avg(_list):
    return sum(_list) / len(_list)