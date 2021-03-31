from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
    """
    helper function to check data parameter and return length/size of data
    :param data: data set given by the user
    :return: integer representing length or shape[0] of data
    """
    if isinstance(data, list):
        data_size = len(data)
    else:
        data_size = data.shape[0]
    if data_size == 0:
        raise Exception(f'An empty collection or list was given for the parameter\'data\'.')
    return data_size


def avg(_list):
    """
    takes avg of list of values
    :param _list: given list
    :return: float avg
    """
    return float(sum(_list) / len(_list))


def normalize(_list):
    """
    normalize a list of numbers
    :param _list: list of numbers
    :return: list normalized
    """
    _sum = sum(_list)
    return[i/_sum for i in _list]


def cluster_dist(point, cluster_center):
    """
    given two points returns Euclidean distance between the points
    :param point: point in n-dimensional space
    :param cluster_center: point in n-dimensional space
    :return: float distance between the two
    """
    sum_ = sum([(point[i] - cluster_center[i]) ** 2 for i in range(len(point))])
    return sum_ ** .5


def infer_k(data):
    """
    iterates from 2-10 testing how many clusters best fit the given data
    (uses silhouette index to guage)
    :param data: given data
    :return: best fit k
    """
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_coefficients.append(score)

    return silhouette_coefficients.index(max(silhouette_coefficients)) + 2

