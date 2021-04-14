from helper_functions import get_num_samples, get_data_size
from thinsage_multiclass import multiclass_subsample_prob
from sklearn.decomposition import PCA

def multiclass_image(data, size, reduc=None, expand=None, k=None, inflate_k=2):
    """
    pca :
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.transform
    I assume data is array-like of images where each "image" is 2d array-like or flattened to 1d
    :param data: array-like images, image is array-like
    :param reduc: function to reduce dimensionality of images
                if reduc is none default will be pca.fit_transform => pca.fit() and pca.transform
    :return: subsampled images, in their original form (original dimension)
    """
    # reduce dimensionality of each image
    if reduc is None:
        pca = PCA()
        reduc = pca.fit_transform
    if expand is None:
        if not pca:
            pca = PCA()
        expand = pca.inverse_transform
    reduc_data = reduc(data)

    # cluster and sample based on "new" images
    reduc_sample = multiclass_subsample_prob(reduc_data, size, k, inflate_k)

    # transform subsampled images back into original form
    return expand(reduc_sample)
