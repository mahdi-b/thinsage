from thinsage_multifeature import MultiFeature
from sklearn.decomposition import PCA


class MultiFeatureImages:
    def __init__(self):
        self.multi = MultiFeature()

    def multifeature_image(self, data, size, reduc=None, expand=None, k=None, inflate_k=2):
        """
        pca:
        https: // scikit - learn.org / stable / modules / generated / sklearn.decomposition.PCA.html  # sklearn.decomposition.PCA.transform
        I assume data is array - like of images where each "image" is 2d array-like or flattened
        to 1d array-like
        :param data:
        :param size:
        :param reduc:
        :param expand:
        :param k:
        :param inflate_k:
        :return:
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
        reduc_sample = self.multi.multifeature_subsample_prob(data=reduc_data, size=size, k=k, inflate_k=inflate_k)

        # transform subsampled images back into original form
        return expand(reduc_sample)
