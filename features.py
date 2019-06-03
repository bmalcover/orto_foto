import numpy as np
from skimage.feature import greycomatrix, greycoprops, hog
from sklearn.decomposition import PCA


def glcm_F(image, angles, distances, prop, symmetric=True, normed=False):

    n_features = len(angles) * len(distances)

    features = np.zeros(((n_features * len(prop)) + 2))
    glcm = greycomatrix(image, distances=distances, angles=angles, symmetric=symmetric, normed=normed)

    for idx, p in enumerate(prop):  # obtenim features de la matriu GLCM
        f = greycoprops(glcm, prop=p)
        features[(idx * n_features): (idx + 1) * n_features] = f.flatten()

    features[(n_features * len(prop)) - 2] = np.mean(glcm[:, :])  # mean
    features[(n_features * len(prop)) - 1] = np.std(glcm[:, :])  # sd

    return features


def HOG(image, px, orientations=9, block_norm="L1"):

    features = hog(image, orientations=orientations, pixels_per_cell=(px, px), cells_per_block=(1, 1), transform_sqrt=True,
                      block_norm=block_norm, feature_vector=True)

    return features


def pca_F(train_set, test_set, percent=0.95):

    pca = PCA(n_components=train_set.shape[1])

    pca.fit(train_set)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    r = np.where(cumsum > percent)
    pca = PCA(n_components=r[0][0])

    x_train = pca.fit_transform(train_set)
    x_test = pca.transform(test_set)

    print("Train", x_train.shape[0], "Test", x_test.shape[0])

    return x_train, x_test
