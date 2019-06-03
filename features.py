import numpy as np
from skimage.feature import greycomatrix, greycoprops, hog


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


def HOG(image, px, block_norm="L1"):

    features = hog(image, orientations=4, pixels_per_cell=(px, px), cells_per_block=(1, 1), transform_sqrt=True,
                      block_norm=block_norm, feature_vector=True)

    return features