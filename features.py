import numpy as np
from skimage.feature import greycomatrix, greycoprops


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

