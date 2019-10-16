from definitions import definitions as df
import os
import pickle
import cv2
import numpy as np
from features import glcm_F


def calcul(img, marjada, alcada, clf):

    s = df.sizes[0]
    d = 2.0

    img = img / d
    img = img.astype(np.uint8)

    ih, iw = 0, 0
    h, w = img.shape

    print(h, w)

    features = np.zeros((w*h, 8))  # TODO CUTREEE
    glcm_features = np.zeros(6)

    count = 0
    for i in range((s//2), (ih + h), 1):
        for j in range((s//2), (iw + w), 1):
            idx = i-(s//2), i + (s//2)
            jdx = j-(s//2), j + (s//2)

            submatrix = img[jdx[0]: jdx[1], idx[0]:idx[1]]
            submarjada = marjada[jdx[0]: jdx[1], idx[0]:idx[1]]
            subalcada = alcada[jdx[0]: jdx[1], idx[0]:idx[1]]
            glcm_F(submatrix, angles=df.angles, features=glcm_features, distances=df.dist, prop=df.prop, d=d)

            features[count, 0: glcm_features.shape[0]] = np.copy(glcm_features)
            features[count, -1] = (np.count_nonzero(submarjada[:]) / submarjada.size)
            features[count, -2] = np.mean(subalcada[:])

            count += 1

    resultat = clf.predict(features)
    resultat = np.reshape(resultat, (w, h), order='F')

    return resultat
