from definitions import definitions as df
import os
import pickle
import cv2
import numpy as np
from features import glcm_F


def calcul(img, marjada, alcada, clf, ofset):


    d = 2.0

    img = img / d
    img = img.astype(np.uint8)

    h, w = img.shape
    print("hi", end=" ")
    print(h, w)

    features = np.zeros((w*h, 8))  # TODO CUTREEE
    glcm_features = np.zeros(6)

    for i in range(ofset, h-ofset, 1):
        for j in range(ofset, w-ofset, 1):

            idx = i-ofset, i + ofset
            jdx = j-ofset, j + ofset
            count = np.ravel_multi_index(([i], [j]), (h, w))

            submatrix = img[jdx[0]: jdx[1], idx[0]:idx[1]]
            submarjada = marjada[jdx[0]: jdx[1], idx[0]:idx[1]]
            subalcada = alcada[jdx[0]: jdx[1], idx[0]:idx[1]]

            glcm_F(submatrix, angles=df.angles, features=glcm_features, distances=df.dist, prop=df.prop, d=d)

            features[count, 0: glcm_features.shape[0]] = np.copy(glcm_features)
            features[count, -1] = (np.count_nonzero(submarjada[:]) / submarjada.size)
            features[count, -2] = np.mean(subalcada[:])

    resultat = clf.predict(features)
    resultat = np.reshape(resultat, (w, h), order='F')
    resultat = np.copy(resultat[ofset:resultat.shape[0]-ofset, ofset:resultat.shape[1]-ofset])
    return resultat
