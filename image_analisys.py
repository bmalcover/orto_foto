from definitions import definitions as df
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from features import glcm_F


def calcul(init, size):
    s = 55
    d = 2.0

    clf = pickle.load(open(df.clf + os.altsep + "res_Random Forest_20190722-15.clf", "rb"))

    img = cv2.imread(df.imatges + "Clip_Clip_Mosaic_orto56_STPH_D.tif", -1)
    marjada = cv2.imread(df.imatges + "Marjades_Clip_Clip_Mosaic_orto56_STPH_D_meu.tif", -1)
    alcada = cv2.imread(df.imatges + "MDP_Clip_Clip_Mosaic_orto56_STPH_D.tif", -1)

    img = img / d
    img = img.astype(np.uint8)

    ih, iw = init
    h, w = size

    print(h, w)

    features = np.zeros((w*h, 8))  # TODO CUTREEE
    glcm_features = np.zeros(6)
    count = 0
    for i in range(ih, (ih + h), 1):
        print(i)
        for j in range(iw, (iw + w), 1):
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

    # plt.subplot(1, 2, 1), plt.imshow(img[ih: ih + h, iw: iw + w], cmap='gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(resultat)
    # plt.title('Classificat'), plt.xticks([]), plt.yticks([])
    #
    # plt.show()
