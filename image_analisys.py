from definitions import definitions as df
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from features import glcm_F


s = 55
d = 8.0

clf = pickle.load(open(df.clf + os.altsep + "res_Random Forest_20190716-17.clf", "rb"))

img = cv2.imread(df.imatges + "Clip_Clip_Mosaic_orto56_STPH_D.tif", -1)
marjada = cv2.imread(df.imatges + "Marjades_Clip_Clip_Mosaic_orto56_STPH_D_meu.tif", -1)
alcada = cv2.imread(df.imatges + "MDP_Clip_Clip_Mosaic_orto56_STPH_D.tif", -1)

img = img / d
img = img.astype(np.uint8)

ih, iw = (2000, 2000)
h, w, = (3000, 3000)  # img.shape
print(h, w)
resultat = np.zeros((h, w))
resultat[:] = -1

features = np.zeros((w*h, 9))  # TODO CUTREEE

for i in range(ih + s//2, (h+ih)-(s//2), 1):
    print(i)

    for j in range(iw + s//2, (w+iw)-(s//2), 1):
        idx = i-(s//2), i + (s//2)
        jdx = j-(s//2), j + (s//2)

        submatrix = img[idx[0]: idx[1], jdx[0]:jdx[1]]
        submarjada = marjada[idx[0]: idx[1], jdx[0]:jdx[1]]
        subalcada = alcada[idx[0]: idx[1], jdx[0]:jdx[1]]

        glcm_features = glcm_F(submatrix, angles=df.angles, distances=df.dist, prop=df.prop, d=d)

        features[j-iw, 0: glcm_features.shape[0]] = glcm_features
        features[j-iw, -1] = (np.count_nonzero(submarjada[:]) / submarjada.size)
        features[j-iw, -2] = np.mean(subalcada[:])

resultat = clf.predict(features)
resultat = np.reshape(resultat, (h, w))
plt.imshow(resultat)
plt.imshow(img[ih: ih + h, iw: iw + w])

plt.subplot(1, 2, 1), plt.imshow(img[ih: ih + h, iw: iw + w], cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(resultat)
plt.title('Classificat'), plt.xticks([]), plt.yticks([])

plt.show()
