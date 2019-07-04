from skimage.feature import greycomatrix, greycoprops
from definitions import definitions as df
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from features import glcm_F, HOG, pca_F


s = 55
d = 32.0

clf = pickle.load(open("res_Random Forest_20190703-17.clf", "rb"))

img = cv2.imread(df.imatges + "classificacio_01.png", -1)
img = img[:, :, 0]
resultat = img.copy()

img = img / d
img = img.astype(np.uint8)


h, w, = img.shape
print(h, w)

for i in range(s//2, h-(s//2), 1):
    print(i)
    for j in range(s//2, w-(s//2), 1):

        xs = []
        submatrix = img[i:i + s, j:j + s]
        glcm_features = glcm_F(submatrix, angles=df.angles, distances=df.dist, prop=df.prop, d=d)

        features = np.zeros((glcm_features.shape[0]))  # + 9))
        features[0: glcm_features.shape[0]] = glcm_features

        tipus = clf.predict(features.reshape(1, -1))

        if tipus == "agricola":
            resultat[i, j] = 0
        elif tipus == "forestal_arbrat":
            resultat[i, j] = 1
        else:
            resultat[i, j] = 2

plt.imshow(resultat)
plt.show()
