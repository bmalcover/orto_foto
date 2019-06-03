from skimage.feature import greycomatrix, greycoprops
from definitions import definitions as df
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from features import glcm_F, HOG, pca_F


s = 13
d = 4.0

clf = pickle.load(open("res_Random Forest_20190523-18.clf", "rb" ))

img = cv2.imread(df.imatges + "classificacio_01.png")
img = img[:, :, 0]

img = img / d
img = img.astype(np.uint8)


h, w, = img.shape
print(h, w)

for i in range(0, h, s):
    print(i)
    for j in range(0, w, s):
        submatrix = img[i:i + s, j:j + s]

        img = img[:, :, 0] / d

        img = img.astype(np.uint8)

        glcm_features = glcm_F(img, angles=df.angles, distances=df.dist, prop=df.prop)
        HOG_features = HOG(img, s, 9)

        features = np.zeros((glcm_features.shape[0] + 9))
        features[0: glcm_features.shape[0]] = glcm_features
        features[glcm_features.shape[0]:] = HOG_features

        xs.append(features)
        xs = np.asarray(xs)

        if df.config["min_max"]:
            min_max_scaler = StandardScaler()
            xs = min_max_scaler.fit_transform(xs)

        resultat = clf.predict(xs)


        if resultat == "agricola":
            img[i:i + s, j:j + s] = 0
        elif resultat ==  "forestal_arbrat":
            img[i:i + s, j:j + s] = 128
        else:
            img[i:i + s, j:j + s] = 255

plt.imshow(img)
plt.show()
