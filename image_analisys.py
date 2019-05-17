from skimage.feature import greycomatrix, greycoprops
from definitions import definitions as df
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


s = 9
d = 3.0


clf = pickle.load(open("res_Gradient Boosting_20190515-20.clf", "rb" ))

img = cv2.imread(df.imatges + "classificacio_02.png")
img = img[:, :, 0]

img = img / d
img = img.astype(np.uint8)


h, w, = img.shape
print(h, w)

for i in range(0, h, s):
    print(i)
    for j in range(0, w, s):
        submatrix = img[i:i + s, j:j + s]

        glcm = greycomatrix(submatrix, distances=df.dist, angles=df.angles, symmetric=True, normed=False)

        n_features = len(df.angles) * len(df.dist)
        m = np.zeros((n_features * len(df.prop)))
        xs =[]
        for idx, p in enumerate(df.prop):  # obtenim features de la matriu GLCM
            f = greycoprops(glcm, prop=p)
            m[(idx * n_features): (idx + 1) * n_features] = f.flatten()
        # Conjunts d'entrenament

        xs.append(m)
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
