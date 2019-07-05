from skimage.feature import greycomatrix, greycoprops
from definitions import definitions as df
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from features import glcm_F


s = 55
d = 16.0

clf = pickle.load(open("res_Random Forest_20190704-17.clf", "rb"))

img = cv2.imread(df.imatges + "prova_A.tif", -1)
img = img[:, :, 0]
resultat = np.zeros(img.shape)

img = img / d
img = img.astype(np.uint8)
resultat[:] = -1


h, w, = img.shape
print(h, w)

for i in range(s//2, h-(s//2), 1):
    print(i)
    for j in range(s//2, w-(s//2), 1):

        xs = []
        submatrix = img[i-(s//2):i + (s//2), j-(s//2):j + (s//2)]
        glcm_features = glcm_F(submatrix, angles=df.angles, distances=df.dist, prop=df.prop, d=d)

        tipus = clf.predict(glcm_features.reshape(1, -1))

        resultat[i, j] = tipus

plt.imshow(resultat)
plt.show()
