from skimage.feature import greycomatrix, greycoprops
from definitions import definitions as df
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from features import glcm_F


s = 55
d = 8.0

clf = pickle.load(open("res_Random Forest_20190716-17.clf", "rb"))

img = cv2.imread(df.imatges + "Clip_Clip_Mosaic_orto56_STPH_D.tif", -1)
marjada = cv2.imread(df.imatges + "Marjades_Clip_Clip_Mosaic_orto56_STPH_D_meu.tif", -1)
alcada = cv2.imread(df.imatges + "MDP_Clip_Clip_Mosaic_orto56_STPH_D.tif", -1)

img = img / d
img = img.astype(np.uint8)

ih, iw = (55, 55)
h, w, = (3000, 3000) #img.shape
print(h, w)
resultat = np.zeros((h,w))
resultat[:] = -1

features = np.zeros((9)) # TODO CUTREEE
for i in range(ih + s//2, (h+ih)-(s//2), 1):
    print(i)
    for j in range(iw + s//2, (w+iw)-(s//2), 1):

        xs = []
        submatrix = img[i-(s//2):i + (s//2), j-(s//2):j + (s//2)]
        submarjada = marjada[i-(s//2):i + (s//2), j-(s//2):j + (s//2)]
        subalcada = alcada[i-(s//2):i + (s//2), j-(s//2):j + (s//2)]

        glcm_features = glcm_F(submatrix, angles=df.angles, distances=df.dist, prop=df.prop, d=d)

         = glcm_features
        features[-1] = (np.count_nonzero(submarjada[:]) / submarjada.size)
        features[-2] = np.mean(subalcada[:])

        tipus = clf.predict(features.reshape(1, -1))

        resultat[i-ih, j-iw] = tipus

plt.imshow(resultat)
plt.imshow(img[ih: ih + h, iw: iw + w ])

plt.subplot(1,2,1), plt.imshow(img[ih: ih + h, iw: iw + w ], cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(resultat)
plt.title('Classificat'), plt.xticks([]), plt.yticks([])

plt.show()
