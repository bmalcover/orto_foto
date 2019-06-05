from skimage.feature import greycomatrix, greycoprops
from definitions import definitions as df
import pickle
import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from features import glcm_F, HOG, pca_F


s = 18
d = 1.0

clf = pickle.load(open("res_Gradient Boosting_20190604-18.clf", "rb"))

img = cv2.imread(df.imatges + "classificacio_01.png")
img = img[:, :, 0]

img = img / d
img = img.astype(np.uint8)


h, w, = img.shape
print(h, w)

for i in range(0, h, s):
    print(i)
    for j in range(0, w, s):

        xs = []
        submatrix = img[i:i + s, j:j + s]

        #glcm_features = glcm_F(submatrix, angles=df.angles, distances=df.dist, prop=df.prop)
        HOG_features, himg = HOG(submatrix, s, 9)

        img[i:i + s, j:j + s] = himg

hogImage = exposure.rescale_intensity(himg, out_range=(0, 255))
hogImage = hogImage.astype("uint8")


plt.imshow(img, cmap="Greys")
plt.show()
