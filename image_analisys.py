from skimage.feature import greycomatrix, greycoprops
from definitions import definitions as df
import pickle
import cv2
import copy
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from features import glcm_F, HOG, pca_F


s = 9
d = 64.0

#clf = pickle.load(open("res_Gradient Boosting_20190604-18.clf", "rb"))

img = cv2.imread(df.imatges + "classificacio_01.png", -1)
img = img[:, :, 0]
#
img = img / d
img = img.astype(np.uint8)



#
h, w, = img.shape
print(h, w)
imgs = []
for i in range(len(df.prop)+2):
    imgs.append(np.zeros(img.shape))


for i in range(0, h, s):
    print(i)
    for j in range(0, w, s):

        xs = []
        submatrix = img[i:i + s, j:j + s]

        glcm_features = glcm_F(submatrix, angles=df.angles, distances=df.dist, prop=df.prop, d=d)
        #HOG_features, himg = HOG(submatrix, s, 9)

        for t in range(len(df.prop)+2):

            imgs[t][i:i + s, j:j + s] = glcm_features[t]

# HOG_features, himg = HOG(img, s, 9)
#
# hogImage = exposure.rescale_intensity(himg, out_range=(0, 255))
# hogImage = hogImage.astype("uint8")

titles = copy.copy(df.prop)
titles.append("MEAN")
titles.append("SD")

for i in range(len(titles)):
    fig = plt.figure(i)
    plt.imshow(imgs[i])
    plt.title(titles[i])
    plt.colorbar()
    plt.show()
    plt.clf()

fig = plt.figure(33)
plt.imshow(img)
plt.title("full")
plt.colorbar()
plt.show()
