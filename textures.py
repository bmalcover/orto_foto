import os
import cv2
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from definitions import definitions
from classifiers import classificadors

from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)

#prop = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]




resultats = {}

for classificador in classificadors:

    resultats[classificador["title"]] = []

    for d in range(len(definitions.divisions)):
        resultats[classificador["title"]].append([])

print(resultats)

for size in definitions.sizes:  # De cada mida volem obtenir totes les imatges de tots els datasets

    xs = []
    y = []
    print("AVALUAM : " + str(size))
    print("#######################################")
    for ts in definitions.tipus_sol.keys():  # per cada conjunt d'entrenament

        nom_path = definitions.path + "\\" + ts + "\\" + str(size) + "\\"

        for image_name in (os.listdir(nom_path)):

            img = cv2.imread(nom_path + image_name)
            img = img[:, :, 0]

            glcm = greycomatrix(img, distances=definitions.dist, angles=definitions.angles, symmetric=True, normed=False)

            n_features = len(definitions.angles) * len(definitions.dist)
            m = np.zeros((n_features * len(definitions.prop)))

            for idx, p in enumerate(definitions.prop):  # obtenim features de la matriu GLCM
                f = greycoprops(glcm, prop=p)
                m[(idx*n_features): (idx + 1) * n_features] = f.flatten()
            # Conjunts d'entrenament
            xs.append(m)
            y.append(ts)

    X = np.asarray(xs)
    Y = np.asarray(y)

    print(X.shape, Y.shape)

    # Normalitzar les dades
    # min_max_scaler = StandardScaler()
    # X = min_max_scaler.fit_transform(X)

    for idx, d in enumerate(definitions.divisions):
        XX = X / d
        #XX = XX.astype(np.uint8)
        X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.25, random_state=23)

        if definitions.config["do_pca"] == True:

            pca = PCA(n_components=X_train.shape[1])

            principalComponents = pca.fit(X_train)

            cumsum = np.cumsum(pca.explained_variance_ratio_)

            r = np.where(cumsum > 0.9)

            pca = PCA(n_components=r[0][0])
            X_train = pca.fit_transform(X_train)
            print("Train")
            print(X_train.shape)

            X_test = pca.transform(X_test)
            print("Test")
            print(X_test.shape)

        # For amb diferents classificadors

        for classificador in classificadors:
            clf = GridSearchCV(classificador['clf'], param_grid=classificador['params'], verbose=0 )
            clf.fit(X_train, y_train)

            pred = clf.predict(X_test)
            print(classificador['title'])
            print(confusion_matrix(y_test, pred))
            print("##################################")
            print(classification_report(y_test, pred))
            _, recall, _, _ = precision_recall_fscore_support(y_test, pred, average="macro")

            resultats[classificador['title']][idx].append(recall)

for classificador in classificadors:
    plt.figure()
    for jdx, d in enumerate(definitions.divisions):

        plt.plot(resultats[classificador['title']][jdx], label= str(d))

    plt.title("Recall " + classificador['title'])
    plt.legend()
    plt.xticks(np.arange(len(definitions.sizes)), list(definitions.sizes))
    plt.xlabel("Patch size")
    plt.savefig("Recall " + classificador['title']  + ".png")
    plt.close()







