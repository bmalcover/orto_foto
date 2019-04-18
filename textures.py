import os
import cv2
import time
import copy
import numpy as np
import pickle

import matplotlib.pyplot as plt

from definitions import definitions as df
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

max_recall = 0
best_clf = None
resultats = {}
config = None



for classificador in classificadors:

    resultats[classificador["title"]] = []

    for d in range(len(df.divisions)):
        resultats[classificador["title"]].append([])

for divisio, d in enumerate(df.divisions):

    for size in df.sizes:  # De cada mida volem obtenir totes les imatges de tots els datasets

        xs = []
        y = []
        print("AVALUAM : " + str(size))
        print("#######################################")
        for ts in df.tipus_sol.keys():  # per cada conjunt d'entrenament

            nom_path = df.path + "\\" + ts + "\\" + str(size) + "\\"

            for image_name in (os.listdir(nom_path)):

                img = cv2.imread(nom_path + image_name)
                img = img[:, :, 0] / d

                img = img.astype(np.uint8)

                glcm = greycomatrix(img, distances=df.dist, angles=df.angles, symmetric=True, normed=False)

                n_features = len(df.angles) * len(df.dist)
                m = np.zeros((n_features * len(df.prop)))

                for idx, p in enumerate(df.prop):  # obtenim features de la matriu GLCM
                    f = greycoprops(glcm, prop=p)
                    m[(idx*n_features): (idx + 1) * n_features] = f.flatten()
                # Conjunts d'entrenament
                xs.append(m)
                y.append(ts)

        X = np.asarray(xs)
        Y = np.asarray(y)

        print(X.shape, Y.shape)

        if df.config["min_max"]:
        # Normalitzar les dades
            min_max_scaler = StandardScaler()
            X = min_max_scaler.fit_transform(X)

        XX = X
        #XX = XX.astype(np.uint8)
        X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.25, random_state=23)

        if df.config["do_pca"] == True:

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

            if recall > max_recall:
                max_recall = recall
                best_clf = copy.deepcopy(clf)
                config = d, size

            resultats[classificador['title']][divisio].append(recall)

for classificador in classificadors:
    plt.figure()
    for jdx, d in enumerate(df.divisions):

        plt.plot(resultats[classificador['title']][jdx], label= str(d))

    plt.title("Recall " + classificador['title'])
    plt.legend()
    plt.xticks(np.arange(len(df.sizes)), list(df.sizes))
    plt.xlabel("Patch size")
    plt.savefig("Recall " + classificador['title'] + "_" + timestr  + ".png")
    plt.close()

    timestr = time.strftime("%Y%m%d-%H")
    f = open("res_" + classificador['title'] + "_" + timestr + ".clf", 'wb')
    pickle.dump(clf, f)

    print(config[0], " - ", config[1])




