import os
import cv2
import time
import copy
import numpy as np
import pickle
import random

import matplotlib.pyplot as plt

from features import glcm_F, HOG, pca_F
from definitions import definitions as df
from classifiers import classificadors


from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

np.random.seed(42)

max_recall = 0
best_clf = None
resultats = {}
config = None

random.seed(33)


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

            nom_path = df.path + os.sep + ts + os.sep + str(size) + os.sep

            mostres = df.config["n_mostres"]

            if mostres > len(os.listdir(nom_path)):
                mostres = len(os.listdir(nom_path))

            seleccionades = random.sample(os.listdir(nom_path), mostres)

            for image_name in seleccionades:

                img = cv2.imread(nom_path + image_name)
                img = img[:, :, 0] / d

                img = img.astype(np.uint8)

                glcm_features = glcm_F(img, angles=df.angles, distances=df.dist, prop=df.prop,d=d)
                #HOG_features = HOG(img, size, 9)

                features = np.zeros((glcm_features.shape[0]))# + 9))
                features[0: glcm_features.shape[0]] = glcm_features
                #features[glcm_features.shape[0]:] = HOG_features

                xs.append(features)
                y.append(ts)

        X = np.asarray(xs)
        Y = np.asarray(y)

        print(X.shape, Y.shape)

        if df.config["min_max"]:  # Normalitzar les dades

            min_max_scaler = StandardScaler()
            X = min_max_scaler.fit_transform(X)

        XX = X
        X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.25, random_state=23)

        if df.config["do_pca"]:

            X_train, X_test = pca_F(X_train, X_test, 0.9999)


        # For amb diferents classificadors

        for classificador in classificadors:
            clf = GridSearchCV(classificador['clf'], param_grid=classificador['params'], verbose=0, iid='warn')
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
                title = classificador['title']

            resultats[classificador['title']][divisio].append(recall)

timestr = time.strftime("%Y%m%d-%H")

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

f = open("res_" + title + "_" + timestr + ".clf", 'wb')
pickle.dump(best_clf, f)
f.close()

with open("parameters", 'a') as fw:

    fw.write(timestr + "     " + title)
    fw.write("\n")
    fw.write("recall: " + str(max_recall))
    fw.write("\n")
    fw.write(str(config[0]) + " - " + str(config[1]))
    fw.write("\n")
    fw.write("n_mostres: " + str(df.config["n_mostres"]))
    fw.write("\n")
    fw.write("##############################")
    fw.write("\n")




