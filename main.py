import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from definitions import definitions as df

# Programa que genera totes les matriuz nXn que no contenen el color blanc
# les mides son agafades de la literatura: 3,5,7,9,11,13,15,55

for tipus in df.tipus_sol.keys():

    if not os.path.isdir(df.path + tipus):

        os.makedirs(df.path + tipus)
        os.makedirs(df.path + tipus + "_marjades")
        os.makedirs(df.path + tipus + "_alcades")

    for nom in df.tipus_sol[tipus]:

        nom_marjades = df.capa_marjades(nom)
        nom_alcades = df.capa_alcades(nom)
        print(df.path + nom + ".tif")
        img = cv2.imread(df.path + nom + ".tif", -1)
        marjada = cv2.imread(df.path_marjades + nom_marjades, -1)
        alcada = cv2.imread(df.path_alcades + nom_alcades, -1)

        alcada = (alcada / np.amax(alcada[:])) * 255

        h, w = img.shape

        for s in df.sizes:

            contador = 0
            folder_path = df.path + tipus + os.altsep + str(s)
            folder_m_path = df.path + tipus + "_marjades" + os.altsep + str(s)
            folder_a_path = df.path + tipus + "_alcades" + os.altsep + str(s)

            print(folder_path)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)
                os.makedirs(folder_m_path)
                os.makedirs(folder_a_path)

                print(" Creant " + str(s) + " per " + nom)

            for i in range(0, h-s, 25):
                for j in range(0, w-s, 25):
                    submatrix = img[i:i+s, j:j+s]
                    submarjada = marjada[i:i+s, j:j+s]
                    subalcada = alcada[i:i+s, j:j+s]

                    if 256 not in submatrix:

                        cv2.imwrite(folder_path + os.altsep + str(contador) + ".bmp", submatrix)
                        cv2.imwrite(folder_m_path + os.altsep + str(contador) + ".bmp", submarjada)
                        cv2.imwrite(folder_a_path + os.altsep + str(contador) + ".bmp", subalcada)
                        contador += 1

