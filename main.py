import cv2
import os
import numpy as np
from definitions import definitions

# Programa que genera totes les matriuz nXn que no contenen el color blanc
# les mides son agafades de la literatura: 3,5,7,9,11,13,15

for tipus in definitions.tipus_sol.keys():

    if not os.path.isdir(definitions.path + tipus):
        os.makedirs(definitions.path + tipus)

    for nom in definitions.tipus_sol[tipus]:

        img = cv2.imread(definitions.path+ nom + ".png")

        img = img[:, :, 0]
        h, w = img.shape

        dst = np.zeros((int(h/1), int(w/1)))
        dst = cv2.resize(img, dst.shape)

        img = np.copy(dst)
        h, w = img.shape


        for s in definitions.sizes:

            contador = 0
            folder_path = definitions.path + tipus + "\\" + str(s)
            print(folder_path)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

                print(" Creant " +str(s) + " per " + nom)

            for i in range(0, h-s, s):
                for j in range(0, w-s, s):
                    submatrix = img[i:i+s, j:j+s]

                    if 255 not in submatrix and contador < 1000:
                        cv2.imwrite(folder_path + "\\" + str(contador) + ".png", submatrix)
                        contador += 1




