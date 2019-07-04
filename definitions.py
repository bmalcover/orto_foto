import os
import numpy as np
class definitions:

    path = ""
    imatges = "imatges/"

    config = {"n_mostres": 500}

    if os.name == "nt":  # os.join
        path = "C:\\Users\\gabri\\Dropbox (Maisie)\\Ortofoto\\arees_entrenament\\png\\"
    else:
        path = "/home/biel/Dropbox (Maisie)/Ortofoto/arees_entrenament/png"

    #entropy
    prop = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    tipus_sol = {"agricola": ["e_agricola", "e_agricola_1"],
                 "forestal_arbrat": ["e_forestal_arbrat", "e_forestal_arbrat_1" ],
                 "forestal_no_arbrat": ["e_forestal_no_arbrat", "e_forestal_no_arbrat_1", "e_forestal_no_arbrat_2"]}

    sizes = [55] #range(3, 11, 2)

    divisions = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1]

    angles = [0] #range(0, 181, 45)
    dist = [1] #range(1, 5)
