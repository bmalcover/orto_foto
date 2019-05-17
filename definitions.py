import os

class definitions:

    path = ""
    imatges = "imatges/"

    n_mostres = 6500

    config = {"do_pca": False, "min_max": False}

    if os.name == "nt":  # os.join
        path = "C:\\Users\\gabri\\Dropbox (Maisie)\\Ortofoto\\arees_entrenament\\png\\"
    else:
        path = "patates"

    prop = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    tipus_sol = {"agricola": ["e_agricola", "e_agricola_1"],
                 "forestal_arbrat" : ["e_forestal_arbrat", "e_forestal_arbrat_1" ],
             "forestal_no_arbrat": ["e_forestal_no_arbrat", "e_forestal_no_arbrat_1"]}

    sizes = [9] #range(3, 11, 2) #  range(3, 21, 2)

    divisions = [32.0, 16.0, 8.0, 4.0, 2.0, 1]

    angles = range(0, 181, 45)
    dist = range(1, 5)
