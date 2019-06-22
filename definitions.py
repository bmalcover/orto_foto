import os

class definitions:

    path = ""
    imatges = "imatges/"

    config = {"do_pca": False, "min_max": False, "n_mostres": 1000}

    if os.name == "nt":  # os.join
        path = "C:\\Users\\gabri\\Dropbox (Maisie)\\Ortofoto\\arees_entrenament\\png\\"
    else:
        path = "/home/biel/Dropbox (Maisie)/Ortofoto/arees_entrenament/png"

    #entropy
    prop = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    tipus_sol = {"agricola": ["e_agricola", "e_agricola_1"],
                 "forestal_arbrat": ["e_forestal_arbrat", "e_forestal_arbrat_1" ],
                 "forestal_no_arbrat": ["e_forestal_no_arbrat", "e_forestal_no_arbrat_1"]}

    sizes = [11] #range(3, 11, 2)

    divisions = [32.0, 16.0, 8.0, 4.0, 2.0, 1]

    angles = [0,180] #range(0, 181, 45)
    dist = [1] #range(1, 5)
