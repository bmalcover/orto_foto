import os
import numpy as np


class definitions:

    path = ""
    clf = "clf"
    imatges = "imatges/"
    entrenament = "png/"
    resultats  = "resultats/"

    config = {"n_mostres": 1000}

    if os.name == "nt":  # os.join
        path = "entrenament/Rasters/Arees_entrenament/"
        path_marjades = "entrenament/Rasters/Clip_Marjades_Sectors/"
        path_alcades = "entrenament/Rasters/MDP/"
    else:
        path = "patata"

    #entropy
    prop = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    tipus_sol = {"agricola": ["arees_entrenament_C_agricola","arees_entrenament_D_agricola"],
                 "forestal_arbrat": ["arees_entrenament_C_forestal_arbrat","arees_entrenament_D_forestal_arbrat"],
                 "forestal_no_arbrat": ["arees_entrenament_C_forestal_NO_arbrat", "arees_entrenament_D_forestal_no_arbrat", "arees_entrenament_E_forestal_no_arbrat"]}



    def capa_marjades(nom):

        return "Marjades_Clip_Mosaic_orto56_STPH_C_1.tif"

    def capa_alcades(nom):

        return "MDP_Clip_Mosaic_orto56_STPH_C.tif"

    sizes = [55] #range(3, 11, 2)

    divisions = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1]

    angles = [0]#,45,90] #range(0, 181, 45)
    dist = [1] # range(1, 5)

