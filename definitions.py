import os

class definitions:

    config = { "do_pca": False}

    path = ""

    if os.name == "nt":  # os.join
        path = "C:\\Users\\gabri\\Dropbox (Maisie)\\Ortofoto\\arees_entrenament\\png\\"
    else:
        path = "patates"

    tipus_sol = {"agricola": ["e_agricola", "e_agricola_1"], "forestal_arbrat" : ["e_forestal_arbrat", "e_forestal_arbrat_1"],
             "forestal_no_arbrat": ["e_forestal_no_arbrat", "e_forestal_no_arbrat_1"]}

    sizes = range(3, 21, 2)
