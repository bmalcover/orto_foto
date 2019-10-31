import os
import cv2
import pickle
import math
import numpy as np
from osgeo import gdal
from image_analisys import calcul
from definitions import definitions as df
from sentinelhub import pixel_to_utm
import matplotlib.pyplot as plt

print(df.clf)

clf = pickle.load(open(df.clf + os.sep + "res_Random Forest_20190909-19.clf", "rb"))
ofset = df.sizes[0] // 2

idx = 0
xBSize = 1020
yBSize = 1020

out_driver = gdal.GetDriverByName("GTiff")
gtif = gdal.Open(df.imatges + "Mosaic_orto56_STPH.tif")
marjades = gdal.Open(df.imatges + "marjades_STPH.tif")
mdp = gdal.Open(df.imatges + "Mosaic_MDP25_STPH_50cm.tif")

x_offset, px_w, rot1, y_offset, rot2, px_h = gtif.GetGeoTransform()
projection = gtif.GetProjection()
band = gtif.GetRasterBand(1)

marges = marjades.GetRasterBand(1)
alcades = mdp.GetRasterBand(1)

rows, cols = band.YSize, band.XSize  # Suposam que totes tenen el mateix size TODO comprovar

for i in range(ofset, rows, yBSize):

    if i + yBSize < rows:
        numRows = yBSize

        for j in range(ofset, cols, xBSize):

            if j + xBSize < cols:
                numCols = xBSize

                jj = j - ofset
                ii = i - ofset
                nC = numCols + (2*ofset)
                nR = numRows + (2*ofset)

                data = band.ReadAsArray(jj, ii, nC, nR)
                marge = marges.ReadAsArray(jj, ii, nC, nR)
                altura = alcades.ReadAsArray(jj, ii, nC, nR)

                if np.unique(data).shape[0] > 1:

                    resultat = calcul(data, marge, altura, clf, ofset)

                    outfile = "res/" + str(idx) + ".tiff"

                    out_data = out_driver.Create(str(outfile), yBSize, xBSize, 1, gdal.GDT_UInt16)
                    out_data.GetRasterBand(1).WriteArray(resultat)

                    posX, posY = pixel_to_utm(i, j, gtif.GetGeoTransform())

                    out_data.SetGeoTransform([posX, px_w, rot1, posY,  rot2, px_h])
                    out_data.SetProjection(projection)
                    idx += 1
                    print(idx)

print("pasisut")
