import os
import pickle
from definitions import definitions as df
from osgeo import gdal
from image_analisys import calcul
import numpy as np

print(df.clf)

clf = pickle.load(open(df.clf + os.sep + "res_Random Forest_20190909-19.clf", "rb"))

idx = 0
xBSize = 1000
yBSize = 1000

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

for i in range(0, rows, yBSize):

    if i + yBSize < rows:
        numRows = yBSize
    else:
        numRows = rows - i
    for j in range(0, cols, xBSize):
        if j + xBSize < cols:
            numCols = xBSize
        else:
            numCols = cols - j

        data = band.ReadAsArray(j, i, numCols, numRows)
        data = np.flip(data)  # ALERTA!

        marge = marges.ReadAsArray(j, i, numCols, numRows)
        marge = np.flip(marge)

        altura = alcades.ReadAsArray(j, i, numCols, numRows)
        altura = np.flip(altura)

        if np.unique(data).shape[0] > 1:  # si no tota la matriu és igual


            #PROCESSAMENT

            resultat = calcul(data, marge, altura, clf)


            outfile = "res/" + str(idx) + ".tiff"
            out_data = out_driver.Create(str(outfile), yBSize, xBSize, 1, gdal.GDT_UInt16)
            out_data.GetRasterBand(1).WriteArray(resultat)

            # this is how to get the coordinate in space.
            posX = (px_w * i) + (rot1 * j) + x_offset
            posY = (rot2 * i) + (px_h * j) + y_offset
            # shift to the center of the pixel
            posX += px_w / 2.0
            posY += px_h / 2.0

            out_data.SetGeoTransform([posY, px_w, rot1, posX, rot2, px_h])
            out_data.SetProjection(projection)
            idx += 1
            print(idx)

print("pasisut")
