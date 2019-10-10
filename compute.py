import cv2
from definitions import definitions as df
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
'''
clf = pickle.load(open(df.clf + os.altsep + "res_Random Forest_20190909-19.clf", "rb"))

img = cv2.imread(df.imatges + "Clip_Clip_Mosaic_orto56_STPH_D.tif", -1)
marjada = cv2.imread(df.imatges + "Marjades_Clip_Clip_Mosaic_orto56_STPH_D_meu.tif", -1)
alcada = cv2.imread(df.imatges + "MDP_Clip_Clip_Mosaic_orto56_STPH_D.tif", -1)

'''



gtif = gdal.Open(df.imatges + "Mosaic_orto56_STPH.tif")
print(gtif.GetMetadata())
print(gtif.GetGeoTransform())

#img = cv2.imread(df.imatges + "Mosaic_orto56_STPH.tif", -1)
myarray = np.array(gtif.GetRasterBand(1).ReadAsArray())
plt.imshow(myarray)
plt.show()
print("pasisut")