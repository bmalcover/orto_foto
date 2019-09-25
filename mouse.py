# import the necessary packages
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from image_analisys import calcul
from definitions import definitions as df

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:

        refPt.append((x, y))
        cropping = False

        cv2.rectangle(small_image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", small_image)


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        cv2.rectangle(small_image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", small_image)


# construct the argument parser and parse the arguments

image = cv2.imread(df.imatges + "Clip_Clip_Mosaic_orto56_STPH_D.tif", -1)
image = image.astype(np.uint8)
rsz = 12
small_image = cv2.resize(image, (image.shape[1]//rsz, image.shape[0]//rsz))


clone = small_image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", small_image)
    key = cv2.waitKey(30) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        small_image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:

    refPt = np.asarray(refPt)
    refPt *= rsz
    roi = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

    cv2.waitKey(25)
    cv2.destroyWindow("image")
    size = abs(refPt[0][0] - refPt[1][0]), abs(refPt[0][1] - refPt[1][1])

    resultat = calcul((refPt[0]), size) * 65
    resultat = resultat.astype(np.uint8)

    timestr = time.strftime("%Y%m%d-%H")
    cv2.imwrite(df.resultats + timestr + "_classificat.png", resultat)
    cv2.imwrite(df.resultats + timestr + "_original.png", roi)
