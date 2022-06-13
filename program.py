import cv2
import cv2.aruco as aruco

import numpy as np

import os
import math

import imutils

directory = "Aruco"
markers = {}

#dictionary to store id and filename as key and value pairs

#function to find aruco id
def findAruco(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()

    (corners,ids,rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters= arucoParam)

    return(ids[0][0])

files = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        image = cv2.imread(f,1)
        markers[findAruco(image)]=filename


#function to rotate all aruco to straight

def rotateAruco(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    cx = 0
    cy = 0
    slope = 0
    angle = 0
    (corners,ids,rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters= arucoParam)

    if len(corners)>0:
        ids = ids.flatten()
        for (markerCorner, markerId) in zip(corners,ids):
            corners = markerCorner.reshape(4,2)
            (topLeft, topRight, bottomLeft, bottomRight) = corners
            topLeft = (int(topLeft[0]),int(topLeft[1]))
            topRight = (int(topRight[0]),int(topRight[1]))
            bottomLeft = (int(bottomLeft[0]),int(bottomLeft[1]))
            bottomRight = (int(bottomRight[0]),int(bottomRight[1]))

            slope = (topRight[1]-topLeft[1])/(topRight[0]-topLeft[0])
            angle = math.atan(slope)*180/3.14


    rows = image.shape[0]
    cols = image.shape[1]
    m = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    d = cv2.warpAffine(image,m,(cols,rows))
    cropped_image = cv2.resize(d,(0,0),fx=0.5,fy=0.5)
    cv2.waitKey(0)
    new_image = cropped_image[:][35:,35:]
    updated_image = new_image[:][:228,:228]
    file = "Resized\\"+filename
    cv2.imwrite(file,updated_image)

for filename in os.listdir("Aruco"):
    f = os.path.join("Aruco", filename)
    if os.path.isfile(f):
        image = cv2.imread(f,1)
        files.append(filename)
        rotateAruco(image)

#calling the main image
main_image = cv2.imread("CVtask.jpg", 1)
main_image = cv2.resize(main_image, (0,0), fx=0.2, fy=0.2)
main_threshold = cv2.cvtColor(main_image,cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(main_threshold, 240, 255, cv2.THRESH_BINARY)

#finding contours in the main image
contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#list to store midpoints of various squares
points = []

#list to store angles of various squares
angles = []

for cnt in contours:

    approx = cv2.approxPolyDP(cnt, 0.05*cv2.arcLength(cnt, True), True)

    if len(approx)==4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = float(w)/h
        if (ratio>0.95) and (ratio<1.05):
            x1, y1 = approx[0][0][0], approx[0][0][1]
            x2, y2 = approx[1][0][0], approx[1][0][1]
            x3, y3 = approx[2][0][0], approx[2][0][1]
            x4, y4 = approx[3][0][0], approx[3][0][1]
            xc = float((x1+x3)/2)
            yc = float((y1+y3)/2)
            points.append([xc,yc])
            slope = float((y2-y1)/(x2-x1))
            angle = math.degrees(math.atan(slope))
            angles.append(angle)

#dictionary to relate id and color
id_value = {}

#relating id and square
for i in points:
    color = main_image[int(i[1])][int(i[0])]
    if color[0] == 0:
        id_value[3] = [int(i[0]),int(i[1])]
    elif color[0] == 210:
        id_value[4] = [int(i[0]),int(i[1])]
    elif color[0] == 79:
        id_value[2] = [int(i[0]),int(i[1])]
    elif color[0] == 9:
        id_value[1] = [int(i[0]),int(i[1])]

#sequence of id to be pasted
ids = [3,4,2,1]

for i in range(4):
    filename = "Resized/"+markers[ids[i]]
    aruco_1 = cv2.imread(filename,1)
    angle = angles[i]

    #Rotating the main image
    main_image = imutils.rotate_bound(main_image,-angle)

    #main image to grayscale
    actual = cv2.cvtColor(main_image,cv2.COLOR_BGR2GRAY)

    #detecting contours
    _, imag = cv2.threshold(actual, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(imag, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx)==4:
            x, y, w, h = cv2.boundingRect(approx)
            aruco_1 = cv2.resize(aruco_1,(w,h))
            ratio = float(w)/h
            ltx = approx[0][0][0]
            rtx = approx[1][0][0]
            ratio1 = abs(ltx-rtx)/w
            lty = approx[0][0][1]
            rty = approx[1][0][1]
            ratio2 = abs(lty-rty)/h
            x3, y3 = approx[2][0][0], approx[2][0][1]
            x4, y4 = approx[3][0][0], approx[3][0][1]
            xc = int((x1+x3)/2)
            yc = int((y1+y3)/2)
            if (ratio>0.95) and (ratio<1.05):
                if ((ratio1 > 0.90) and (ratio1 < 1.10)) or ((ratio2 > 0.90) and (ratio2 < 1.10)):
                    #covering area with white color
                    main_image[y:y+h,x:x+w]=(255,255,255)
                    #covering area with aruco
                    main_image[y:y+h,x:x+w]=aruco_1

    #Rotating main image back
    main_image = imutils.rotate_bound(main_image,angle)

#Cropping the main image
main_image=main_image[595:840,550:895]
main_image = cv2.resize(main_image, (0,0), fx=1.2, fy=1.2)

#Sharpening the main image
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
image_sharp = cv2.filter2D(src=main_image, ddepth=-1, kernel=kernel)

cv2.imshow('result_image', image_sharp)
cv2.imwrite("result_image.jpg",image_sharp)

cv2.waitKey(0)
cv2.destroyAllWindows()
