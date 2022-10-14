import cv2
import numpy as np
from os import walk
import os

white = [255, 255, 255]
black = [0, 0, 0]

def getImageFiles(path):

    filenames = next(walk(path))[2]

    imagefiles = [path+f for f in filenames if (f.endswith(".jpg") or f.endswith(".png"))]

    return imagefiles


def saveImage(image, path, name):
    #print("saving image: ",path,name)
    cv2.imwrite(path+str(name)+".png", image)


def extractIndividualNumbers(image, path, counter):
        
    resized = cv2.resize(image, (1156, 216))

    (h, w) = resized.shape[:2]

    stepW = w // 17
    stepH = h // 3

    for y in range(0,h, stepH):
        index = 0
        add = 0
        for x in range(0,w,stepW):
            crop_img = resized[y:y+stepH, x+add:x+stepW+add]

            crop_img = preprocessing(crop_img)
            saveImage(crop_img, path, counter)
            counter += 1
            index+=1

            if(index%4==0):
                add+=1

    return counter


def fillBlackLines(image, direction:int, lastY:int):
    (h, w) = image.shape[:2]
    
    if (direction == 0):
        for y in range(h):
            if(max(image[y, 0:w]) == 0):
                image[y, 0:w] = 255
                lastY = y
    else:
        for x in range(w):
            k = lastY
            k+=1
            if(k < h):
                if(max(image[k:h, x]) == 0):
                    image[k:h, x] = 255
    
    return image, lastY

def whitePadding(image, quantity:int):
    (h, w) = image.shape[:2]
    
    #image[0:quantity, 0:w] = 255
    #image[h-quantity:h, 0:w] = 255

    image[0:h, 0:quantity] = 255
    image[0:h, w-quantity:w] = 255

    return image

def preprocessing(img):

    (h, w) = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img[0:h, 4:w-4], (3,3), cv2.BORDER_DEFAULT)
    _ , img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    img, l = fillBlackLines(img, 0, 0)
    #img_dilation, _ = fillBlackLines(img_dilation, 1, l)

    img = whitePadding(img, 5)

    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    return img_dilation

zeroesPath = "./results/zeroes/"
onesPath = "./results/ones/"
twosPath = "./results/twos/"
threesPath = "./results/threes/"
foursPath = "./results/fours/"
fivesPath = "./results/fives/"
sixesPath = "./results/sixes/"
sevensPath = "./results/sevens/"
eightsPath = "./results/eights/"
ninesPath = "./results/nines/"

def processSamples(groupId):

    groupA = []
    groupB = []
    
    if(groupId == 0):
        groupA = getImageFiles("./groupA/")
    else:
        groupB = getImageFiles("./groupB/")
    

    groupAPaths = [zeroesPath, onesPath, twosPath, threesPath, foursPath]
    groupBPaths = [fivesPath, sixesPath, sevensPath, eightsPath, ninesPath]

    groupACounts = [0 for i in range(5)]
    groupBCounts = [0 for i in range(5)]


    groupPaths = [groupAPaths, groupBPaths]
    groupCounts = [groupACounts, groupBCounts]
    groups = [groupA, groupB]

    for path in groups[groupId]:

        src = cv2.imread(path)

        ancho = 850
        alto = int(ancho * 1.2)

        src = cv2.resize(src, (ancho, alto))

        (h,w) = src.shape[:2]

        moveB = 0
        excessB = 0
        move = 182
        excess = 28

        for l in range(5):
            bef = moveB+excessB
            crop_img = src[bef: bef + move, 0:w]
            groupCounts[groupId][l] = extractIndividualNumbers(crop_img, groupPaths[groupId][l], groupCounts[groupId][l])
            #print(groupACounts[l], " ", groupAPaths[l])
            excessB += excess
            moveB += move


def main():

    dirs = [zeroesPath, onesPath, twosPath, threesPath, foursPath, fivesPath, sixesPath, sevensPath, eightsPath, ninesPath]

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    processSamples(0)
    processSamples(1)
    
    return


main()
