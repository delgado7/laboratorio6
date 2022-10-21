from tokenize import group
import cv2
import numpy as np
from os import walk
import os, random
import time
import shutil
import math
from matplotlib import pyplot as plt

white = [255, 255, 255]
black = [0, 0, 0]

histograms = [[], [], [], [], [], [], [], [], [], []] # Lista que guardará el 70% de los histogramas por cada dígito

meanVecs = [] # Lista que contiene todos los valores promedio tomando en ceanta el 70% de los histogramas
varianceVecs = [] # Lista que contiene las varianzas de cada grupo de píxeles tomando en cuenta el 70% de los histogramas

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

graphsPath = "./barGraphs/"

entries = []

def getImageFiles(path):

    filenames = next(walk(path))[2]

    imagefiles = [path+f for f in filenames if (f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"))]

    return imagefiles


def saveImage(image, path, name, num, group):
    #print("saving image: ",path,name)
    imageName = str(name)+".png"

    if imageName in entries:
        histogram = generateHistogram(image)
        histograms[5*group+num].append(histogram)
    
    cv2.imwrite(path+imageName, image)


def extractIndividualNumbers(image, path, counter, group, num):
        
    resized = cv2.resize(image, (1156, 216))
    (h, w) = resized.shape[:2]
    stepW = w // 17
    stepH = h // 3

    for y in range(0,h, stepH):
        index = 0
        add = 0
        for x in range(0,w,stepW):
            crop_img = resized[y:y+stepH, x+add:x+stepW+add]
            
            crop_img = preprocessing(crop_img, group, num)
            crop_img = center(crop_img)
            saveImage(crop_img, path, counter, num, group)
            
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
    
    image[0:quantity, 0:w] = 255
    image[h-1, 0:w] = 255

    image[0:h, 0:quantity] = 255
    image[0:h, w-quantity:w] = 255

    return image

def preprocessing(img, group, number):

    (h, w) = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    l = (3,3)

    if(group == 0):
        if(number == 4):
            l = (5,5)
    else:
        if(number == 0):
            l = (5,5)
    img = cv2.GaussianBlur(img[0:h, 4:w-4], l, cv2.BORDER_DEFAULT)
    _ , img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    img, l = fillBlackLines(img, 0, 0)
    #img, _ = fillBlackLines(img, 1, l)

    img = whitePadding(img, 5)

    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #img_dilation = cv2.dilate(img, kernel, iterations=1)

    img2 = cv2.bitwise_not(closing)
    #showImage(img2, "og1")
    

    #"showImage(img2, "og1")
    

    contoursL = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #"print("cnt", contoursL)

    contours = contoursL[0] if len(contoursL) == 2 else contoursL[1]
    big_contour = max(contours, key=cv2.contourArea)
    
    
    #print(big_contour)


    # draw largest contour as white filled on black background as mask
    (h2, w2) = img2.shape[:2]
    mask = np.zeros((h2, w2), dtype=np.uint8)
    #showImage(mask, "mask1")
    cv2.drawContours(mask, [big_contour], 0, 255, -1)
    #showImage(mask, "mask2")

    # use mask to black all but largest contour
    result = img.copy()
    result[mask==0] = 255

    result = cv2.dilate(result, kernel, iterations=1)
    

    return result

def showImage(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def processSamples(groupId):

    groupA = []
    groupB = []
    
    if(groupId == 0):
        groupA = getImageFiles("./groupA/")
    else:
        groupB = getImageFiles("./groupB/")
    
    #print(groupA)
    #print(groupB)
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
            #if(l==1 and groupId==1):
                #showImage(crop_img, "crop")
            #kernel = np.ones((1,1), np.uint8)
            #crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE, kernel)
            groupCounts[groupId][l] = extractIndividualNumbers(crop_img, groupPaths[groupId][l], groupCounts[groupId][l], groupId, l)
            #print(groupACounts[l], " ", groupAPaths[l])
            excessB += excess
            moveB += move

def verticalHistogram(image, clusterQuantity):

    height, width = image.shape
    histogram = np.zeros(clusterQuantity)

    for i in range(0, height):
        currentCluster = 0

        for j in range(0, width):
            if j%4==0 and j!=0:
                currentCluster += 1
            
            if image[i][j] == 0:
                histogram[currentCluster] += 1
            
    return histogram

def horizontalHistogram(image, clusterQuantity):

    height, width = image.shape
    histogram = np.zeros(clusterQuantity)

    currentCluster = 0
    for i in range(0, height):
        if i%4==0 and i!=0:
            currentCluster += 1

        for j in range(0, width):
            if image[i][j] == 0:
                histogram[currentCluster] += 1
            
    return histogram

def saveHistogram(valuesRange, histogram, currentNumber):
    numberNames = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    plt.gcf().clear() 
    plt.figure(figsize=(12, 5))
    plt.bar(valuesRange, histogram, align="center")
    plt.xticks(valuesRange, )

    plt.xlabel("Cluster")
    plt.ylabel("Pixel quantity")

    plt.savefig("./barGraphs/"+str(numberNames[currentNumber])+".png")

def generateHistogram(image):
    height, width = image.shape

    vClusterQuantity = math.ceil(width/4)
    hClusterQuantity = math.ceil(height/4)

    verticalHist = verticalHistogram(image, vClusterQuantity)
    horizontalHist = horizontalHistogram(image, hClusterQuantity)

    histogram = np.concatenate((verticalHist, horizontalHist))

    return histogram

def getBarGraphics():
    paths = [zeroesPath, onesPath, twosPath, threesPath, foursPath, fivesPath, sixesPath, sevensPath, eightsPath, ninesPath]

    for i in range(10):

        imageName = os.listdir(paths[i])[0]
        image = cv2.imread(paths[i]+imageName, cv2.IMREAD_GRAYSCALE)

        histogram = generateHistogram(image)
        saveHistogram(range(len(histogram)), histogram, i)

def loadImage(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image

def getImageLimits(image, color):
    (h, w) = image.shape[:2]

    upper = 0
    lower = h-1
    left = 0
    right = w-1
    
    for y in range(h):
        if min(image[y,0:w]) == color:
            upper = y
            break
    
    for y in range(h-1, -1, -1):
        if min(image[y,0:w]) == color:
            lower = y
            break
        
    for x in range(w):
        if min(image[0:h,x]) == color:
            left = x
            break
    
    for x in range(w-1,-1,-1):
        if min(image[0:h,x]) == color:
            right = x
            break
        
    
    return upper, lower, left, right

def center(image):

    upper, lower, left, right = getImageLimits(image, 0)

    #print(upper, lower, left, right)

    img_crp = image[upper:lower, left:right]

    bordersize = 10
    img_crp = cv2.copyMakeBorder(img_crp,top=bordersize,bottom=bordersize,left=bordersize,right=bordersize,borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])

    img_crp = cv2.resize(img_crp, (60, 64))
    img_crp = cv2.threshold(img_crp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)[1]

    #img_crp = cv2.blur(img_crp,(5,5))

    #kernel = np.ones((5,5), np.uint8)
    #img_crp = cv2.dilate(img_crp, kernel, iterations=1)

    return img_crp

# Función que procesa todos los histogramas de entrenamiento y calcula su media y varianza
def getHistMeanVariance():
    for i in range(10):
        meanVec = np.array(histograms[i])
        varianceVec = np.array(histograms[i])

        meanVec = np.transpose(meanVec)
        meanVec = np.mean(meanVec, 1)
        meanVec = np.ndarray.tolist(meanVec)
        meanVecs.append(meanVec)

        varianceVec = np.transpose(varianceVec)
        varianceVec = np.var(varianceVec, 1)
        varianceVec = np.ndarray.tolist(varianceVec)
        varianceVecs.append(varianceVec)

    file = open("modelo.txt", "w")
    file.write(str([meanVecs]+[varianceVecs]))
    file.close()

def seventy():
    for i in range(534):
        chosen = False
        while not chosen:
            toBeAdded = str(random.randint(0,764))+".png"
            if toBeAdded not in entries:
                entries.append(toBeAdded)
                chosen = True
    
    file = open("set_entrenamiento.txt", "w")
    file.write(str(entries))
    file.close()

def main():

    dirs = [zeroesPath, onesPath, twosPath, threesPath, foursPath, fivesPath, sixesPath, sevensPath, eightsPath, ninesPath, graphsPath]

    for d in dirs:
        if(os.path.exists(d)):
            shutil.rmtree(d)
        os.makedirs(d)

    seventy()

    processSamples(0)
    processSamples(1)

    getBarGraphics()

    getHistMeanVariance()

    return

main()

def fillBlackLines2(image, direction:int, lastY:int):
    (h, w) = image.shape[:2]
    if (direction == 0):
        for y in range(h):
            if(max(image[y, 0:w]) == 0):
                image[y, 0:w] = 255
                lastY = y

        for y in range(0,int(int(h*0.1))):
            if(sum(image[y, 0:w])/255 < 0.25*w):
                image[y, 0:w] = 255
                lastY = y

        for y in range(int(h*0.85), h):
            if(sum(image[y, 0:w])/255 < 0.25*w):
                image[y, 0:w] = 255
                lastY = y

    else:
        for x in range(w):
            k = lastY
            k+=1
            if(k < h):
                if(max(image[k:h, x]) == 0):
                    image[k:h, x] = 255

        for x in range(0,int(w*0.25)):
            if(sum(image[0:h, x])/255 < 0.10*h):
                image[0:h, x] = 255

        for x in range(int(w*0.75), w):
            if(sum(image[0:h, x])/255 < 0.10*h):
                image[0:h, x] = 255

    return image, lastY

def a(img):
    img2 = cv2.bitwise_not(img)
    #showImage(img2, "og1")
    

    #"showImage(img2, "og1")
    

    contoursL = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #"print("cnt", contoursL)

    contours = contoursL[0] if len(contoursL) == 2 else contoursL[1]
    big_contour = max(contours, key=cv2.contourArea)
    
    
    #print(big_contour)


    # draw largest contour as white filled on black background as mask
    (h2, w2) = img2.shape[:2]
    mask = np.zeros((h2, w2), dtype=np.uint8)
    #showImage(mask, "mask1")
    cv2.drawContours(mask, [big_contour], 0, 255, -1)
    #showImage(mask, "mask2")

    # use mask to black all but largest contour
    result = img.copy()
    result[mask==0] = 255