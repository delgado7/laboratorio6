from re import X
from tokenize import group
import cv2
import numpy as np
from os import walk
import os, random
import time
import shutil
import math
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


white = [255, 255, 255]
black = [0, 0, 0]

histograms = [[], [], [], [], [], [], [], [], [], []] # Lista que guardará el 70% de los histogramas por cada dígito

meanVecs = [] # Lista que contiene todos los valores promedio tomando en ceanta el 70% de los histogramas
varianceVecs = [] # Lista que contiene las varianzas de cada grupo de píxeles tomando en cuenta el 70% de los histogramas

HuMoments = []

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

transformationsPath = "./transformations/"
huMomentsTablesPath = "./huMomentsTables/"
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
            crop_img = cv2.bitwise_not(crop_img)

            kernel = np.ones((3,3), np.uint8)
            #crop_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE, kernel)
            crop_img = cv2.erode(crop_img, kernel, iterations=1)

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

    contoursL = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contoursL[0] if len(contoursL) == 2 else contoursL[1]
    big_contour = max(contours, key=cv2.contourArea)
    

    # draw largest contour as white filled on black background as mask
    (h2, w2) = img2.shape[:2]
    mask = np.zeros((h2, w2), dtype=np.uint8)
    
    cv2.drawContours(mask, [big_contour], 0, 255, -1)

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

    img_crp = image[upper:lower, left:right]

    bordersize = 10
    img_crp = cv2.copyMakeBorder(img_crp,top=bordersize,bottom=bordersize,left=bordersize,right=bordersize,borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])

    img_crp = cv2.resize(img_crp, (60, 64))
    img_crp = cv2.threshold(img_crp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)[1]


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

def getHuMoments():
    paths = [zeroesPath, onesPath, twosPath, threesPath, foursPath, fivesPath, sixesPath, sevensPath, eightsPath, ninesPath]
    
    for i in range(10):

        imageName = os.listdir(paths[i])[0]
        image = cv2.imread(paths[i]+imageName, cv2.IMREAD_GRAYSCALE)

        moments = cv2.moments(image)
        huMoments = cv2.HuMoments(moments)
        
        for s in range(0,7):
            huMoments[s] = round(-1* math.copysign(1.0, huMoments[s]) * math.log10(abs(huMoments[s])), 6)

        getHuMomentsVariations(image, huMoments, i, paths[i]+imageName)
        
        HuMoments.append(huMoments.flatten())


    plt.clf()
    row_headers = ["Dígito: {0}".format(i) for i in range(10)]
    column_headers = ["H[{0}]".format(i) for i in range(7)]

    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    
    the_table = plt.table(cellText=HuMoments,
                        rowLabels=row_headers,
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=column_headers,
                        loc='center')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(6)
    plt.axis('off')
    plt.savefig(huMomentsTablesPath+"/HuMoments"+".png", dpi=400)



def getHuMomentsVariations(image, huMomentsImage, i, imagePath):
    row_headers = ["Original", "Rotación", "Desplazamiento", "Amplificada", "Reducida"]
    column_headers = ["H[{0}]".format(i) for i in range(7)]

    huMomentsVariation = []
    huMomentsVariation.append(huMomentsImage.flatten())

    a = complex(1.3, 0)
    b = complex(0,0)
    img_amplificada = transform(image, a, b)
    cv2.imwrite(transformationsPath+"/amplificada{0}.png".format(i), img_amplificada)

    img_amplificada = cv2.cvtColor(img_amplificada, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(img_amplificada)
    huMoments = cv2.HuMoments(moments)
    huMomentsVariation.append(getMoments(huMoments).flatten())
    
    img_amp = mpimg.imread(transformationsPath+"/amplificada{0}.png".format(i), cv2.IMREAD_GRAYSCALE)

    
    a = complex(0.6, 0)
    img_disminuida = transform(image, a, b)
    cv2.imwrite(transformationsPath+"/disminuida{0}.png".format(i), img_disminuida)

    img_disminuida = cv2.cvtColor(img_disminuida, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(img_disminuida)
    huMoments = cv2.HuMoments(moments)
    huMomentsVariation.append(getMoments(huMoments).flatten())
    
    img_dim = mpimg.imread(transformationsPath+"/disminuida{0}.png".format(i))

    a = complex(0.8, 0.6)
    b = complex(0, 0)
    img_rotada = transform(image, a, b)
    cv2.imwrite(transformationsPath+"/rotada{0}.png".format(i), img_rotada)

    img_rotada = cv2.cvtColor(img_rotada, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(img_rotada)
    huMoments = cv2.HuMoments(moments)
    huMomentsVariation.append(getMoments(huMoments).flatten())

    img_rot = mpimg.imread(transformationsPath+"/rotada{0}.png".format(i))

    a = complex(1, 0)
    b = complex(8, 8)
    img_desplazada = transform(image, a, b)
    cv2.imwrite(transformationsPath+"/desplazada{0}.png".format(i), img_desplazada)

    img_desplazada = cv2.cvtColor(img_desplazada, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(img_desplazada)
    huMoments = cv2.HuMoments(moments)
    huMomentsVariation.append(getMoments(huMoments).flatten())

    img_des = mpimg.imread(transformationsPath+"/desplazada{0}.png".format(i))

    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    
    the_table = plt.table(cellText=huMomentsVariation,
                        rowLabels=row_headers,
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=column_headers,
                        loc='center right')    

    (h,w) = image.shape[:2]
    dim = (math.trunc(h*3.5)-8, math.trunc(w*3.5))

  
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_rot = cv2.resize(img_rot, dim, interpolation = cv2.INTER_AREA)
    img_des = cv2.resize(img_des, dim, interpolation = cv2.INTER_AREA)
    img_amp = cv2.resize(img_amp, dim, interpolation = cv2.INTER_AREA)
    img_dim = cv2.resize(img_dim, dim, interpolation = cv2.INTER_AREA)

    plt.figimage(image, 657, 1430, zorder=3, cmap='gist_gray')
    plt.figimage(img_rot, 657, 1135, zorder=3)
    plt.figimage(img_des, 657, 837, zorder=3)
    plt.figimage(img_amp, 657, 540, zorder=3)
    plt.figimage(img_dim, 657, 247, zorder=3)

    cellDict = the_table.get_celld()
    for a in range(0,len(column_headers)):
        cellDict[(0,a)].set_width(.1)
        for b in range(1,len(huMomentsVariation)+1):
            cellDict[(b,a)].set_height(0.2)
            cellDict[(b,a)].set_width(0.1)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(4)
    
    plt.axis('off')
    plt.savefig(huMomentsTablesPath+"/HuMomentsVariation{0}".format(i) + ".png", dpi=400)
    

def getMoments(huMoments):
    for s in range(0,7):
        huMoments[s] = round(-1* math.copysign(1.0, huMoments[s]) * math.log10(abs(huMoments[s])), 6)
    return huMoments


"""         MAPEO       """


def pixelToPlane(h,w,x,y):
    return (x - w//2, h//2-y)

def planeToPixel(h,w,x,y):
    return (int(x + w//2), int(h//2-y))

def bilinearMap(a, b, c, d, z):
    w = (a*z + b) / (c*z + d) 
    return w

def inverseMapping(a, b, c, d, w):
    z1 = (-d*w+b)/(c*w-a)
    return z1

def bilinearMappingImg(image, a, b, c, d):
    (h, w) = image.shape[:2]

    (planeWHeight, planeWWidth) = (h, w)

    planeW = np.zeros((planeWHeight, planeWWidth, 3), np.uint8)

    for y in range(0, h):
        for x in range(0, w):

            (planarX, planarY) = pixelToPlane(h,w,x,y)

            mapped = bilinearMap(a, b, c, d, complex(planarX, planarY))

            realPartFromPoint = int(mapped.real)
            imagPartFromPoint = int(mapped.imag)

            (newImageX, newImageY) = planeToPixel(planeWHeight,planeWWidth,realPartFromPoint,imagPartFromPoint)

            

            if (newImageX >= 0 and newImageX < planeWWidth and newImageY >= 0 and newImageY < planeWHeight):
                imagePixel = image[y, x]
                planeW[newImageY, newImageX] = imagePixel

    return planeW

"""
Recuperación de pixeles perdidos utilizando el inverso 
"""
def fillMissingPixelsInverse(targetImage, referenceImage, a, b, c, d):

    (referenceH, referenceW) = referenceImage.shape[:2]
    
    #(ZPlaneW, ZplaneH, ZplaneXOffset, ZplaneYOffset) = getMapDimensions(referenceImage, a, b, c, d, 0)
    (targetH, targetW) = targetImage.shape[:2]
    
    for y in range(targetH):
        for x in range(targetW):

            if ( (targetImage[y, x].all() == 0)):

                (planarX, planarY) = pixelToPlane(targetH,targetW,x,y)

                mapped = inverseMapping(a, b, c, d, complex(planarX, planarY))

                realPartFromPoint = int(mapped.real)
                imagPartFromPoint = int(mapped.imag)

                (newImageX, newImageY) = planeToPixel(targetH,targetW,realPartFromPoint,imagPartFromPoint)

                
                (xInReferenceImage, yInReferenceImage) = (math.ceil(newImageX), math.ceil(newImageY))
                
                if ((0 <= xInReferenceImage < referenceW) and (0 <= yInReferenceImage < referenceH)):
                    targetImage[y, x] = referenceImage[yInReferenceImage,xInReferenceImage]

    return targetImage

def transform(image, a, b):
    img1 = bilinearMappingImg(image,a,b,0,1)
    img2 = fillMissingPixelsInverse(img1,image, a, b, 0, 1)

    return img2

def main():

    dirs = [zeroesPath, onesPath, twosPath, threesPath, foursPath, fivesPath, sixesPath, sevensPath, eightsPath, ninesPath, graphsPath, transformationsPath, huMomentsTablesPath]

    for d in dirs:
        if(os.path.exists(d)):
            shutil.rmtree(d)
        os.makedirs(d)

    seventy()

    processSamples(0)
    processSamples(1)

    getHuMoments()

    #getBarGraphics()

    #getHistMeanVariance()

    return

main()
