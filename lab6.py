import cv2
import time
import numpy as np
from os import walk

white = [255, 255, 255]
black = [0, 0, 0]

def getImageFiles(path):

    filenames = next(walk(path))[2]

    imagefiles = [path+f for f in filenames if (f.endswith(".jpg") or f.endswith(".png"))]

    return imagefiles

def paintVerticals(image, x:int, color:list):
    (h, _) = image.shape[:2]
    
    for y in range(h):
        #print(image[y,x])
        image[y, x] = color

    return image

def paintHorizontals(image, y:int, color:list):
    (_, w) = image.shape[:2]
    
    for x in range(w):
        #print(image[y,x])
        image[y, x] = color

    return image

def vectorsAreEqual(v1, v2):
    print(v1, v2)
    return v1[0] == v2[0] and v1[1] == v2[1] and v1[2] == v2[2]


def scanner(image, direction:int, ancho:int):
    # 0 = scan horizontal ->
    # 1 = scan vertical v

    (imgH, imgW) = image.shape[:2]

    whitePixelCount:int = 0
    foundX = 0

    if(direction == 0):
        scanResult = np.zeros((imgH, 1, 3), np.uint8)
        for x in range(imgW):
            whitePixelCount = 0
            for y in range(imgH):
                if(image[y, x] == 255):
                    whitePixelCount += 1
                scanResult[y, 0] = image[y, x]
            if(whitePixelCount > 100):
                image = paintVerticals(image, x, 150)
                foundX = x
                print("found x: "+str(x))
                break
            #print("scan in x: "+str(x)+" with "+str(whitePixelCount)+" white pixels")
    else:
        scanResult = np.zeros((1, imgH, 3), np.uint8)
        for y in range(imgH):
            whitePixelCount = 0
            for x in range(imgW):
                if(image[y, x] == 255):
                    whitePixelCount += 1
                scanResult[0, x] = image[y, x]
            if(whitePixelCount > ancho*0.85):
                print(ancho*0.85)
                image = paintHorizontals(image, y, 50)
                foundX = y
                print("found y: "+str(x))
                break


    return foundX

def scannerRL(image, direction:int, ancho:int):
    # 0 = scan horizontal ->
    # 1 = scan vertical v

    (imgH, imgW) = image.shape[:2]

    whitePixelCount:int = 0

    if(direction == 0):
        scanResult = np.zeros((imgH, 1, 3), np.uint8)
        for x in range(imgW-1, -1, -1):
            whitePixelCount = 0
            for y in range(imgH-1,-1,-1):
                if(image[y, x] == 255):
                    whitePixelCount += 1
                scanResult[y, 0] = image[y, x]
            if(whitePixelCount > 100):
                image = paintVerticals(image, x, 150)
                foundX = x
                break
    else:
        scanResult = np.zeros((1, imgH, 3), np.uint8)
        for y in range(imgH-1,-1,-1):
            whitePixelCount = 0
            for x in range(imgW-1,-1,-1):
                if(image[y, x] == 255):
                    whitePixelCount += 1
                scanResult[0, x] = image[y, x]
            if(whitePixelCount > 100):
                image = paintHorizontals(image, y, 50)
                foundX = y
                print("found y: "+str(x))
                break
            #print("scan in x: "+str(x)+" with "+str(whitePixelCount)+" white pixels")
            

    return foundX

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
    img = cv2.GaussianBlur(img[4:h-4, 4:w-4], (3,3), cv2.BORDER_DEFAULT)
    _ , img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    img, l = fillBlackLines(img, 0, 0)
    #img_dilation, _ = fillBlackLines(img_dilation, 1, l)

    img = whitePadding(img, 5)

    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    return img_dilation

def processSamples(groupId):

    amount0 = 0
    amount1 = 0
    amount2 = 0
    amount3 = 0
    amount4 = 0
    amount5 = 0
    amount6 = 0
    amount7 = 0
    amount8 = 0
    amount9 = 0

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

    groupA = []
    groupB = []
    
    if(groupId == 0):
        groupA = getImageFiles("./groupA/")
    else:
        groupB = getImageFiles("./groupB/")
    

    groupAPaths = [zeroesPath, onesPath, twosPath, threesPath, foursPath]
    groupBPaths = [fivesPath, sixesPath, sevensPath, eightsPath, ninesPath]

    groupACounts = [amount0, amount1, amount2, amount3, amount4]
    groupBCounts = [amount5, amount6, amount7, amount8, amount9]


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

    processSamples(0)
    processSamples(1)
    

            
    """
    ancho = 750
    alto = int(ancho * 1.414)

    src = cv2.resize(src, (ancho, alto))
    #src = cv2.GaussianBlur(src, (3, 3), 0)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src_gray, cv2.CV_16S, ksize=3)
    abs_dst = cv2.convertScaleAbs(dst)

    (h, w) = abs_dst.shape[:2]

    #(cX, cY) = (w // 2, h // 2)

    #M = cv2.getRotationMatrix2D((cX, cY), 0, 1.0)
    #rotated = cv2.warpAffine(abs_dst, M, (w, h))
    
    cv2.imshow("as", abs_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    x1 = scanner(abs_dst, 0, ancho)
    x2 = scannerRL(abs_dst, 0, ancho)

    abs_dst = abs_dst[0:h, x1:x2]
    cv2.imshow("as", abs_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("s")

    (h, w) = abs_dst.shape[:2]
    y1 = scannerRL(abs_dst, 1, w)

    print(y1)

    (h, w) = abs_dst.shape[:2]
    abs_dst = abs_dst[0:y1, 0:w]
    cv2.imshow("as", abs_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    minus = 0
    for l in range(2):
        (h, w) = abs_dst.shape[:2]
        #k = abs_dst[w-int(7)-minus:h-minus, 0:w]
        k = abs_dst[w-int(54)-minus:h-minus, 0:w]
        cv2.imshow("as", k)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        minus += (h//2 + 20)
    
    

    (h, w) = abs_dst.shape[:2]
    print(h, w)
    k = abs_dst[h-700:h, 0:w]
    cv2.imshow("as", k)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    """
    return


main()
