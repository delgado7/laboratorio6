import cv2
import pathlib
import math
import numpy
import ast

path = "./results"

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

def verticalHistogram(image, clusterQuantity):

    height, width = image.shape
    histogram = numpy.zeros(clusterQuantity)

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
    histogram = numpy.zeros(clusterQuantity)

    currentCluster = 0
    for i in range(0, height):
        if i%4==0 and i!=0:
            currentCluster += 1

        for j in range(0, width):
            if image[i][j] == 0:
                histogram[currentCluster] += 1
            
    return histogram

def generateHistogram(image):
    height, width = image.shape

    vClusterQuantity = math.ceil(width/4)
    hClusterQuantity = math.ceil(height/4)

    verticalHist = verticalHistogram(image, vClusterQuantity)
    horizontalHist = horizontalHistogram(image, hClusterQuantity)

    histogram = numpy.concatenate((verticalHist, horizontalHist))

    return histogram

def processNumbers(bottomLimit, topLimit):
    grayImage = cv2.imread("./results/ones/1.png", cv2.IMREAD_GRAYSCALE)

    histogram = generateHistogram(grayImage)

    isZero = numpy.logical_and(histogram >= bottomLimit[0], histogram <= topLimit[0])
    isOne = numpy.logical_and(histogram >= bottomLimit[1], histogram <= topLimit[1])
    isTwo = numpy.logical_and(histogram >= bottomLimit[2], histogram <= topLimit[2])
    isThree = numpy.logical_and(histogram >= bottomLimit[3], histogram <= topLimit[3])
    isFour = numpy.logical_and(histogram >= bottomLimit[4], histogram <= topLimit[4])
    isFive = numpy.logical_and(histogram >= bottomLimit[5], histogram <= topLimit[5])
    isSix = numpy.logical_and(histogram >= bottomLimit[6], histogram <= topLimit[6])
    isSeven = numpy.logical_and(histogram >= bottomLimit[7], histogram <= topLimit[7])
    isEight = numpy.logical_and(histogram >= bottomLimit[8], histogram <= topLimit[8])
    isNine = numpy.logical_and(histogram >= bottomLimit[9], histogram <= topLimit[9])

    if isZero.all(): return 0
    if isOne.all(): return 1
    if isTwo.all(): return 2
    if isThree.all(): return 3
    if isFour.all(): return 4
    if isFive.all(): return 5
    if isSix.all(): return 6
    if isSeven.all(): return 7
    if isEight.all(): return 8
    if isNine.all(): return 9

    
def getModel(modelName):
    file = open(modelName, "r")
    model = file.readline()
    file.close()

    return ast.literal_eval(model) # Convert string to python list

def getBottomLimits(model, scale):
    model = numpy.array(model)
    return model[0] - numpy.power(model[1], 1/2) * scale # model[0] = meanVec, model[1] = varianceVec

def getTopLimits(model, scale):
    model = numpy.array(model)
    return model[0] + numpy.power(model[1], 1/2) * scale # model[0] = meanVec, model[1] = varianceVec

def main():
    model = getModel("modelo.txt")

    bottomLimits = getBottomLimits(model, 2)
    topLimits = getTopLimits(model, 2)

    result = processNumbers(bottomLimits, topLimits)

    print(result)

main()




