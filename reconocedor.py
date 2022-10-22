from xmlrpc.client import boolean
import cv2
import pathlib
import math
import numpy
import ast
import shutil
import os, random

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

thirties = []

dirs = [zeroesPath, onesPath, twosPath, threesPath, foursPath, fivesPath, sixesPath, sevensPath, eightsPath, ninesPath]

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

    realCounters = numpy.zeros((10))
    predictedCounters = numpy.zeros((10))

    for route in dirs:
        for num in thirties:
            realCounters[dirs.index(route)] += 1

            path = str(route+num)
            grayImage = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            histogram = generateHistogram(grayImage)

            falseValues = []
            booleanValues = []

            for count in range(10):
                booleanValues.append(numpy.sort(numpy.logical_and(histogram >= bottomLimit[count], histogram <= topLimit[count])))

            for isNum in booleanValues:
                _, counts = numpy.unique(isNum, return_counts=True)
                falseValues.append(counts[0])

            if dirs.index(route) == falseValues.index(min(falseValues)): predictedCounters[dirs.index(route)] += 1
    
    print(realCounters)
    print(predictedCounters)
    

def thirty():
    file = open(r"set_entrenamiento.txt", "r")
    training = file.readline()
    file.close()

    training = ast.literal_eval(training)

    for i in range(230):
        chosen = False
        while not chosen:
            toBeAdded = str(random.randint(0,764))+".png"
            if toBeAdded not in training and toBeAdded not in thirties:
                thirties.append(toBeAdded)
                chosen = True 
    
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

    bottomLimits = getBottomLimits(model, 0.95)
    topLimits = getTopLimits(model, 0.95)

    thirty()

    processNumbers(bottomLimits, topLimits)

main()




