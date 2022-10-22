from xmlrpc.client import boolean
import cv2
import pathlib
import math
import numpy
import ast
import shutil
import os, random

thirties = []

def thirty():
    file = open(r"set_entrenamiento.txt", "r")
    training = file.readline()
    file.close()

    training = ast.literal_eval(training)

    for i in range(764):
        toBeAdded = str(i)+".png"
        if toBeAdded not in training:
            shutil.copy("./results/zeroes/"+toBeAdded, "C:/Users/Reyner/Desktop/results/zeroes/ze"+toBeAdded)
            shutil.copy("./results/ones/"+toBeAdded, "C:/Users/Reyner/Desktop/results/ones/on"+toBeAdded)
            shutil.copy("./results/twos/"+toBeAdded, "C:/Users/Reyner/Desktop/results/twos/tw"+toBeAdded)
            shutil.copy("./results/threes/"+toBeAdded, "C:/Users/Reyner/Desktop/results/threes/thr"+toBeAdded)
            shutil.copy("./results/fours/"+toBeAdded, "C:/Users/Reyner/Desktop/results/fours/fo"+toBeAdded)
            shutil.copy("./results/fives/"+toBeAdded, "C:/Users/Reyner/Desktop/results/fives/fiv"+toBeAdded)
            shutil.copy("./results/sixes/"+toBeAdded, "C:/Users/Reyner/Desktop/results/sixes/six"+toBeAdded)
            shutil.copy("./results/sevens/"+toBeAdded, "C:/Users/Reyner/Desktop/results/sevens/se"+toBeAdded)
            shutil.copy("./results/eights/"+toBeAdded, "C:/Users/Reyner/Desktop/results/eights/ei"+toBeAdded)
            shutil.copy("./results/nines/"+toBeAdded, "C:/Users/Reyner/Desktop/results/nines/ni"+toBeAdded)

thirty()