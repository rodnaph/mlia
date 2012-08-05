
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def sortDistances(inX, dataSet):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    return distances.argsort()

def nearestClasses(sortedDistances, labels, k):
    classes={}
    for i in range(k):
        voteIlabel = labels[ sortedDistances[i] ]
        classes[ voteIlabel ] = classes.get( voteIlabel, 0 ) + 1
    return classes

def classify0(inX, dataSet, labels, k):
    sortedDistances = sortDistances(inX, dataSet)
    classes = nearestClasses(sortedDistances, labels, k)
    sortedClasses = sorted( classes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClasses[ 0 ][ 0 ]

def readlines(path):
    fr = open(path)
    return fr.readlines()

def file2matrix(filename):
    lines = readlines( filename )
    returnMat = zeros((len(lines),3))
    classLabelVector = []
    index = 0
    for line in lines:
        listFromLine = line.strip().split( '\t' )
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

