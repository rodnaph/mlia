
from numpy import *
from os import listdir

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

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

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

#
#  improving matches from a dating site
#

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix( './data/ch02/datingTestSet.txt' )
    normMat, ranges, minVals = autoNorm( datingDataMat )
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with %s, the real answer is %s" % (classifierResult, datingLabels[i])
        if ( classifierResult != datingLabels[i] ): errorCount += 1.0
    print "the total error rate is %f" % (errorCount / float(numTestVecs))

#
#  kNN on handwritten digits
#

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def hwTrainingMat(dataDir):
    trainingFileList = listdir( '%s/trainingDigits' % dataDir )
    hwLabels = []
    m = len( trainingFileList )
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[ i ]
        fileStr = fileNameStr.split( '.' )[ 0 ]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('%s/trainingDigits/%s' % (dataDir, fileNameStr))
    return trainingMat, hwLabels

def handwritingClassTest(dataDir):
    trainingMat, hwLabels = hwTrainingMat(dataDir)
    testFileList = listdir('%s/testDigits' % dataDir)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('%s/testDigits/%s' % (dataDir, fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 2)
        print "the classifier came back with: %s, the real answer is: %s" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount

