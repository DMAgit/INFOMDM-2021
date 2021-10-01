from node import Node
import itertools
import numpy as np

def gini_index(labels):
        totalTrue = np.sum(labels) #Get the amount of true labels
        probTrue = totalTrue / labels.shape[0] #Calculate the proportion of true lables

        return probTrue * (1 - probTrue) #Calculate the gini index as shown in the lecture slides

class DescisionTree:

    def __init__(self) -> None:
        self.rootNode = None

        self.nmin = 0
        self.minleaf = 0
    
    def predict(self, dataRows):

        resultLabels = [self.rootNode.predict(row) for row in dataRows]   
        return resultLabels

    def construct(self, x, y, nmin: int, minleaf: int):

        #Store the parameters
        self.nmin = nmin
        self.minleaf = minleaf

        #Check whether there is any training data, and whether each data point has a label
        assert x.shape > (0,0)
        assert x.shape[0] == y.shape[0]

        self.rootNode = self.grow_tree(x, y)
        
    #Returns all possible splits
    def getPossibleSplits(self, x, y): #Get all the possinle splits, for all the features
        featureIndices = range(x.shape[1])

        possibleSplits = [self.getSplitsPerFeature(x, index) for index in featureIndices]
        
        allCombinations = [(index,splitValue) for index in featureIndices for splitValue in possibleSplits[index]]
        return allCombinations

    #Returns the possible splits per feature
    def getSplitsPerFeature(self, x, featureIndex): #Get all possible splits in a single feature (halfway between consecutive values)
        featureValuesSorted = np.sort(np.unique(x[:,featureIndex]))
        featureValuesAveraged = (featureValuesSorted[0:-1]+featureValuesSorted[1:])/2

        return featureValuesAveraged

    #Returns the best splitCombination
    def getBestSplit(self, x, y, allCombinations):
        allScores = [self.GetCurrentScore(x, y, combination) for combination in allCombinations]        
        bestCombination = allCombinations[np.argmin(allScores)]
        return bestCombination

    #Returns the delta i, without calculating the impurity of current node, so you want the smallest one
    def GetCurrentScore(self, x, y, combination):
        xLeft, xRight, yLeft, yRight = self.getCurrentSplit(x, y, combination)
        return gini_index(yLeft) * len(yLeft) / len(y) + gini_index(yRight) * len(yRight) / len(y)

    #Returns the split dataset and labels according to combination
    def getCurrentSplit(self, x, y, combination):
        featureIndex, splitValue = combination

        #Get the masks
        leftMask = x[:,featureIndex] < splitValue
        rightMask = ~leftMask

        return x[leftMask], x[rightMask], y[leftMask], y[rightMask]
    
    #Grow the tree
    def grow_tree(self, x, y):
        allCombinations = self.getPossibleSplits(x,y)

        currentNode = Node()
        currentNode.setFinalClassLabel(y) #Get the majority vote for each Node

        if(len(allCombinations) == 0):
            print("No more possible combinations")
            return currentNode

        bestFeatureIndex, bestSplitValue = self.getBestSplit(x,y, allCombinations)
        xLeft, xRight, yLeft, yRight = self.getCurrentSplit(x, y, (bestFeatureIndex, bestSplitValue)) #Since it only happens once every iteration and is written in C no need to cache the results earlier. Redoing it is fine since it improves clean code
        
        if np.all(y == y[0]):
            print("All labels the same")
            return currentNode

        if len(yLeft) >= self.nmin and len(yRight) >= self.nmin:
            nodeLeft = self.grow_tree(xLeft, yLeft)
            nodeRight = self.grow_tree(xRight, yRight)

            currentNode.left = nodeLeft
            currentNode.right = nodeRight
            currentNode.setSplitFunction(bestFeatureIndex, bestSplitValue)

        return currentNode