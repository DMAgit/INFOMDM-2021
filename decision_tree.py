from node import Node
import itertools
import numpy as np

#Calculate the gini index as shown in the lecture slides
def gini_index(labels):
        totalTrue = np.sum(labels) #Get the amount of true labels
        probTrue = totalTrue / labels.shape[0] #Calculate the proportion of true lables

        return probTrue * (1 - probTrue) 

class DescisionTree:
    def __init__(self) -> None:
        self.rootNode = None
        self.nmin = None
        self.minleaf = None
    
    #Make the predictions
    def predict(self, dataRows):
        resultLabels = [self.rootNode.predict(row) for row in dataRows]   
        return resultLabels

    #Construct the trees
    def construct(self, x, y, nmin: int, minleaf: int):

        #Store the parameters
        self.nmin = nmin
        self.minleaf = minleaf

        #Check whether there is any training data, and whether each data point has a label
        assert x.shape > (0,0)
        assert x.shape[0] == y.shape[0]

        self.rootNode = self.grow_tree(x, y)
        
    #Get all the possible splits, for all the features
    def getPossibleSplits(self, x, y): 
        featureIndices = range(x.shape[1])

        possibleSplits = [self.getSplitsPerFeature(x, index) for index in featureIndices]
        
        #Store it in a tuple for easy access
        allCombinations = [(index,splitValue) for index in featureIndices for splitValue in possibleSplits[index]]
        return allCombinations

    #Get all possible splits in a single feature (halfway between consecutive values because the data is all numerical)
    def getSplitsPerFeature(self, x, featureIndex): 
        featureValuesSorted = np.sort(np.unique(x[:,featureIndex]))
        featureValuesAveraged = (featureValuesSorted[0:-1]+featureValuesSorted[1:])/2

        return featureValuesAveraged

    #Returns the best splitCombination according to the impurity function
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

        #Get the masks for easy access
        leftMask = x[:,featureIndex] < splitValue
        rightMask = ~leftMask

        return x[leftMask], x[rightMask], y[leftMask], y[rightMask]
    
    #Grow the tree (recursive function)
    def grow_tree(self, x, y):
        currentNode = Node()
        currentNode.setFinalClassLabel(y) #Get the majority vote for each Node
        
        currentNode.trainingValuesIndices = x[:,0] #Debugging values

        if np.all(y == y[0]): #All labels the same            
            return currentNode

        if len(y) < self.nmin: #The restriction according to the assignment
            return currentNode

        allCombinations = self.getPossibleSplits(x,y)
        if(len(allCombinations) == 0): #No more possible combinations
            return currentNode

        #Get the best split and split the data accordingly
        bestFeatureIndex, bestSplitValue = self.getBestSplit(x,y, allCombinations)
        xLeft, xRight, yLeft, yRight = self.getCurrentSplit(x, y, (bestFeatureIndex, bestSplitValue)) #Since it only happens once every iteration and is written in C no need to cache the results earlier. Redoing it is fine since it improves clean code

        #Check the minleaf restriction
        if len(yLeft) >= self.minleaf and len(yRight) >= self.minleaf:
            #recursively generate the child nodes aswell
            currentNode.left = self.grow_tree(xLeft, yLeft)
            currentNode.right = self.grow_tree(xRight, yRight)

            currentNode.setSplitValues(bestFeatureIndex, bestSplitValue)

        return currentNode