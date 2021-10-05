from node import Node
import itertools
import numpy as np
import sys


# Calculate the gini index as shown in the lecture slides
def gini_index(labels):
    totalTrue = np.sum(labels)  # Get the amount of true labels
    probTrue = totalTrue / labels.shape[0]  # Calculate the proportion of true labels

    return probTrue * (1 - probTrue)


class DescisionTree:
    def __init__(self) -> None:
        self.rootNode = None
        self.nmin = None
        self.minleaf = None
        self.nfeat = None

    # Make the predictions
    def predict(self, dataRows):
        resultLabels = [self.rootNode.predict(row) for row in dataRows]
        return resultLabels

    # Construct the trees
    def construct(self, x, y, nmin: int, minleaf: int, nfeat: int):

        # Store the parameters
        self.nmin = nmin
        self.minleaf = minleaf
        self.nfeat = nfeat

        # Check whether there is any training data, and whether each data point has a label
        assert x.shape > (0, 0)
        assert x.shape[0] == y.shape[0]
        # Check that nfeat is more than 0 and isn't a number greater than the number of columns in the data
        assert 0 < nfeat <= x.shape[1]

        self.rootNode = self.grow_tree(x, y)

    # Get all the possible splits, for all the features
    def getPossibleSplits(self, x):
        featureIndices = np.random.choice(range(x.shape[1]), size=self.nfeat, replace=False)
        featureIndices.sort()
        # Choose random indices (columns of x), of size nfeat, without replacement

        possibleSplits = []
        index = 0
        for i in range(x.shape[1]):
            if index < len(featureIndices):
                if i == featureIndices[index]:
                    possibleSplits.append(self.getSplitsPerFeature(x, featureIndices[0]))
                    index += 1
                else:
                    possibleSplits.append(np.empty(0))
            else:
                possibleSplits.append(np.empty(0))

        # Store it in a tuple for easy access
        allCombinations = [(index, splitValue) for index in featureIndices for splitValue in possibleSplits[index]]
        return allCombinations

    # Get all possible splits in a single feature (halfway between consecutive values because the data is all numerical)
    def getSplitsPerFeature(self, x, featureIndex):
        featureValuesSorted = np.sort(np.unique(x[:, featureIndex]))
        featureValuesAveraged = (featureValuesSorted[0:-1] + featureValuesSorted[1:]) / 2

        return featureValuesAveraged

    # Returns the best splitCombination according to the impurity function
    def getBestSplit(self, x, y, allCombinations):
        allScores = [self.GetCurrentScore(x, y, combination) for combination in allCombinations]
        bestCombination = allCombinations[np.argmin(allScores)]

        if np.min(allScores) == 1000:  # Since the reduction can't be more than 1 anyway,
            # this is a sufficient check on whether the split is allowed
            return None

        return bestCombination

    # Returns the delta i, without calculating the impurity of current node (is constant), so you want the smallest one
    def GetCurrentScore(self, x, y, combination):
        # TODO probably more efficient to not make the split itself, but use the method as described in the assignment get started
        xLeft, xRight, yLeft, yRight = self.getCurrentSplit(x, y, combination)

        # Check the minleaf restriction
        if len(yLeft) < self.minleaf or len(yRight) < self.minleaf:
            return 1000  # Since the reduction can't be more than 1 anyway, 1000 is more than sufficient.

        return gini_index(yLeft) * len(yLeft) / len(y) + gini_index(yRight) * len(yRight) / len(y)

    # Returns the split dataset and labels according to combination
    def getCurrentSplit(self, x, y, combination):
        featureIndex, splitValue = combination

        # Get the masks for easy access
        leftMask = x[:, featureIndex] < splitValue
        rightMask = ~leftMask

        return x[leftMask], x[rightMask], y[leftMask], y[rightMask]

    # Grow the tree (recursive function)
    def grow_tree(self, x, y):
        currentNode = Node()
        currentNode.setFinalClassLabel(y)  # Get the majority vote for each Node

        currentNode.trainingValuesIndices = x[:, 0]  # Debugging values

        if gini_index(y) == 0:  # All labels are the same
            return currentNode

        if len(y) < self.nmin:  # The restriction according to the assignment
            return currentNode

        allCombinations = self.getPossibleSplits(x)
        if len(allCombinations) == 0:  # No more possible combinations (needed if nmin = 0)
            return currentNode

        # Get the best split and split the data accordingly
        bestSplit = self.getBestSplit(x, y, allCombinations)

        # Check if there is an allowed split
        if bestSplit is None:
            return currentNode

        bestFeatureIndex, bestSplitValue = bestSplit
        xLeft, xRight, yLeft, yRight = self.getCurrentSplit(x, y, (bestFeatureIndex, bestSplitValue))
        # Since it only happens once every iteration and is written in C no need to cache the results earlier.
        # Redoing it is fine since it improves clean code

        # recursively generate the child nodes as well
        currentNode.left = self.grow_tree(xLeft, yLeft)
        currentNode.right = self.grow_tree(xRight, yRight)

        currentNode.setSplitValues(bestFeatureIndex, bestSplitValue)

        return currentNode
