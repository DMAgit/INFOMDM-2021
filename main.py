from tree import Node
import numpy as np
import pandas as pd

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)

data = credit_data[:, :5] 
classLabels = credit_data[:, 5]

splitGroup1 = data[:5,:]
splitGroup2 = data[5:10,:]

splitLabels1 = classLabels[:5]
splitLabels2 = classLabels[5:10]




def tree_grow(x, y, nmin: int, minleaf: int, nfeat):
    
    bestleftDataSet,bestrightDataSet,bestleftLabelSet,bestrightLabelSet,bestFeatureIndex,bestFeatureSplitValue = get_best_split(x, y)

    currentNode = Node(bestFeatureIndex, bestFeatureSplitValue)

    currentNode.left = tree_grow(bestleftDataSet, bestleftLabelSet)
    currentNode.right = tree_grow(bestrightDataSet, bestrightLabelSet)

    return currentNode


#TODO use the numpy indexing for this instead of the for loop
def get_current_split(featureIndex, featureSplitValue, data, labels):
    leftDataSet, rightDataSet = [], []
    leftLabelSet, rightLabelSet = [], []

    for index, dataRow in enumerate(data):
        if dataRow[featureIndex] <= featureSplitValue:
            leftDataSet.append(dataRow)
            leftLabelSet.append(labels[index])
        else:
            rightDataSet.append(dataRow)
            rightLabelSet.append(labels[index])

    return leftDataSet, rightDataSet, leftLabelSet, rightLabelSet

def gini_index(labels):
    totalTrue = np.sum(labels)
    totalAmount = len(labels)

    probTrue = totalTrue / totalAmount

    return probTrue * (1 - probTrue)



#Use numpy indexing later
def get_best_split(data, labels):

    rowAmount, featureAmount = data.shape

    bestGiniValue = 1
    bestleftDataSet = None
    bestrightDataSet = None
    bestleftLabelSet = None
    bestrightLabelSet = None
    bestFeatureIndex = None
    bestFeatureSplitValue = None

    for featureIndex in range(featureAmount):
        featureValuesSorted = np.sort(np.unique(data[:,featureIndex]))
        featureValuesAveraged = (featureValuesSorted[0:-1]+featureValuesSorted[1:rowAmount])/2
        for splitValue in featureValuesAveraged:
            leftDataSet, rightDataSet, leftLabelSet, rightLabelSet = get_current_split(featureIndex, splitValue, data, labels)
            giniValue = len(leftDataSet) / rowAmount * gini_index(leftLabelSet) + len(rightDataSet)/ rowAmount *  gini_index(rightLabelSet)
            print("The gini value for featureIndex ", featureIndex, " and splitValue : ", splitValue, " is ",giniValue)

            if giniValue < bestGiniValue:
                bestGiniValue = giniValue
                bestleftDataSet = leftDataSet
                bestrightDataSet = rightDataSet
                bestleftLabelSet = leftLabelSet
                bestrightLabelSet = rightLabelSet
                bestFeatureIndex = featureIndex
                bestFeatureSplitValue = splitValue
    

    print("The best giniValue was ", bestGiniValue, " with splitValue ", bestFeatureSplitValue)

    return bestleftDataSet,bestrightDataSet,bestleftLabelSet,bestrightLabelSet,bestFeatureIndex,bestFeatureSplitValue

def tree_pred(x, tree):
    return None

def tree_grow_b():
    return None


def tree_pred_b():
    return None

# TODO
# Make a tree structure
# Draw Tree
# Get Possible splits
# Evaluate splits
# Make split
# Form tree
# Make predictions
# Finalise
# Bagging
# Random Forests