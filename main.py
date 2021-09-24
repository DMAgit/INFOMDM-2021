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



    return None

def get_best_split(data, labels):
    income_sorted = np.sort(np.unique(data[:,3]))




    # for featureIndex in range(len(data)):
    #     for row in data:
    #         leftDataSet, rightDataSet, leftLabelSet, rightLabelSet = get_current_split(featureIndex, row[featureIndex], data, labels)
    #         gineValue = gini_index

    return None

#TODO use the numpy indexing for this instead of the for loop
def get_current_split(featureIndex, featureSplitValue, data, labels):
    leftDataSet, rightDataSet = [], []
    leftLabelSet, rightLabelSet = [], []

    for index, dataRow in enumerate(data):
        if data[featureIndex] < featureSplitValue:
            leftDataSet.append(dataRow)
            leftLabelSet.append(labels[index])
        else:
            rightDataSet.append(dataRow)
            rightLabelSet.append(labels[index])

    return leftDataSet, rightDataSet, leftLabelSet, rightLabelSet


def tree_pred(x, tree):
    return None


def tree_grow_b():
    return None


def tree_pred_b():
    return None

# Calculates the gini_index using the formulate from slide ....  
def gini_index(labels):
    totalTrue = np.sum(labels)
    totalAmount = labels.shape[0]

    probTrue = totalTrue / totalAmount

    return probTrue * (1 - probTrue)


print(gini_index(np.array([1,0,1,1,1,0,0,1,1,0,1])))



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