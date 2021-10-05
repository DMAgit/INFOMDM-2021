import numpy as np
import pandas as pd
from decision_tree import DescisionTree
from visualiser import Visualiser
from sklearn.metrics import confusion_matrix
from statistics import mode
from sklearn.metrics import accuracy_score

DATAPATH = 'data/pimaIndians.txt'
DELIMETER = ','
SKIP_HEADER = False


def main():
    # Get the data, and split the values from the labels
    credit_data = np.genfromtxt(DATAPATH, delimiter=DELIMETER, skip_header=SKIP_HEADER)
    x = credit_data[:, :-1]
    y = credit_data[:, -1]

    test_pred_b(x, y, 20, 5, 4, 1)
    test_pred_b(x, y, 20, 5, 4, 10)
    test_pred_b(x, y, 20, 5, 4, 100)

    
    # Visualise the tree in the console
    # visualiser = Visualiser()
    # visualiser.visualiseTree(tree)


    
    # print(predictions)


# Construct and return the tree
def tree_grow(x, y, nmin: int, minleaf: int, nfeat: int):
    tree = DescisionTree()
    tree.construct(x, y, nmin, minleaf, nfeat)
    return tree


def tree_pred(x, tr):
    predictedLabels = tr.predict(x)
    return predictedLabels


# Random forests
def tree_grow_b(x, y, nmin: int, minleaf: int, nfeat: int, m: int):
    tree_list = []
    for i in range(m):
        tree_list.append(tree_grow(x, y, nmin, minleaf, nfeat))
    return tree_list  # list of tree objects of len m


def tree_pred_b(tree_list, x):
    allResults = [tree_pred(x, tree) for tree in tree_list]
    allResultsArray = np.array(allResults)
    summedResult = sum(allResultsArray).tolist()  # get a sum of the results from each tree for each row
    finalResult = []

    # since the outputs are binary, we can use the sum of the output and compare it against the number of models used
    # if it is more than half of the models, most models voted 1, so we assign it as such
    # otherwise, most models voted 0, so assign it as a 0
    for i in summedResult:
        # TODO: Tiebreaker
        if i > int(len(tree_list)/2):
            finalResult.append(1)
        else:
            finalResult.append(0)

    return finalResult

def test_pred_b(x, y, nmin: int, minleaf: int, nfeat: int, m: int):
    treeList = tree_grow_b(x, y, nmin, minleaf, nfeat, m)
    baggingPredictions = tree_pred_b(treeList, x)
    print("m: ",m)
    print(accuracy_score(y,baggingPredictions))
    print(confusion_matrix(y, baggingPredictions))

if __name__ == "__main__":
    main()
