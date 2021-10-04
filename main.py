import numpy as np
import pandas as pd
from decision_tree import DescisionTree
from visualiser import Visualiser
from sklearn.metrics import confusion_matrix

DATAPATH = 'data/pimaIndians.txt'
DELIMETER = ','
SKIP_HEADER = False


def main():
    # Get the data, and split the values from the labels
    credit_data = np.genfromtxt(DATAPATH, delimiter=DELIMETER, skip_header=SKIP_HEADER)
    x = credit_data[:, :-1]
    y = credit_data[:, -1]

    tree = tree_grow(x, y, 20, 5, 8)

    # Visualise the tree in the console
    # visualiser = Visualiser()
    # visualiser.visualiseTree(tree)

    predictions = tree_pred(x, tree)
    print(confusion_matrix(y, predictions))


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
    # TODO: finish this off, currently it returns all of the predicted labels,
    #       but (I think) it should return them 1 by 1 (for each row of x).
    #       This would require to change the tree_pred function and I can't go into it rn :)
    allResults = []
    for tree in tree_list:
        allResults.append(x, tree)


if __name__ == "__main__":
    main()
