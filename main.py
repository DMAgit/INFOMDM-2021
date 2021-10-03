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

    tree = tree_grow(x, y, 20, 5)

    # Visualise the tree in the console
    # visualiser = Visualiser()
    # visualiser.visualiseTree(tree)

    predictions = tree_pred(x, tree)
    print(confusion_matrix(y, predictions))


# Construct and return the tree
def tree_grow(x, y, nmin: int, minleaf: int):  # , nfeat):
    tree = DescisionTree()
    tree.construct(x, y, nmin, minleaf)
    return tree


def tree_pred(x, tr):
    predictedLabels = tr.predict(x)
    return predictedLabels


if __name__ == "__main__":
    main()
