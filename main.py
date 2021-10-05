import numpy as np
import pandas as pd
from decision_tree import DescisionTree
from visualiser import Visualiser
from sklearn.metrics import confusion_matrix
from statistics import mode
from sklearn.metrics import accuracy_score
from functools import wraps
from time import time

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

    #Get the total votes for classifing 1
    summedResults = allResultsArray.sum(axis=0)  # get a sum of the results from each tree for each row
    
    #If the sum is bigger than half the dataset (rounded down for tiebreakers) then it is the majority
    finalResult = summedResults > len(allResults) // 2
    
    return finalResult


#For debugging
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

@timing
def test_pred_b(x, y, nmin: int, minleaf: int, nfeat: int, m: int):
    treeList = tree_grow_b(x, y, nmin, minleaf, nfeat, m)
    baggingPredictions = tree_pred_b(treeList, x)
    print("m: ",m)
    print(accuracy_score(y,baggingPredictions))
    print(confusion_matrix(y, baggingPredictions))

if __name__ == "__main__":
    main()
