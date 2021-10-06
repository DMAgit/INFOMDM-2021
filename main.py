import numpy as np
from typing import List, Any

from decision_tree import DescisionTree
from visualiser import Visualiser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from functools import wraps
from time import time

DATAPATH = 'data/eclipse-metrics-packages-2.0.csv'
DELIMETER = ';'
SKIP_HEADER = True


def main() -> None:
    # Get the training data, and split the values from the labels
    eclipse_data_train = np.genfromtxt(DATAPATH, delimiter=DELIMETER, skip_header=SKIP_HEADER)
    x_train = eclipse_data_train[:, 2:44]  # we want all of this slice other than 3
    x_train = np.delete(x_train, 1, 1)  # can this and the above be done in 1 line?

    y_train = eclipse_data_train[:, 3]
    # we want y to be a boolean (we try to predict whether there is or is not a bug, not how many there are)
    y_train[y_train > 0] = 1  # so we change all non-0 values to 1 https://stackoverflow.com/a/19666680/14598178

    # Get the test data
    eclipse_data_test = np.genfromtxt('data/eclipse-metrics-packages-3.0.csv', delimiter=DELIMETER, skip_header=SKIP_HEADER)
    x_test = eclipse_data_test[:, 2:44]  # we want all of this slice other than 3
    x_test = np.delete(x_test, 1, 1)  # can this and the above be done in 1 line?

    y_test = eclipse_data_test[:, 3]
    # we want y to be a boolean (we try to predict whether there is or is not a bug, not how many there are)
    y_test[y_test > 0] = 1  # so we change all non-0 values to 1

    test_pred(x_train, y_train, x_test, y_test, 15, 5, 41)  # for Analysis 1
    test_pred_b(x_train, y_train, x_test, y_test, 15, 5, 41, 100)  # for Analysis 2
    test_pred_b(x_train, y_train, x_test, y_test, 20, 5, 6, 100)  # for Analysis 3

    # Visualise the tree in the console
    # visualiser = Visualiser()
    # visualiser.visualiseTree(tree)


# Construct and return the tree
def tree_grow(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int) -> DescisionTree:
    tree = DescisionTree()
    tree.construct(x, y, nmin, minleaf, nfeat)
    return tree


def tree_pred(x: np.ndarray, tr: DescisionTree) -> Any:
    predictedLabels = tr.predict(x)
    return predictedLabels


# Random forests
def tree_grow_b(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int, m: int) -> List[DescisionTree]:
    tree_list = []
    for i in range(m):
        index = np.random.choice(x.shape[0], x.shape[0], replace=True)
        # Get an array of indexes of size x (with replacement)
        tree_list.append(tree_grow(x[index], y[index], nmin, minleaf, nfeat))
        # Use the index to subset x and y
    return tree_list  # list of tree objects of len m


def tree_pred_b(tree_list: List, x: np.ndarray) -> bool:
    allResults = [tree_pred(x, tree) for tree in tree_list]
    allResultsArray = np.array(allResults)

    # Get the total votes for classifying 1
    summedResults = allResultsArray.sum(axis=0)  # get a sum of the results from each tree for each row

    # If the sum is bigger than half the dataset (rounded down for tiebreakers) then it is the majority
    finalResult = summedResults > len(allResults) // 2

    return finalResult


# For debugging
def timing(f: Any) -> Any:
    @wraps(f)
    def wrap(*args, **kw) -> Any:
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
              (f.__name__, te - ts))
        return result

    return wrap


@timing
def test_pred(x_train, y_train, x_test, y_test, nmin: int, minleaf: int, nfeat: int):
    tree = tree_grow(x_train, y_train, nmin, minleaf, nfeat)
    treePrediction = tree_pred(x_test, tree)
    print(accuracy_score(y_test, treePrediction))
    print(confusion_matrix(y_test, treePrediction))


@timing
def test_pred_b(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, nmin: int, minleaf: int, nfeat: int, m: int) -> None:
    treeList = tree_grow_b(x_train, y_train, nmin, minleaf, nfeat, m)
    baggingPredictions = tree_pred_b(treeList, x_test)
    print("m: ", m)
    print(accuracy_score(y_test, baggingPredictions))
    print(confusion_matrix(y_test, baggingPredictions))


if __name__ == "__main__":
    main()
