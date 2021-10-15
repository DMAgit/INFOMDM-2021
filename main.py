import numpy as np
from typing import List, Tuple, Union

DATAPATH = 'data/eclipse-metrics-packages-2.0.csv'
DELIMETER = ';'
SKIP_HEADER = True


class Node:
    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.splitFunction = None
        self.finalClassLabel = None

        self.index = None
        self.splitValue = None

        # Debugging purposes
        self.trainingValuesIndices = None
        self.trainingLabels = None

    # Set the childNodes
    def setChildNodes(self, left, right) -> None:
        self.left = left
        self.right = right

    # Set the splitValues
    def setSplitValues(self, index, splitValue) -> None:
        self.index = index
        self.splitValue = splitValue

    # Get the majority class label for this node
    def setFinalClassLabel(self, labels) -> None:
        self.trainingLabels = labels

        positiveAmount = np.sum(labels)
        negativeAmount = len(labels) - positiveAmount

        # TODO figure out tie breaker
        if positiveAmount > negativeAmount:
            self.finalClassLabel = 1
        else:
            self.finalClassLabel = 0

    # Make the prediction with a datarow
    def predict(self, dataRow) -> None:
        if self.index is None:  # Leaf node
            return self.finalClassLabel

        # Debugging purposes
        assert self.left is not None
        assert self.right is not None

        # Iterate to the next node according to the split value
        if dataRow[self.index] <= self.splitValue:
            return self.left.predict(dataRow)
        else:
            return self.right.predict(dataRow)


def gini_index(labels: np.ndarray) -> float:
    """
    Calculate the gini index as shown in the lecture slides

    :param labels: np.ndarray
    :return: Gini index (float)
    """
    totalTrue = np.sum(labels)  # Get the amount of true labels
    probTrue = totalTrue / labels.shape[0]  # Calculate the proportion of true labels

    return probTrue * (1 - probTrue)


class DecisionTree:
    def __init__(self) -> None:
        self.rootNode = None
        self.nmin = None
        self.minleaf = None
        self.nfeat = None

    def predict(self, dataRows: np.ndarray) -> list:
        """
        Make the predictions

        :param dataRows: rows of the data to make predictions on
        :return: list of predictions
        """
        resultLabels = [self.rootNode.predict(row) for row in dataRows]
        return resultLabels

    def construct(self, x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int) -> None:
        """
        Construct the trees

        :param x: array of features and their values
        :param y: labels
        :param nmin: minimum number of observations that a node must contain, for it to be allowed to be split
        :param minleaf: minimum number of observations required for a leaf node
        :param nfeat: the number of features that should be considered for each split
        :return: None
        """
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

    def getPossibleSplits(self, x) -> List[Tuple[int, float]]:
        """
        Get all the possible splits, for all the features

        :param x: array of features and their values
        :return: combinations of possible splits
        """
        # Choose random indices (columns of x), of size nfeat, without replacement
        # We use this to choose the nfeat features we are interested in
        featureIndices = np.random.choice(range(x.shape[1]), size=self.nfeat, replace=False)
        featureIndices.sort()

        # We want a list of possible splits which contains the values of the splits for features which we are interested
        # in and None values for the features we are not interested in
        possibleSplits = np.full(x.shape[1], None, dtype=object)
        possibleSplits[featureIndices] = [self.getSplitsPerFeature(x, index) for index in featureIndices]

        # Store it in a tuple for easy access TODO add explanation
        allCombinations = [(index, splitValue) for index in featureIndices for splitValue in possibleSplits[index]]
        return allCombinations

    def getSplitsPerFeature(self, x: np.ndarray, featureIndex: np.int32) -> float:
        """
        Get all possible splits in a single feature
        (halfway between consecutive values because the data is all numerical)

        :param x: array of features and their values
        :param featureIndex: array of indices of features to consider
        :return: featureValuesAveraged
        """
        featureValuesSorted = np.sort(np.unique(x[:, featureIndex]))
        featureValuesAveraged = (featureValuesSorted[0:-1] + featureValuesSorted[1:]) / 2

        return featureValuesAveraged

    def getBestSplit(self, x: np.ndarray, y: np.ndarray, allCombinations: list) -> Union[Tuple[int, float], None]:
        """
        :param x: array of features and their values
        :param y: labels
        :param allCombinations: list of combinations of possible splits
        :return: the best splitCombination according to the impurity function
        """
        allScores = [self.GetCurrentScore(x, y, combination) for combination in allCombinations]
        bestCombination = allCombinations[np.argmin(allScores)]

        if np.min(allScores) == 1000:  # Since the reduction can't be more than 1 anyway,
            # this is a sufficient check on whether the split is allowed
            return None
        return bestCombination

    def GetCurrentScore(self, x, y, combination) -> Union[float, int]:
        """
        Returns the delta i, without calculating the impurity of current node (is constant),
        so you want the smallest one

        :param x: array of features and their values
        :param y: labels
        :param combination: split combination
        :return: delta i
        """
        # TODO probably more efficient to not make the split itself, but use the method as described
        #  in the assignment get started
        xLeft, xRight, yLeft, yRight = self.getCurrentSplit(x, y, combination)

        # Check the minleaf restriction
        if len(yLeft) < self.minleaf or len(yRight) < self.minleaf:
            return 1000  # Since the reduction can't be more than 1 anyway, 1000 is more than sufficient.

        return gini_index(yLeft) * len(yLeft) / len(y) + gini_index(yRight) * len(yRight) / len(y)

    def getCurrentSplit(self, x: np.ndarray, y: np.ndarray, combination: Tuple[int, float]) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :param x: array of features and their values
        :param y: labels
        :param combination: split combination
        :return: split dataset and labels according to combination
        """
        featureIndex, splitValue = combination
        # Get the masks for easy access
        leftMask = x[:, featureIndex] < splitValue
        rightMask = ~leftMask

        return x[leftMask], x[rightMask], y[leftMask], y[rightMask]

    def grow_tree(self, x: np.ndarray, y: np.ndarray) -> Node:
        """
        Grow the tree (recursive function)

        :param x: array of features and their values
        :param y: labels
        :return: currentNode
        """
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


def main():
    # Get the training data, and split the values from the labels
    eclipse_data_train = np.genfromtxt(DATAPATH, delimiter=DELIMETER, skip_header=SKIP_HEADER)
    x_train = eclipse_data_train[:, 2:44]  # we want all of this slice other than 3

    x_train = np.delete(x_train, 1, 1)  # can this and the above be done in 1 line?

    y_train = eclipse_data_train[:, 3]
    # we want y to be a boolean (we try to predict whether there is or is not a bug, not how many there are)
    y_train[y_train > 0] = 1  # so we change all non-0 values to 1 https://stackoverflow.com/a/19666680/14598178
    # Get the test data
    eclipse_data_test = np.genfromtxt('data/eclipse-metrics-packages-3.0.csv', delimiter=DELIMETER,
                                      skip_header=SKIP_HEADER)
    x_test = eclipse_data_test[:, 2:44]  # we want all of this slice other than 3
    x_test = np.delete(x_test, 1, 1)  # can this and the above be done in 1 line?

    y_test = eclipse_data_test[:, 3]
    # we want y to be a boolean (we try to predict whether there is or is not a bug, not how many there are)
    y_test[y_test > 0] = 1  # so we change all non-0 values to 1

    # Tree_1 = tree_grow(x_train, y_train, 15, 5, 41)  # analysis 1
    # Tree_2 = tree_grow_b(x_train, y_train, 15, 5, 41, 100)  # analysis 2
    # Tree_3 = tree_grow_b(x_train, y_train, 15, 5, 6, 100)  # analysis 3


# Construct and return the tree
def tree_grow(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int) -> DecisionTree:
    """
    Construct the tree

    :param x: array of features and their values
    :param y: labels
    :param nmin: minimum number of observations that a node must contain, for it to be allowed to be split
    :param minleaf: minimum number of observations required for a leaf node
    :param nfeat: the number of features that should be considered for each split
    :return: Trained decision tree
    """
    tree = DecisionTree()
    tree.construct(x, y, nmin, minleaf, nfeat)
    return tree


def tree_pred(x: np.ndarray, tr: DecisionTree) -> list:
    """
    Make predictions with a tree

    :param x: array of features and their values to be predicted on
    :param tr: Decision tree to make the prediction with
    :return: Predictions
    """
    predictedLabels = tr.predict(x)
    return predictedLabels


# Random forests
def tree_grow_b(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int, m: int) -> List[DecisionTree]:
    """
    Construct m number of trees (for bagging/rf)

    :param x: array of features and their values
    :param y: labels
    :param nmin: minimum number of observations that a node must contain, for it to be allowed to be split
    :param minleaf: minimum number of observations required for a leaf node
    :param nfeat: the number of features that should be considered for each split
    :param m: number of trees to use for prediction
    :return: List of trained decision trees
    """
    tree_list = []
    for i in range(m):
        index = np.random.choice(x.shape[0], x.shape[0], replace=True)
        # Get an array of indexes of size x (with replacement)
        tree_list.append(tree_grow(x[index], y[index], nmin, minleaf, nfeat))
        # Use the index to subset x and y
    return tree_list  # list of tree objects of len m


def tree_pred_b(tree_list: List, x: np.ndarray) -> bool:
    """
    Make predictions with a list of trees

    :param tree_list: list of decision trees to make predictions with
    :param x: array of features and their values (to be predicted on)
    :return: Predictions
    """
    allResults = [tree_pred(x, tree) for tree in tree_list]
    allResultsArray = np.array(allResults)

    # Get the total votes for classifying 1
    summedResults = allResultsArray.sum(axis=0)  # get a sum of the results from each tree for each row

    # If the sum is bigger than half the dataset (rounded down for tiebreakers) then it is the majority
    finalResult = summedResults > len(allResults) // 2

    return finalResult


if __name__ == "__main__":
    main()
