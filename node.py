import numpy as np


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
    def setFinalClassLabel(self, labels):
        self.trainingLabels = labels

        positiveAmount = np.sum(labels)
        negativeAmount = len(labels) - positiveAmount

        # TODO figure out tie breaker
        if positiveAmount > negativeAmount:
            self.finalClassLabel = 1
        else:
            self.finalClassLabel = 0

    # Make the prediction with a datarow
    def predict(self, dataRow):
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
