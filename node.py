

class Node:
    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.splitFunction = None
        self.finalClassLabel = None

        self.index = None
        self.splitValue = None
        self.trainingValues = None

    def setChildNodes(self, left, right) -> None:
        self.left = left
        self.right = right
    
    def setSplitFunction(self, index, splitValue) -> None:
        self.index = index
        self.splitValue = splitValue

    def setFinalClassLabel(self, labels):
        self.trainingValues = labels
        positiveAmount = sum(labels)
        negativeAmount = len(labels) - positiveAmount

        #TODO figure out tie breaker
        if positiveAmount > negativeAmount:
            self.finalClassLabel = 1
        else:
            self.finalClassLabel = 0

    def predict(self, dataRow):
        
        if self.index is None:
            return self.finalClassLabel

        #Debugging purposes
        assert self.left is not None 
        assert self.right is not None

        if dataRow[self.index] < self.splitValue:
            return self.left.predict(dataRow)
        else:
            return self.right.predict(dataRow)

        
