
# class Tree:
#     def __init__(self):
#         pass


class Node:
    def __init__(self, featureIndex, featureSplitValue):
        self.left = None
        self.right = None

        self.featureIndex = featureIndex
        self.featureSplitValue = featureSplitValue

        self.finalClassLabel = None


    def predict(self, dataRow):

        if self.finalClassLabel is not None:
            return self.finalClassLabel

        if self.left is not None:
            if dataRow[self.featureIndex] < self.featureSplitValue:
                return self.left.predict(dataRow)
        
        if self.right is not None:
            return self.right.predict(dataRow)

        raise Exception("ERRORRR")