
# class Tree:
#     def __init__(self):
#         pass


class Node:
    def __init__(self, featureIndex, featureSplitValue):
        self.left = None
        self.right = None

        self.featureIndex = None
        self.featureSplitValue = None


    def predict(self, dataRow):
        if dataRow[self.featureIndex] < self.featureSplitValue:
            return self.left
        
        return self.right