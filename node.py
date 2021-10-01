

class Node:
    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.splitFunction = None
        self.finalClassLabel = None

    def setChildNodes(self, left, right) -> None:
        self.left = left
        self.right = right
    
    def setSplitFunction(self, splitFunction) -> None:
        self.splitFunction = splitFunction