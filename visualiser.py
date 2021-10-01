class Visualiser:

    #Visualise the tree in ascii form
    def visualiseTree(self, tree):
        depth = 0
        stack = [(tree.rootNode, 0)]

        while(len(stack) > 0):
            currentNode, currentDepth = stack.pop()

            if currentNode.index is None: #is leaf node
                print(currentDepth * "\t", "leaf node with class label", currentNode.finalClassLabel,"trainingValues",currentNode.trainingValuesIndices,"remaining labels",currentNode.trainingLabels)
            else:
                print(currentDepth * "\t", "node with splitIndex", currentNode.index, "and splitValue", currentNode.splitValue,"trainingValues",currentNode.trainingValuesIndices,"remaining labels",currentNode.trainingLabels)
                stack.append((currentNode.left, currentDepth + 1))
                stack.append((currentNode.right, currentDepth + 1))