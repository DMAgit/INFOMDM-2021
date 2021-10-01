class Visualiser:

    def visualiseTree(self, tree):
        depth = 0
        stack = [(tree.rootNode, 0)]

        while(len(stack) > 0):
            currentNode, currentDepth = stack.pop()
            if currentNode.index is None:
                print(currentDepth * "\t", "leaf node with class label", currentNode.finalClassLabel)
            else:
                print(currentDepth * "\t", "node with splitIndex ", currentNode.index, " and splitValue ", currentNode.splitValue)
                stack.append((currentNode.left, currentDepth + 1))
                stack.append((currentNode.right, currentDepth + 1))
            


            