import numpy as np
import pandas as pd
from decision_tree import DescisionTree
from visualiser import Visualiser

DATAPATH = 'data/credit.txt'
DELIMETER = ','
SKIP_HEADER = True

def main():
    #Get the data, and split the values from the labels
    credit_data = np.genfromtxt(DATAPATH, delimiter=DELIMETER, skip_header=SKIP_HEADER)
    x = credit_data[:,:-1]
    y = credit_data[:,-1]

    tree = tree_grow(x, y, 3, 1)

    #Visualise the tree in the console
    visualiser = Visualiser()
    visualiser.visualiseTree(tree)

#Construct and return the tree
def tree_grow(x, y, nmin: int, minleaf: int): #, nfeat):
    tree = DescisionTree()
    tree.construct(x, y, nmin, minleaf)

    return tree

if __name__ == "__main__":
    main()