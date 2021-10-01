import numpy as np
import pandas as pd
from decision_tree import DescisionTree

DATAPATH = 'data/credit.txt'
DELIMETER = ','
SKIP_HEADER = True

def main():
    credit_data = np.genfromtxt(DATAPATH, delimiter=DELIMETER, skip_header=SKIP_HEADER)
    x = credit_data[:,:-1]
    y = credit_data[:,-1]

    tree_grow(x, y, 0, 0)

def tree_grow(x, y, nmin: int, minleaf: int): #, nfeat):
    tree = DescisionTree()
    tree.construct(x, y, nmin, minleaf)

if __name__ == "__main__":
    main()