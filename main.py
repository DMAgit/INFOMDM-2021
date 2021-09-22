import numpy as np
import pandas as pd

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)

print(credit_data)


def tree_grow(x, y, nmin: int, minleaf: int, nfeat):
    return None


def tree_pred(x, tree):
    return None


def tree_grow_b():
    return None


def tree_pred_b():
    return None


# --------------------------------------------
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini
