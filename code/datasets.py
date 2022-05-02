from numpy import *
import util

from sklearn.datasets import load_breast_cancer


class BreastCancerDataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = None  # X (data) of training set.
    tr_y = None  # Y (label) of training set.
    val_x = None  # X (data) of validation set.
    val_y = None  # Y (label) of validation set.

    data_x = None
    data_y = None

    def __init__(self):

        dataset = load_breast_cancer()
        BreastCancerDataset.data_x = dataset['data']
        print(BreastCancerDataset.data_x)
        BreastCancerDataset.data_y = dataset['target']
        print(BreastCancerDataset.data_y)

    def getDataset_reg(self):
        # TODO: YOUR CODE HERE
        tr_x = None  # TODO: YOUR CODE HERE
        tr_y = None  # TODO: YOUR CODE HERE
        val_x = None  # TODO: YOUR CODE HERE
        val_y = None  # TODO: YOUR CODE HERE

        return [self.tr_x, self.tr_y, self.val_x, self.val_y]

    def getDataset_cls(self):
        # TODO: YOUR CODE HERE
        tr_x = None  # TODO: YOUR CODE HERE
        tr_y = None  # TODO: YOUR CODE HERE
        val_x = None  # TODO: YOUR CODE HERE
        val_y = None  # TODO: YOUR CODE HERE

        return [self.tr_x, self.tr_y, self.val_x, self.val_y]
