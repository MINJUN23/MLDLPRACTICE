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
    val_x = None # X (data) of validation set.
    val_y = None # Y (label) of validation set.

    data_x = None
    data_y = None

    def __init__(self):
      
        dataset = load_breast_cancer()
        self.data_x = dataset['data']
        self.data_y = dataset['target']
        
    

    def getDataset_reg(self):
        tr_x = None ### TODO: YOUR CODE HERE
        tr_y = None  ### TODO: YOUR CODE HERE
        val_x = None ### TODO: YOUR CODE HERE
        val_y = None ### TODO: YOUR CODE HERE

        return [self.tr_x, self.tr_y, self.val_x, self.val_y]
    
    
    def getDataset_cls(self):    
        self.tr_x = self.data_x  ### TODO: YOUR CODE HERE
        self.tr_y = self.data_y   ### TODO: YOUR CODE HERE
        self.val_x = None  ### TODO: YOUR CODE HERE
        self.val_y = None ### TODO: YOUR CODE HERE
        return [self.tr_x, self.tr_y, self.val_x, self.val_y]


