import sys
import inspect
import random

from numpy import *
from pylab import *
from matplotlib import pyplot

import util

def visRegressedLine(X, predicted_y, w):
    pass
    ### TODO: YOUR CODE HERE
    
def visClassifier(X, predicted_y, w):
    pass
    ### TODO: YOUR CODE HERE
    
def visLoss(loss):
    pass
    ### TODO: YOUR CODE HERE
    
def visLikelihood(likelihood):
    pass
    ### TODO: YOUR CODE HERE

def computeClassificationAcc(org_y, predicted_y):
    pass
    '''
        Compute classification accuracy by counting how many predicted_y
        is the same to the org_y
    '''
    ### TODO: YOUR CODE HERE

def computeAvgRegrMSError(org_y, predicted_y):
    pass
    '''
        Compute regression error by average error between predicted_y
        and org_y. Use L2 distance between two values (each eleement 
        in the vector).
    '''
    ### TODO: YOUR CODE HERE