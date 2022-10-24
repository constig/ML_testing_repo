# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from helpers import *

def compute_error(y, tx, w):
    """Computes the error."""
    error = y - tx @ w
    return error 


def compute_mse(y, tx, w):
    """Computes the loss using MSE."""
    N = y.shape[0]
    error = compute_error(y, tx, w)
    MSE = (1/2*N) * np.sum(error, axis=0)
    return MSE


def compute_gradient(y, tx, w):
    """Computes the gradient at w."""
    error = compute_error(y, tx, w)
    N = y.shape[0]
    grad = -1/N * (np.transpose(tx) @ error) 
    return grad