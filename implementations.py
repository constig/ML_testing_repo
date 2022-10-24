# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from helpers import *
from costs import *

"""
For all functions in this file:
Args:
    y: output
    tx: data samples
    w:  initial weight vector
    max_iters: number of steps to run
    batch_size: number of data points in a mini-batch for the stochastic gradient
    gamma: stepsize
    lambda_: regularization parameter
    
Returns:
    w: last weight vector of the method
    loss: the corresponding loss value (cost function)
"""


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Computes the Gradient Descent (GD) algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma*grad
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """Computes the Stochastic Gradient Descent algorithm (SGD)."""
    w = initial_w
    for n_iter in range(max_iters):
        for mb_y, mb_tx in batch_iter(y, tx, batch_size):
            grad = compute_gradient(mb_y, mb_tx, w)
            loss = compute_mse(y, tx, w)
            w = w - gamma*grad
            #n_iter = n_iter + 1 ???? 
    return w, loss
    
    
def least_squares(y, tx):
    """Calculate the least squares solution."""
    N = y.shape[0]
    w_ls = np.linalg.inv(np.transpose(tx) @ tx) @ np.transpose(tx) @ y
    loss = compute_mse(y, tx, w_ls)
    return w_ls, loss
    
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = y.shape[0]
    D = tx.shape[1]
    lambda_prime = lambda_*2*N
    I = np.identity(D)
    w_ridge = np.linalg.inv((np.transpose(tx) @ tx) + lambda_prime*I) @ np.transpose(tx) @ y
    loss = compute_mse(y, tx, w_ridge) 
    return w_ridge, loss
      
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    "add description"
    w = initial_w
    threshold = 1e-8
    losses = []

    # start the logistic regression
    for iter in range(max_iter):
        w,loss = log_GD(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss
    

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    "add description"
    w = initial_w
    threshold = 1e-8
    losses = []

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        penalty = (lambda_/2)*np.linalg.norm(w)**2
        loss = log_reg_loss(y, tx, w)
        grad = log_reg_grad(y, tx, w) + 2*lambda_*w
        w = w - gamma*grad
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss