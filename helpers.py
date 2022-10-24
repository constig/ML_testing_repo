# -*- coding: utf-8 -*-
"""some helper functions."""
import csv
import numpy as np

"""
def load_data(sub_sample=True, add_outlier=False):
    path_dataset = "train.csv"
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1)
    ids = data[:, 0]
    return ids
"""


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            
def sigmoid(t):
    sig = 1/(1+np.exp(-t))
    return sig
def log_reg_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a non-negative loss
    """
    ln = np.log(1+np.exp(tx@w))
    mul = y*tx@w
    L = (1/tx.shape[0])*(ln-mul)
    loss = np.sum(L)
    return loss
def log_reg_grad(y, tx, w):
    """compute the gradient of loss.
    
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a vector of shape (D, 1)
    """
    sig = sigmoid(tx@w)
    grad = (1/tx.shape[0])*np.transpose(tx)@(sig-y)
    return grad


def log_GD(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1
    """
    loss = log_reg_loss(y, tx, w)
    grad = log_reg_grad(y, tx, w)
    wp = w - gamma*grad
    return wp,loss

def reg_log_GD(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1
    """
    penalty = (lambda_/2)*np.linalg.norm(w)**2
    loss = log_reg_loss(y, tx, w)
    grad = log_reg_grad(y, tx, w) + 2*lambda_*w
    w = w - gamma*grad
    return wp,loss

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a hessian matrix of shape=(D, D) 
    """
    S = np.eye(tx.shape[0])
    for i in range(tx.shape[0]):
        sn = sigmoid(tx[i,:]@w)*(1-sigmoid(tx[i,:]@w))
        S[i,i] *= sn
    hess = (1/tx.shape[0])*(np.transpose(tx)@S@tx)
    return hess