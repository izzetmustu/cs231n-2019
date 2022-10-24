from builtins import range
from re import T
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]          # Number of training data

    for i in range(N):
      scores = X[i].dot(W)                                  # Scores of current training sample
      scores -= np.max(scores)                              # Numerical stabilitiy(Occurs while taking exps)
      softmax = np.exp(scores) /np.sum(np.exp(scores))
      softmax_logged = -np.log(softmax)                      # Softmax scores of current training sample
      loss += softmax_logged[y[i]]
      for j in range(W.shape[1]):               # dScores/dW = X[i]
        if j == y[i]:
          dW[:,j] += X[i]*softmax[j] - X[i]     # Gradient of -log(softmax[j]) = softmax[j] - 1 if j == y[i]
        else:                           
          dW[:,j] += X[i]*softmax[j]            # Gradient of -log(softmax[j]) = softmax[j] if j == y[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= N
    # Add regularization to the loss.
    loss += reg*np.sum(X*X)

    # Right now the gradient is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.    
    dW /= N
    # Add regularization to the gradient(derivative of regularization term).
    dW += reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]          # Number of training data

    scores = X.dot(W)                                       # Scores of current training sample
    scores -= np.max(scores, axis=1, keepdims=True)         # Numerical stabilitiy(Occurs while taking exps)
    softmax = np.exp(scores) /np.sum(np.exp(scores), axis=1, keepdims=True)
    softmax_logged = -np.log(softmax)                       # Softmax scores of current training sample
    loss = np.sum(softmax_logged[np.arange(N), y])          # Calculate loss          
    softmax[np.arange(N), y] -= 1                           # Gradient of -log(softmax[j]) = softmax[j] - 1 if j == y[i]
    dW = X.T.dot(softmax)                                   # Gradient of -log(softmax[j]) = softmax[j] if j == y[i]


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= N
    # Add regularization to the loss.
    loss += reg*np.sum(X*X)

    # Right now the gradient is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.    
    dW /= N
    # Add regularization to the gradient(derivative of regularization term).
    dW += reg*2*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
