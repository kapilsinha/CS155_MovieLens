# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import copy

def grad_U(Ui, Yij, Vj, ai, bj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta * (reg * Ui - (Vj * (Yij - (np.dot(Ui, Vj) + ai + bj))))

def grad_V(Vj, Yij, Ui, ai, bj, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta * (reg * Vj - (Ui * (Yij - (np.dot(Ui, Vj) + ai + bj))))

def get_err(U, V, Y, a, b, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    err = 0
    reg_term = ((np.linalg.norm(U) ** 2) + (np.linalg.norm(V) ** 2)) * reg / 2
    for k in range(len(Y)):
        i = Y[k][0] - 1
        j = Y[k][1] - 1
        err += (Y[k][2] - (np.dot(U[i], V[j]) + a[i] + b[j])) ** 2

    err /= 2

    err = (err + reg_term)/(len(Y))

    return err

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """

    # Initialize U and V to be uniform random variables in the interval
    U = np.random.random((M, K)) - 0.5
    V = np.random.random((N, K)) - 0.5

    # Initialize bias terms
    a = np.random.random((M)) - 0.5
    b = np.random.random((N)) - 0.5

    # Error for epoch 0
    err = get_err(U, V, Y, a, b, reg)
    first_diff = 0
    epoch = 1
    prev = 0

    while(epoch <= max_epochs):
        # Shuffle the order in which we traverse the training data
        indices = np.random.permutation(range(len(Y)))
        for k in indices:
            i = Y[k][0] - 1
            j = Y[k][1] - 1

            U[i] -= grad_U(U[i], Y[k][2], V[j], a[i], b[j], reg, eta)
            V[j] -= grad_V(V[j], Y[k][2], U[i], a[i], b[j], reg, eta)

        prev = err
        err = get_err(U, V, Y, a, b, reg)

        # Get the first difference in errors between epoch 1 and epoch 0
        if(epoch == 1):
            first_diff = prev - err

        # Stopping condition
        if (epoch > 1 and prev - err <= (eps * first_diff)):
            break

        epoch += 1

    return (U, V, err)
