import numpy as np


def rbf(X, Y, l=1.0, sigma_f=1.0):
    """
    Returns matrix of RBF kernel between every pair of m vectors and n vectors.

    Args:
        X: (m x d) matrix of m vectors in R^d
        Y: (n x d) matrix of n vectors in R^d
        l: length scale parameter for RBF kernel
        sigma_f: signal variance parameter for RBF kernel

    Returns:
        (m x n) matrix of RBF kernel values
    """
    distXY = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.multiply(X, Y.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * distXY)
