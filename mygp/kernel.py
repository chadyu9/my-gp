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
    distXY = (
        np.sum(X**2, axis=1).reshape(-1, 1)
        + np.sum(Y**2, axis=1)
        - 2 * np.dot(X, Y.T)
    )
    return sigma_f**2 * np.exp(-0.5 / l**2 * distXY)


def linear(X, Y, sigma_b=0.0, sigma_v=1.0, c=0.0):
    """
    Returns matrix of linear kernel between every pair of m vectors and n vectors.

    Args:
        X: (m x d) matrix of m vectors in R^d
        Y: (n x d) matrix of n vectors in R^d
        sigma_b: constant variance for linear kernel
        sigma_v: vertical variance parameter for linear kernel
        c: constant offset parameter for linear kernel
    """
    return sigma_b**2 + sigma_v**2 * (X - c) @ (Y.T - c)


def rq(X, Y, l=1.0, sigma=1.0, alpha=1.0):
    """
    Returns matrix of rational quadratic kernel between every pair of m vectors and n vectors.

    Args:
        X: (m x d) matrix of m vectors in R^d
        Y: (n x d) matrix of n vectors in R^d
        l: length scale parameter for rational quadratic kernel
        sigma: signal variance parameter for rational quadratic kernel
        alpha: alpha parameter for rational quadratic kernel
    """
    distXY = (
        np.sum(X**2, axis=1).reshape(-1, 1)
        + np.sum(Y**2, axis=1)
        - 2 * np.multiply(X, Y.T)
    )
    return sigma**2 * (1 + distXY / (2 * alpha * l**2)) ** (-alpha)
