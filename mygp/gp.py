import numpy as np


class GP:
    def __init__(self, train_x, train_y, kernel, sigma_n=1e-8):
        """
        Initialize Gaussian Process with training data and kernel function, and set prior distribution.

        Args:
            train_x: (m x d) matrix of m vectors in R^d
            train_y: (m x 1) vector of m scalars
            kernel: kernel function
            sigma_n: assumed noise level of data
        """
        self.X = train_x
        self.Y = train_y
        self.kernel = kernel

        self.mu = np.zeros(len(train_x))
        self.cov = kernel(train_x, train_x) + sigma_n**2 * np.eye(len(train_x))

    def predict(self, test_x):
        """
        Computes posterior mean and covariance matrix for given test data, as well as the log marginal likelihood.

        Args:
            test_x: (n x d) matrix of n vectors in R^d

        Returns:
            (n x 1) vector of predicted means
            (n x n) matrix of predicted covariances
        """
        L_cov = np.linalg.cholesky(self.cov)
        alpha = np.linalg.solve(L_cov.T, np.linalg.solve(L_cov, self.Y))
        K_s = self.kernel(self.X, test_x)
        K_ss = self.kernel(test_x, test_x)
        mu_s = K_s.T @ alpha

        v = np.linalg.solve(L_cov, K_s)
        cov_s = K_ss - v.T @ v

        log_likelihood = -0.5 * self.Y.T @ alpha - np.sum(np.log(np.diag(L_cov))) - len(self.X) / 2 * np.log(2 * np.pi)
        return mu_s, cov_s, log_likelihood
