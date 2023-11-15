import jax.numpy as jnp
import matplotlib.pyplot as plt


class GP:
    def __init__(self, X_s, train_x, train_y, kernel, sigma_n=1e-8):
        """
        Initialize Gaussian Process with training data and kernel function, and set prior distribution on train data.

        Args:
            train_x: (m x d) matrix of m vectors in R^d
            train_y: (m x 1) vector of m scalars
            kernel: kernel function
            sigma_n: assumed noise level of data
        """
        self.X = train_x
        self.Y = train_y
        self.kernel = kernel

        self.X_s = X_s
        self.sigma_n = sigma_n

        # Prior mean and covariance on test points
        self.mu = jnp.zeros(len(X_s))
        self.cov = kernel(X_s, X_s) + sigma_n**2 * jnp.eye(len(X_s))

    def predict(self, X_s):
        """
        Computes posterior mean and covariance matrix for given test data, as well as the log marginal likelihood.

        Args:
            test_x: (n x d) matrix of n vectors in R^d

        Returns:
            (n x 1) vector of predicted means
            (n x n) matrix of predicted covariances
        """
        # Lower triangular solve to obtain alpha and thus posterior mean
        L_cov = jnp.linalg.cholesky(
            self.kernel(self.X, self.X) + self.sigma_n**2 * jnp.eye(len(self.X))
        )
        alpha = jnp.linalg.solve(L_cov.T, jnp.linalg.solve(L_cov, self.Y))
        K_s = self.kernel(self.X, X_s)
        mu_s = K_s.T @ alpha
        self.mu = mu_s.reshape(
            -1,
        )

        # Compute posterior covariance matrix
        K_ss = self.cov
        v = jnp.linalg.solve(L_cov, K_s)
        cov_s = K_ss - v.T @ v
        self.cov = cov_s

        # Compute log marginal likelihood
        log_likelihood = (
            -0.5 * self.Y.T @ alpha
            - jnp.sum(jnp.log(jnp.diag(L_cov)))
            - len(self.X) / 2 * jnp.log(2 * jnp.pi)
        )
        return mu_s, cov_s, log_likelihood

    def plot(self, train_x=None, train_y=None, samples=[]):
        confidence_interval = 1.96 * jnp.sqrt(jnp.diag(self.cov))

        plt.fill_between(
            self.X_s.flatten(),
            self.mu + confidence_interval,
            self.mu - confidence_interval,
            alpha=0.1,
            color="b",
        )

        plt.plot(self.X_s, self.mu, "r", lw=1, label="Mean")
        for i in range(len(samples)):
            plt.plot(self.X_s, samples[i], lw=1, label=f"Sample {i+1}")
        if train_x is not None and train_y is not None:
            plt.plot(train_x, train_y, "ro", label="Training points")
        plt.legend()
        plt.show()
