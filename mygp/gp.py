import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from flax.training import train_state
from flax.core import FrozenDict


class GP:
    def __init__(
        self,
        train_x,
        train_y,
        kernel,
        hyperparams,
        sigma_n=1e-8,
        optimizer=optax.adam(1e-3),
    ):
        """
        Initialize Gaussian Process with training data and kernel function, and set prior distribution on train data.

        Args:
            train_x: (m x d) matrix of m vectors in R^d for training data
            train_y: (m x 1) vector of m scalars representing target values
            kernel: kernel function
            sigma_n: assumed noise level of data
        """
        self.X = train_x
        self.Y = train_y
        self.kernel = kernel

        # Create the train state
        self.init_train_state(
            params={**hyperparams, "sigma_n": sigma_n},
            optimizer=optimizer,
        )

    def init_train_state(self, params, optimizer):
        """
        Initialize FLAX training state.

        Args:
            params: model parameters
            optimizer: optimizer

        Returns:
            training state
        """
        self.train_state = train_state.TrainState.create(
            apply_fn=self.compute_post_dist, params=FrozenDict(params), tx=optimizer
        )

    def rff_basis(self, x, num_basis, key):
        """
        Generate a set of Random Fourier feature (RFF) basis functions assuming the kernel is SE.

        Args:
            x: (d,) vector in R^d
            num_basis: number of basis functions
            key: JAX random PRNG key

        Returns:
            (num_basis,) vector of RFF basis functions
        """

        # There must be an even number of basis functions for the RFF
        assert num_basis % 2 == 0
        (d,) = x.shape

        # Generate random weights
        weights = jr.multivariate_normal(
            key, jnp.zeros(d), jnp.eye(d), (num_basis // 2,)
        )

        # Return RFF basis functions
        return jnp.sqrt(2 / num_basis) * jnp.concatenate(
            (
                jnp.sin(jnp.dot(weights, x)),
                jnp.cos(jnp.dot(weights, x)),
            )
        )

    def prior_f(self, x, num_basis, key):
        # Split key for computing rff_basis
        key, subkey = jr.split(key)

        # Compute RFF basis and random weights from standard normal
        basis = self.rff_basis(x, num_basis, subkey)
        w = jr.normal(key, (num_basis,))
        return jnp.dot(w, basis)

    def compute_prior_dist(self, X_s):
        """
        Computes prior mean and covariance matrix for given test data.

        Args:
            X_s: (n x d) matrix of n vectors in R^d for test data

        Returns:
            (n x 1) vector of prior means
            (n x n) matrix of prior covariances
        """
        # Get hyperparameters and noise level from training state
        sigma_n = self.train_state.params["sigma_n"]
        hyperparams = {
            k: v for k, v in self.train_state.params.items() if k != "sigma_n"
        }

        # Compute prior mean
        mu_s = jnp.zeros(X_s.shape)

        # Compute prior covariance matrix
        cov_s = self.kernel(X_s, X_s, **hyperparams) + sigma_n**2 * jnp.eye(len(X_s))

        return mu_s, cov_s

    def update(self, X_s, y, f, key):
        """
        Computes posterior sample given prior sample via pathwise conditioning.

        Args:
            X_s: (n x d) matrix of n vectors in R^d for test data
            y: (n x 1) vector of observations with noise at training points
            f: Random Fourier feature function approximating prior
            key: JAX random PRNG key
        """
        # Get hyperparameters and noise level from training state
        sigma_n = self.train_state.params["sigma_n"]
        hyperparams = {
            k: v for k, v in self.train_state.params.items() if k != "sigma_n"
        }

        # Lower triangular solve of covariance matrix
        K = self.kernel(
            self.X,
            self.X,
            **hyperparams,
        ) + sigma_n**2 * jnp.eye(len(self.X))
        L_cov = jsp.linalg.cholesky(
            K,
            lower=True,
        )

        # Return prior sample + pathwise update
        K_s = self.kernel(self.X, X_s, **hyperparams)
        return jnp.apply_along_axis(
            f, axis=1, arr=X_s
        ) + K_s.T @ jsp.linalg.solve_triangular(
            L_cov.T,
            jsp.linalg.cho_solve(
                (L_cov, True),
                y.ravel()
                - jnp.apply_along_axis(f, axis=1, arr=self.X)
                - jr.multivariate_normal(
                    key, jnp.zeros(len(y)), sigma_n**2 * jnp.eye(len(y))
                ),
            ),
        )

    def compute_post_dist(self, X_s):
        """
        Computes posterior mean and covariance matrix for given test data.

        Args:
            X_s: (n x d) matrix of n vectors in R^d for test data

        Returns:
            (n x 1) vector of predicted means
            (n x n) matrix of predicted covariances
        """
        # Get hyperparameters and noise level from training state
        sigma_n = self.train_state.params["sigma_n"]
        hyperparams = {
            k: v for k, v in self.train_state.params.items() if k != "sigma_n"
        }

        # Lower triangular solve to obtain alpha and thus posterior mean
        K = self.kernel(
            self.X,
            self.X,
            **hyperparams,
        ) + sigma_n**2 * jnp.eye(len(self.X))
        L_cov = jsp.linalg.cholesky(
            K,
            lower=True,
        )

        alpha = jsp.linalg.solve_triangular(
            L_cov.T, jsp.linalg.cho_solve((L_cov, True), self.Y)
        )

        K_s = self.kernel(self.X, X_s, **hyperparams)
        mu_s = K_s.T @ alpha
        mu_s = mu_s.reshape(
            -1,
        )

        # Compute posterior covariance matrix
        K_ss = self.kernel(X_s, X_s, **hyperparams) + 1e-8 * jnp.eye(len(X_s))
        v = jsp.linalg.solve_triangular(L_cov, K_s, lower=True)
        cov_s = K_ss - v.T @ v

        return mu_s, cov_s

    def mll(self, params):
        """
        Computes negative log marginal likelihood of training data.

        Args:
            state: model state

        Returns:
            negative log marginal likelihood of training data
        """
        # Get hyperparameters and noise level from training state
        sigma_n = params["sigma_n"]
        hyperparams = {k: v for k, v in params.items() if k != "sigma_n"}

        # Lower triangular solve to obtain alpha
        L_cov = jsp.linalg.cholesky(
            self.kernel(
                self.X,
                self.X,
                **hyperparams,
            )
            + sigma_n * jnp.eye(len(self.X)),
            lower=True,
        )
        alpha = jsp.linalg.cho_solve(
            (L_cov.T, False), jsp.linalg.cho_solve((L_cov, True), self.Y)
        )

        # Compute negative log marginal likelihood
        log_likelihood = (
            -0.5 * self.Y.T @ alpha
            - jnp.sum(jnp.log(jnp.diag(L_cov)))
            - len(self.X) / 2 * jnp.log(2 * jnp.pi)
        ).squeeze()
        return -log_likelihood

    def step(self):
        """
        Performs one step of training.

        Returns:
            negative log marginal likelihood in current state
        """
        # Evaluate negative log marginal likelihood and its gradients
        loss, grads = jax.value_and_grad(self.mll, has_aux=False)(
            self.train_state.params
        )

        # Update training state
        self.train_state = self.train_state.apply_gradients(grads=grads)
        return loss

    def train(self, num_epochs=1000, print_every=100):
        """
        Trains the GP for given number of epochs.

        Args:
            num_epochs: number of epochs to train for
            print_every: number of epochs between printing loss
        """
        for epoch in range(num_epochs):
            loss = self.step()
            if epoch % print_every == 0:
                print(f"Epoch {epoch+1}: loss = {loss:.3f}")

    def plot(self, X_s, mu, cov, train_x=None, train_y=None, samples=[]):
        """
        Plots mean and error bars for the GP, along with samples.

        Args:
            X_s: (n x d) matrix of n vectors in R^d for test data
            mu: mean of the distribution to be plotted
            cov: covariance matrix for the distribution on the test points
            train_x: (m x d) matrix of m vectors in R^d for optional training data
            train_y: (m x 1) vector of m scalars representing target values
            samples: list of (n x 1) vectors of n samples from posterior distribution
        """

        # Distance for 95% confidence interval
        confidence_interval = 1.96 * jnp.sqrt(jnp.diag(cov))

        # Plot error barrs
        plt.fill_between(
            X_s.ravel(),
            mu.ravel() + confidence_interval,
            mu.ravel() - confidence_interval,
            alpha=0.2,
            color="b",
        )

        # Plot mean and samples
        plt.plot(X_s.ravel(), mu.ravel(), "r", lw=1, label="Mean")
        for i, sample in enumerate(samples):
            plt.plot(X_s.ravel(), sample, lw=1, label=f"Sample {i+1}")
        if train_x is not None:
            plt.plot(train_x.ravel(), train_y.ravel(), "rx", label="Training points")
        plt.legend()
        plt.show()
