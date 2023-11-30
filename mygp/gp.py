import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import optax
from flax.training import train_state
from flax.core import FrozenDict


class GP:
    def __init__(
        self,
        X_s,
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
            X_s: (n x d) matrix of n vectors in R^d for test data
            train_x: (m x d) matrix of m vectors in R^d for training data
            train_y: (m x 1) vector of m scalars representing target values
            kernel: kernel function
            sigma_n: assumed noise level of data
        """
        self.X_s = X_s
        self.X = train_x
        self.Y = train_y
        self.kernel = kernel

        # Create the train state
        self.init_train_state(
            params={**hyperparams, "sigma_n": sigma_n},
            optimizer=optimizer,
        )

        # Prior mean and covariance on test points
        self.mu, self.cov = self.compute_prior(hyperparams, sigma_n)

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
            apply_fn=self.predict, params=FrozenDict(params), tx=optimizer
        )

    def compute_prior(self, hyperparams, sigma_n):
        """
        Computes prior mean and covariance matrix for given test data.

        Args:
            hyperparams: hyperparameters of kernel function
            sigma_n: assumed noise level of data

        Returns:
            (n x 1) vector of prior means
            (n x n) matrix of prior covariances
        """
        # Compute prior mean
        mu_s = jnp.zeros(self.X_s.shape)

        # Compute prior covariance matrix
        cov_s = self.kernel(self.X_s, self.X_s, **hyperparams) + sigma_n**2 * jnp.eye(
            len(self.X_s)
        )

        return mu_s, cov_s

    def predict(self):
        """
        Computes posterior mean and covariance matrix for given test data.

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

        K_s = self.kernel(self.X, self.X_s, **hyperparams)
        mu_s = K_s.T @ alpha
        self.mu = mu_s.reshape(
            -1,
        )

        # Compute posterior covariance matrix
        K_ss = self.kernel(self.X_s, self.X_s, **hyperparams) + 1e-8 * jnp.eye(
            len(self.X_s)
        )
        v = jsp.linalg.solve_triangular(L_cov, K_s, lower=True)
        cov_s = K_ss - v.T @ v
        self.cov = cov_s

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

        # Update posterior mean and covariance
        self.predict()

    def plot(self, train_x=None, train_y=None, samples=[]):
        """
        Plots mean and error bars for the GP, along with samples.

        Args:
            train_x: (m x d) matrix of m vectors in R^d for optional training data
            train_y: (m x 1) vector of m scalars representing target values
            samples: list of (n x 1) vectors of n samples from posterior distribution
        """

        # Distance for 95% confidence interval
        confidence_interval = 1.96 * jnp.sqrt(jnp.diag(self.cov))

        # Plot error barrs
        plt.fill_between(
            self.X_s.ravel(),
            self.mu.ravel() + confidence_interval,
            self.mu.ravel() - confidence_interval,
            alpha=0.2,
            color="b",
        )

        # Plot mean and samples
        plt.plot(self.X_s.ravel(), self.mu.ravel(), "r", lw=1, label="Mean")
        for i, sample in enumerate(samples):
            plt.plot(self.X_s.ravel(), sample, lw=1, label=f"Sample {i+1}")
        if train_x is not None:
            plt.plot(train_x.ravel(), train_y.ravel(), "rx", label="Training points")
        plt.legend()
        plt.show()
