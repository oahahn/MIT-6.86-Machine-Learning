"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n = X.shape[0]
    K = mixture.mu.shape[0]
    soft_counts = np.zeros((n, K))
    ll = 0

    for i in range(n):
        # Creates an array of False values wherever a 0 occurs
        non_zero_X = (X[i, :] != 0)
        for j in range(K):
            # [i, non_zero_X] creates a new array of non-zero values
            log_likelihood = log_gaussian(X[i, non_zero_X], mixture.mu[j, non_zero_X],
                                      mixture.var[j])
            # Performing computations in the log domain to help with numerical stability
            soft_counts[i, j] = np.log(mixture.p[j] + 1e-16) + log_likelihood
        # Using logsumexp to help ensure numerical stability
        total_counts = logsumexp(soft_counts[i, :])
        soft_counts[i, :] = soft_counts[i, :] - total_counts
        ll += total_counts

    return np.exp(soft_counts), ll


def log_gaussian(x, mean, var):
    """Computes the values for the log of the posterior probability

    Args:
        x: (d, ) the feature vector
        mean: (d, ) mean of the Gaussian
        var: scalar, variance of the Gaussian

    Returns:
        float: the likelihood
    """
    d = len(x)
    log_result = -d / 2.0 * np.log(2 * np.pi * var)
    log_result -= ((x - mean)**2).sum() / (2 * var)
    return log_result


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    mu_hat = np.zeros((K, d))
    var_hat = np.zeros(K)

    n_hat = post.sum(axis=0)  # 1 x K
    p_hat = n_hat / n  # 1 x K

    for j in range(K):
        var_numer, var_denom = 0, 0
        for l in range(d):
            # Converts the lth x coordinate into a vector of booleans
            # False for every 0 entry and True otherwise
            non_zero_X = (X[:, l] != 0) # n x 1
            # Extract the soft counts corresponding to non-zero entries
            # for this cluster j
            non_zero_post = post[non_zero_X, j] # vector <= n in length
            post_sum = non_zero_post.sum()
            # Only update the mean when it is supported by at least one
            # full point to avoid erratic results
            if (post_sum >= 1):
                # Multiplying the non-zero vector by its soft counts and normalising
                mu_hat[j, l] = (X[non_zero_X, l] @ post[non_zero_X, j]) / post_sum
            # Computing the variance for each cluster j = 1,...,K
            var_numer += ((X[non_zero_X, l] - mu_hat[j, l])**2) @ post[non_zero_X, j]
            var_denom += post_sum
        var_hat[j] = var_numer / var_denom
        # To avoid the variances going to zero a minimum variance is assigned
        if var_hat[j] < min_variance:
            var_hat[j] = min_variance

    return GaussianMixture(mu_hat, var_hat, p_hat)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    last_ll = None
    ll = None
    while (last_ll is None or ll - last_ll > 1e-6 * np.abs(ll)):
        last_ll = ll
        soft_counts, ll = estep(X, mixture)
        mixture = mstep(X, soft_counts, mixture)

    return mixture, soft_counts, ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
