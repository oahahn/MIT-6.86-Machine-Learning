"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
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
        for j in range(K):
            likelihood = gaussian_likelihood(X[i], mixture.mu[j], mixture.var[j])
            soft_counts[i, j] = mixture.p[j]*likelihood
        total_counts = soft_counts[i, :].sum()
        soft_counts[i, :] = soft_counts[i, :] / total_counts
        ll += np.log(total_counts)

    return soft_counts, ll


def gaussian_likelihood(x, mean, var):
    """Computes the likelihood of x being generated from a multi-dimensional 
    Gaussian with uncorrelated components
    
    Args: 
        x: (d, ) the feature vector
        mean: (d, ) mean of the Gaussian
        var: scalar, variance of the Gaussian

    Returns:
        float: the likelihood
    """
    d = len(x)
    result = 1/((2 * np.pi * var)**(d / 2.0))
    exponent = (-1/(2*var)) * ((x - mean)**2).sum()
    return result * np.exp(exponent)


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    mu_hat = np.zeros((K, d))
    var_hat = np.zeros(K)

    n_hat = post.sum(axis=0) # 1 x K
    p_hat = n_hat / n # 1 x K

    for j in range(K):
        # Computing the mean for each cluster j = 1,...,K
        # Multiply each x-vector by the soft counts for this cluster j
        px = X * post[:, j, None] # n x d
        # Sum these over the n data points
        px = px.sum(axis=0) # 1 x d
        # Normalise the values
        mu_hat[j] = px / n_hat[j] # 1 x d

        # Computing the variance for each cluster j = 1,...,K
        # Taking the difference of the x-values and the mean squared
        sqr_diff = (X-mu_hat[j])**2 # n x d
        # Multiplying each value by the soft counts for this cluster j
        p_sqr_diff = sqr_diff * post[:, j, None]  # n x d
        # Summing over all the n's and d's
        p_sqr_diff = p_sqr_diff.sum() # scalar
        # Normalising the values
        var_hat[j] = p_sqr_diff / (d * n_hat[j]) # scalar

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
        mixture = mstep(X, soft_counts)

    return mixture, soft_counts, ll
