"""
uniform_sampler.py
===============================
A sampler which randomly chooses samples between the provided bounds
"""

import numpy as np
from numpy.typing import NDArray
from UBAS.samplers.base_sampler import BaseSampler


class UniformSampler(BaseSampler):
    def __init__(self):
        pass

    def sample(self, bounds, batch_samples=1, *args, **kwargs) -> NDArray:
        """

        Parameters
        ----------
        bounds : NDArray
            The bounds within which to sample. Should be an NDArray of shape (2, n_dimensions) where
            the first element of the 0th axis are the lower bounds and the second element are the upper
            bounds for each dimension.
        batch_samples : int, default=1
            The number of additional samples to return in one call of `sample`
        **args
            Additional arguments should be passed as keyword arguments
        **kwargs
            Additional keyword arguments are ignored for `UniformSampler`

        Returns
        -------
        new_x : NDArray
            The new locations of training data which should augment the current training set of
            shape (`batch_samples`, n_dimensions)
        """
        n_dimensions = bounds.shape[1]

        rng = np.random.default_rng()
        new_x_unscaled = rng.random((batch_samples, n_dimensions))
        new_x = new_x_unscaled * (bounds[1] - bounds[0]) + np.ones((batch_samples, n_dimensions)) * bounds[0]

        return new_x
