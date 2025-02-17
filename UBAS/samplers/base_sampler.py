"""
base_samplers.py
==============================
Abstract base class for data samplers; functions which take a surrogate model,
the training data, and return the inputs which should augment the training set
(i.e., new training points).
"""
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class BaseSampler(ABC):
    """Abstract base class for data samplers."""

    @abstractmethod
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
        *args
            Additional arguments should be passed as keyword arguments
        **kwargs
            Additional arguments to `sample`: refer to each sample documentation for a list of
            all possible arguments
        Returns
        -------
        new_x : NDArray
            The new locations of training data which should augment the current training set of
            shape (`batch_samples`, n_dimensions)
        """

        pass
