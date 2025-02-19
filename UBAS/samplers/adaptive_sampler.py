"""
adaptive_sampler.py
===================
Subclass of `BaseSampler`, and is a base class for all types of adaptive samplers.
The important change is the reimplementation of the sample_step() function which determines how new points are
generated.
"""
import numpy as np
from UBAS.samplers.base_sampler import BaseSampler
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt

class AdaptiveSampler(BaseSampler):
    """Adaptive Sampling Parent Class"""
    def __init__(self, dimension, surrogate, generator, bounds, n_iterations, n_batch_points,
                 initial_inputs, initial_targets, test_inputs=None, test_targets=None, intermediate_training=False,
                 plotter=None, save_interval=5, mean_relative_error=False, n_p_samples=10000):
        """
        Class Initialization. Check BaseSampler documentation for parameter descriptions

        Parameters
        ----------
        n_p_samples : int default=10000
            The number of samples uniformly generated within the bounds to approximate the probability distribution
            from which samples are drawn.
        """
        super().__init__(dimension, surrogate, generator, bounds, n_iterations, n_batch_points,
                         initial_inputs, initial_targets, test_inputs, test_targets,
                         intermediate_training, plotter, save_interval, mean_relative_error)
        self.n_p_samples = n_p_samples

    def sampling_step(self, n_batch_points) -> NDArray:
        """
        Override of sampling_step method in BaseSampler which uses the surrogate to predict `n_p_samples` values
        and draw from the PMF proportional to their widths.

        Parameters
        ----------
        n_batch_points : int
            The number of points to sample in one call of sampling_step

        Returns
        -------
        new_x : NDArray
            The new adaptively sampled inputs
        """
        probability_eval_points = self.sampler.uniformly_sample(self.n_p_samples)
        p_preds, p_bounds = self.predict(probability_eval_points)
        widths = p_bounds[:, 1] - p_bounds[:, 0]
        p = np.where(widths < 0, 0, widths)
        p = p ** 4 / np.sum(p ** 4)

        new_x = self.sampler.adaptively_mc_sample(probability_eval_points, p, n_batch_points)
        return new_x
