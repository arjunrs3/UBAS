"""
central_peak.py
===============
Performs sampling on the central peak function in various dimensions
"""
import numpy as np
from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.samplers.base_sampler import BaseSampler
from UBAS.regressors.quant_nn import QuantNN
from UBAS.generators.input_generator import InputGenerator
from UBAS.samplers.adaptive_sampler import AdaptiveSampler
from copy import deepcopy
import matplotlib.pyplot as plt


def sample_central_peak(dimension, n_iterations, n_batch_points, n_initial_points, n_test_points):
    """Sampling of the central peak function"""

    # Problem setup and initial point generation
    sampling_bounds = (np.ones((2, dimension)) * np.array([[-1, 1]]).T)
    test_bounds = (np.ones((2, dimension)) * np.array([[-0.1, 0.1]]).T)

    initial_point_sampler = InputGenerator(sampling_bounds, dimension)
    test_point_sampler = InputGenerator(test_bounds, dimension)

    generator = CentralPeakGenerator()
    initial_inputs, initial_targets = generator.generate(initial_point_sampler.uniformly_sample(n_initial_points))
    test_inputs, test_targets = generator.generate(test_point_sampler.uniformly_sample(n_test_points))

    # Initialize the uniform sampler
    u_surrogate = QuantNN(input_dim=dimension, output_dim=2, hidden_dim=64, num_layers=6,
                          quantiles=np.array([0.2, 0.8]), max_epochs=500, batch_size=500)

    a_surrogate = deepcopy(u_surrogate)

    u_sampler = BaseSampler(dimension, u_surrogate, generator, sampling_bounds, n_iterations, n_batch_points,
                            initial_inputs, initial_targets, test_inputs, test_targets, intermediate_training=True,
                            mean_relative_error=True)

    a_sampler = AdaptiveSampler(dimension, a_surrogate, generator, sampling_bounds, n_iterations, n_batch_points,
                            initial_inputs, initial_targets, test_inputs, test_targets, intermediate_training=True,
                            mean_relative_error=True, n_p_samples=10000)

    # Perform sampling
    u_sampler.sample(filename="central_peak_test_uniform")

    a_sampler.sample(filename="central_peak_test_adaptive")
    # plot mean_relative_error vs. number of samples
    mre_u = [perf.mean_relative_error for perf in u_sampler.model_performance]
    mre_a = [perf.mean_relative_error for perf in a_sampler.model_performance]
    n_samples = np.arange(n_iterations + 1) * n_batch_points + n_initial_points

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Mean Relative Error')
    ax.plot(n_samples, mre_u, linestyle='dashed', color='black', label='uniform')
    ax.plot(n_samples, mre_a, linestyle='solid', color='blue', label='adaptive')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    sample_central_peak(dimension=2, n_iterations=15, n_batch_points=30, n_initial_points=100, n_test_points=1000)
