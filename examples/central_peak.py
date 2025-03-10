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
from UBAS.plotters.base_plotter import BasePlotter
from UBAS.samplers.adaptive_sampler import AdaptiveSampler
from UBAS.estimators.k_fold_quantile_estimator import KFoldQuantileRegressor
from copy import deepcopy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os


def sample_central_peak(dimension, n_iterations, n_batch_points, n_initial_points, n_test_points):
    """Sampling of the central peak function"""

    alpha = 0.2
    # Problem setup and initial point generation
    sampling_bounds = (np.ones((2, dimension)) * np.array([[-1, 1]]).T)
    test_bounds = (np.ones((2, dimension)) * np.array([[-0.1, 0.1]]).T)

    initial_point_sampler = InputGenerator(sampling_bounds, dimension)
    test_point_sampler = InputGenerator(test_bounds, dimension)

    generator = CentralPeakGenerator()
    initial_inputs, initial_targets = generator.generate(initial_point_sampler.uniformly_sample(n_initial_points))
    test_inputs, test_targets = generator.generate(test_point_sampler.uniformly_sample(n_test_points))

    base_folder = str(dimension) + "D_central_peak"

    u_plotter = BasePlotter("", plotting_interval=5,
                            plots=["scatter_matrix", "pred_vs_actual"],
                            input_names=["X" + str(j+1) for j in range(dimension)], target_name="Y", save_type="png")

    a_plotter = deepcopy(u_plotter)

    # Initialize the uniform sampler
    u_nn = QuantNN(input_dim=dimension, output_dim=2, hidden_dim=64, num_layers=6,
                          quantiles=np.array([alpha / 2, 1 - alpha / 2]), max_epochs=10, batch_size=500)

    u_surrogate = KFoldQuantileRegressor(u_nn, method="plus", cv=KFold(n_splits=5, shuffle=True), alpha=alpha, n_jobs=7)

    a_surrogate = deepcopy(u_surrogate)

    u_directory = os.path.join("D:", os.sep, "UBAS", "projects", base_folder, "Uniform")
    a_directory = os.path.join("D:", os.sep, "UBAS", "projects", base_folder, "Adaptive")

    u_sampler = BaseSampler(u_directory, dimension, u_surrogate, generator, sampling_bounds, n_iterations, n_batch_points,
                            initial_inputs, initial_targets, test_inputs, test_targets, intermediate_training=True,
                            plotter=u_plotter, save_interval=5, mean_relative_error=True)

    a_sampler = AdaptiveSampler(a_directory, dimension, a_surrogate, generator, sampling_bounds, n_iterations, n_batch_points,
                            initial_inputs, initial_targets, test_inputs, test_targets, intermediate_training=True,
                            plotter=a_plotter, save_interval=5, mean_relative_error=True, n_p_samples=1000000)

    # Perform sampling
    track_values = ["mean_relative_error", "mean_width", "coverage"]

    #u_sampler.sample(track_values=track_values, plot_kwargs={"samples": 250})

    a_sampler.sample(track_values=track_values, plot_kwargs={"samples": 250})

    # plot mean_relative_error vs. number of samples
    mre_u = [perf.mean_relative_error for perf in u_sampler.model_performance]
    mre_a = [perf.mean_relative_error for perf in a_sampler.model_performance]
    n_samples = np.arange(n_iterations + 1) * n_batch_points + n_initial_points

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Mean Relative Error')
    ax.semilogy(n_samples, mre_u, linestyle='dashed', color='black', label='uniform')
    ax.semilogy(n_samples, mre_a, linestyle='solid', color='blue', label='adaptive')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    sample_central_peak(dimension=10, n_iterations=20, n_batch_points=50, n_initial_points=5000, n_test_points=1000)
