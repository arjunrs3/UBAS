"""
linear_vs_logistic.py
===========================
A script to evaluate linear vs. affine scaling for adaptive sampling approaches on several different datasets
"""
import numpy as np
from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.generators.benchmark_function_generator import BenchmarkFunctionGenerator
from UBAS.samplers.base_sampler import BaseSampler
from UBAS.regressors.quant_nn_regressor import QuantNNRegressor
from UBAS.generators.input_generator import InputGenerator
from UBAS.plotters.base_plotter import BasePlotter
from UBAS.samplers.adaptive_sampler import AdaptiveSampler
from UBAS.estimators.k_fold_quantile_estimator import KFoldQuantileRegressor
from copy import deepcopy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os


def test_linear_vs_logistic():
    alpha = 0.2
    n_iterations = 40
    batch_points = 20
    n_initial_points = [200, 400, 800]
    test_points = 100000
    n_p_samples = 2000000
    track_values = ["mse", "coverage"]
    n_epochs = 1250
    batch_sizes = [128, 256, 512]
    plot_samples = [None, None, 750]

    functions = [CentralPeakGenerator(-40), CentralPeakGenerator(-1),
                 BenchmarkFunctionGenerator("AMGM"), BenchmarkFunctionGenerator("Alpine02")]
    function_names = ["central_peak_exp_40", "central_peak_exp_1", "AMGM", "Alpine02"]
    dims = [2, 4, 8]
    num_trials = 3
    scaling_methods = ["linear", "logistic"]
    bounds = [np.array([[-1, 1]]).T, np.array([[-1, 1]]).T, np.array([[0, 10]]).T, np.array([[0, 10]]).T]

    plotter = BasePlotter("", plotting_interval=5, plots=["scatter_matrix"],
                            input_names=None, target_name="Y", save_type="png")

    nn = QuantNNRegressor(quantiles=[alpha/2, 1-alpha/2], layers=4, neurons=128, activation="relu", no_epochs=10)

    surrogate = KFoldQuantileRegressor(nn, method="plus", cv=KFold(n_splits=5, shuffle=True), alpha=alpha, n_jobs=7)

    for i, function in enumerate(functions):
        for j, dim in enumerate(dims):
            bound = (np.ones((2, dim)) * bounds[i])
            input_sampler = InputGenerator(bound, dim)
            n_initial_point = n_initial_points[j]
            batch_size = batch_sizes[j]
            plot_sample = plot_samples[j]

            for trial in range(num_trials):
                init_inputs, init_targets = function.generate(input_sampler.uniformly_sample(n_initial_point))
                test_inputs, test_targets = function.generate(input_sampler.uniformly_sample(test_points))

                for scaling_method in scaling_methods:
                    plt.close('all')
                    path = os.path.join("D:", os.sep, "UBAS", "projects", "linear_logistic",
                                        function_names[i], str(dim) + "D", scaling_method, f"trial_{trial + 1}")

                    sampler = AdaptiveSampler(path, dim, deepcopy(surrogate), function, bound, n_iterations,
                                              batch_points, init_inputs, init_targets, test_inputs, test_targets,
                                              intermediate_training=True, plotter=plotter, save_interval=5,
                                              mean_relative_error=False, n_p_samples=n_p_samples,
                                              width_scaling=scaling_method, exponent=dim ** 2 / 2)

                    sampler.plotter.input_names = ["X" + str(j+1) for j in range(dim)]

                    sampler.sample(track_values=track_values, plot_kwargs={"samples": plot_sample},
                                   fit_kwargs={"n_epochs": n_epochs, "batch_size": batch_size})


if __name__ == "__main__":
    test_linear_vs_logistic()
