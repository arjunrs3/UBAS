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
from UBAS.utils.plotting import load_performance_data
from copy import deepcopy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
plt.rcParams.update({"font.size": 20})

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


def plot_results(results, dims, function_names, n_trials):
    n_samples = []
    linear_solution_dict = {}
    logistic_solution_dict = {}
    for result in results:
        linear_solution_dict[result] = []
        logistic_solution_dict[result] = []

    BASE_PATH = os.path.join("D:", os.sep, "UBAS", "projects", "linear_logistic")
    for function in function_names:
        for dim in dims:
            SAVE_PATH = os.path.join(BASE_PATH, function, str(dim)+"D")

            for result in results:
                linear_solution_dict[result] = []
                logistic_solution_dict[result] = []

            for trial in range(n_trials):
                linear_df = load_performance_data(os.path.join(SAVE_PATH, "linear",
                                                               f"trial_{trial + 1}", "performance_data.json"))
                logistic_df = load_performance_data(os.path.join(SAVE_PATH, "logistic",
                                                                 f"trial_{trial + 1}", "performance_data.json"))

                n_samples = linear_df["n_samples"]
                for result in results:
                    linear_solution_dict[result].append(linear_df[result])
                    logistic_solution_dict[result].append(logistic_df[result])

            for result in results:
                PLOT_PATH = os.path.join(SAVE_PATH, f"{function}_{dim}D_{result}_scaling_comparison.png")
                linear_solution = np.array(linear_solution_dict[result])
                logistic_solution = np.array(logistic_solution_dict[result])
                linear_mean = np.mean(linear_solution, axis=0)
                logistic_mean = np.mean(logistic_solution, axis=0)

                fig, ax = plt.subplots(figsize=(8, 8))
                for i, sol in enumerate(linear_solution):
                    ax.plot(n_samples, sol, color="blue", alpha=0.2, linestyle='solid')
                    ax.plot(n_samples, logistic_solution[i], color='black', alpha=0.2, linestyle='dashed')
                ax.plot(n_samples, linear_mean, color='blue', linestyle='solid', label='Linear Scaling')
                ax.plot(n_samples, logistic_mean, color='black', linestyle='dashed', label='Logistic scaling')
                ax.set_xlabel("Number of Samples")
                ax.set_ylabel(result)
                ax.legend()
                ax.set_title(f"{result}: {dim}D {function}")
                if result in ["mse", "mean_relative_error", "max_absolute_error"]:
                    ax.semilogy()
                plt.tight_layout()
                plt.savefig(PLOT_PATH)
                plt.show()


if __name__ == "__main__":
    #test_linear_vs_logistic()
    results = ["mse", "max_absolute_error", "mean_width", "max_width", "coverage"]
    dims = [2, 4, 8]
    function_names = ["central_peak_exp_40", "central_peak_exp_1", "AMGM", "Alpine02"]
    n_trials = 3
    plot_results(results, dims, function_names, n_trials)
