"""
adap_exp_tests.py
===========================
A script to evaluate methods to determine the pmf exponent for adaptive sampling approaches on
several different datasets
"""
import numpy as np
from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.generators.benchmark_function_generator import BenchmarkFunctionGenerator
from UBAS.samplers.adap_exp_adaptive_sampler import AdapExpAdaptiveSampler
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
import pickle
plt.rcParams.update({"font.size": 20})

def test_adap_fixed_exp():
    alpha = 0.2
    n_iterations = 40
    n_batch_points = [20, 40, 60]
    n_initial_points = [200, 400, 800]
    test_points = 100000
    n_p_samples = 2000000
    track_values_mre = ["mse", "coverage", "exponent", "mean_relative_error"]
    track_values_no_mre = ["mse", "coverage", "exponent"]
    n_epochs = 1250
    batch_sizes = [128, 256, 512]
    plot_samples = [None, None, 750]

    functions = [CentralPeakGenerator(-15), CentralPeakGenerator(-1),
                 BenchmarkFunctionGenerator("AMGM"), BenchmarkFunctionGenerator("Alpine02")]
    function_names = ["central_peak_exp_15", "central_peak_exp_1", "AMGM", "Alpine02"]
    dims = [2, 4, 8]
    num_trials = 3
    bounds = [np.array([[-1, 1]]).T, np.array([[-1, 1]]).T, np.array([[0, 10]]).T, np.array([[0, 10]]).T]
    test_bounds = [np.array([[-0.1, 0.1]]).T, np.array([[-1, 1]]).T, np.array([[0, 10]]).T, np.array([[0, 10]]).T]
    mean_relative_errors = [True, False, False, False]

    plotter = BasePlotter("", plotting_interval=5, plots=["scatter_matrix"],
                            input_names=None, target_name="Y", save_type="png")

    nn = QuantNNRegressor(quantiles=[alpha/2, 1-alpha/2], layers=4, neurons=128, activation="relu", no_epochs=10)

    surrogate = KFoldQuantileRegressor(nn, method="plus", cv=KFold(n_splits=5, shuffle=True), alpha=alpha, n_jobs=7)

    for i, function in enumerate(functions):
        mean_relative_error = mean_relative_errors[i]
        if mean_relative_error is True:
            track_values = track_values_mre
        else:
            track_values = track_values_no_mre

        for j, dim in enumerate(dims):
            bound = (np.ones((2, dim)) * bounds[i])
            test_bound = (np.ones((2, dim)) * test_bounds[i])
            input_sampler = InputGenerator(bound, dim)
            test_sampler = InputGenerator(test_bound, dim)
            n_initial_point = n_initial_points[j]
            batch_points = n_batch_points[j]
            batch_size = batch_sizes[j]
            plot_sample = plot_samples[j]

            for trial in range(num_trials):
                init_inputs, init_targets = function.generate(input_sampler.uniformly_sample(n_initial_point))
                test_inputs, test_targets = function.generate(test_sampler.uniformly_sample(test_points))

                path = os.path.join("D:", os.sep, "UBAS", "projects", "adap_exp_fixed",
                                    function_names[i], str(dim) + "D")

                sampler = AdapExpAdaptiveSampler(os.path.join(path, "adap_exp", f"trial_{trial + 1}"), dim,
                                                 deepcopy(surrogate), function, bound, n_iterations, batch_points,
                                                 init_inputs, init_targets, test_inputs, test_targets,
                                                 intermediate_training=True, plotter=plotter, save_interval=5,
                                                 mean_relative_error=mean_relative_error, n_p_samples=n_p_samples,
                                                 width_scaling='linear', starting_exponent=dim ** 2 / 2,
                                                 learning_rate=0.1, momentum_decay=0.1, adaptive_exponent_method="mom",
                                                 max_step=dim * 2, min_exp=1, max_exp=100)

                sampler.plotter.input_names = ["X" + str(j + 1) for j in range(dim)]
                sampler.sample(track_values=track_values, plot_kwargs={"samples": plot_sample},
                               fit_kwargs={"n_epochs": n_epochs, "batch_size": batch_size})

                sampler = AdaptiveSampler(os.path.join(path, "fixed_exp", f"trial_{trial + 1}"), dim,
                                          deepcopy(surrogate), function, bound, n_iterations, batch_points,
                                          init_inputs, init_targets, test_inputs, test_targets,
                                          intermediate_training=True, plotter=plotter, save_interval=5,
                                          mean_relative_error=mean_relative_error, n_p_samples=n_p_samples,
                                          width_scaling='linear', starting_exponent=dim ** 2 / 2)

                sampler.plotter.input_names = ["X" + str(j+1) for j in range(dim)]

                sampler.sample(track_values=track_values, plot_kwargs={"samples": plot_sample},
                               fit_kwargs={"n_epochs": n_epochs, "batch_size": batch_size})

                plt.close('all')

def plot_results(results, dims, function_names, n_trials):
    n_samples = []
    fixed_solution_dict = {}
    adap_exp_solution_dict = {}
    for result in results:
        fixed_solution_dict[result] = []
        adap_exp_solution_dict[result] = []

    BASE_PATH = os.path.join("D:", os.sep, "UBAS", "projects", "adap_exp_fixed")
    for function in function_names:
        for dim in dims:
            SAVE_PATH = os.path.join(BASE_PATH, function, str(dim)+"D")
            for result in results:
                fixed_solution_dict[result] = []
                adap_exp_solution_dict[result] = []

            for trial in range(n_trials):
                fixed_df = load_performance_data(os.path.join(SAVE_PATH, "fixed_exp",
                                                               f"trial_{trial + 1}", "performance_data.json"))
                adap_exp_df = load_performance_data(os.path.join(SAVE_PATH, "adap_exp",
                                                                 f"trial_{trial + 1}", "performance_data.json"))

                n_samples = fixed_df["n_samples"]
                for result in results:
                    fixed_solution_dict[result].append(fixed_df[result])
                    adap_exp_solution_dict[result].append(adap_exp_df[result])

            for result in results:
                PLOT_PATH = os.path.join(SAVE_PATH, f"{function}_{dim}D_{result}_fixed_adap_comparison.png")
                fixed_solution = np.array(fixed_solution_dict[result])
                adap_exp_solution = np.array(adap_exp_solution_dict[result])
                fixed_mean = np.mean(fixed_solution, axis=0)
                adap_exp_mean = np.mean(adap_exp_solution, axis=0)

                fig, ax = plt.subplots(figsize=(8, 8))
                for i, sol in enumerate(fixed_solution):
                    ax.plot(n_samples, sol, color="black", alpha=0.2, linestyle='dashed')
                    ax.plot(n_samples, adap_exp_solution[i], color='blue', alpha=0.2, linestyle='solid')
                ax.plot(n_samples, fixed_mean, color='black', linestyle='dashed', label='Fixed Exponent')
                ax.plot(n_samples, adap_exp_mean, color='blue', linestyle='solid', label='Adaptive Exponent')
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
    #test_adap_fixed_exp()
    results = ["mse", "max_absolute_error", "mean_width", "max_width", "coverage", "exponent"]
    dims = [2, 4, 8]
    function_names = ["central_peak_exp_15", "central_peak_exp_1", "AMGM", "Alpine02"]
    n_trials = 3
    plot_results(results, dims, function_names, n_trials)
