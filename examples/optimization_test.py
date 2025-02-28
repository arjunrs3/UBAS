import numpy as np
from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.generators.benchmark_function_generator import BenchmarkFunctionGenerator
from UBAS.samplers.adap_exp_adaptive_sampler import AdapExpAdaptiveSampler
from UBAS.regressors.quant_nn_regressor import QuantNNRegressor
from UBAS.generators.input_generator import InputGenerator
from UBAS.plotters.base_plotter import BasePlotter
from UBAS.samplers.adaptive_sampler import BaseSampler
from UBAS.estimators.k_fold_quantile_estimator import KFoldQuantileRegressor
from UBAS.utils.plotting import load_performance_data
from copy import deepcopy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from UBAS.utils.find_maxima import find_local_maxima
import pickle
plt.rcParams.update({"font.size": 20})

def test_optimization():
    alpha = 0.2
    n_iterations = 40
    n_batch_points = [20, 40]
    n_initial_points = [200, 400]
    test_points = 100000
    n_p_samples = 2000000
    track_values_mre = ["mse", "coverage", "exponent", "mean_relative_error"]
    track_values_no_mre = ["mse", "coverage", "exponent"]
    track_values_uniform = ['mse', 'coverage']
    n_epochs = 750
    batch_sizes = [128, 256]
    plot_samples = [None, None]

    functions = [CentralPeakGenerator(-15), BenchmarkFunctionGenerator("Alpine02")]
    function_names = ["central_peak_exp_15", "Alpine02"]
    dims = [2, 4]
    num_trials = 1
    bounds = [np.array([[-1, 1]]).T, np.array([[0, 10]]).T]
    test_bounds = [np.array([[-0.1, 0.1]]).T, np.array([[0, 10]]).T]
    mean_relative_errors = [True, False]

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

                path = os.path.join("D:", os.sep, "UBAS", "projects", "opt_tests",
                                    function_names[i], str(dim) + "D")

                sampler = AdapExpAdaptiveSampler(os.path.join(path, "adap", f"trial_{trial + 1}"), dim,
                                                 deepcopy(surrogate), function, bound, n_iterations, batch_points,
                                                 init_inputs, init_targets, test_inputs, test_targets,
                                                 intermediate_training=True, plotter=plotter, save_interval=5,
                                                 mean_relative_error=mean_relative_error, n_p_samples=n_p_samples,
                                                 width_scaling='linear', starting_exponent=dim ** 2 / 2, mode='max',
                                                 learning_rate=0.1, momentum_decay=0.1, adaptive_exponent_method="mom",
                                                 max_step=dim * 2, min_exp=1, max_exp=100)

                sampler.plotter.input_names = ["X" + str(j + 1) for j in range(dim)]
                sampler.sample(track_values=track_values, plot_kwargs={"samples": plot_sample},
                               fit_kwargs={"n_epochs": n_epochs, "batch_size": batch_size})

                sampler = BaseSampler(os.path.join(path, "unif", f"trial_{trial + 1}"), dim,
                                          deepcopy(surrogate), function, bound, n_iterations, batch_points,
                                          init_inputs, init_targets, test_inputs, test_targets,
                                          intermediate_training=True, plotter=plotter, save_interval=5,
                                          mean_relative_error=mean_relative_error)

                sampler.plotter.input_names = ["X" + str(j+1) for j in range(dim)]

                sampler.sample(track_values=track_values_uniform, plot_kwargs={"samples": plot_sample},
                               fit_kwargs={"n_epochs": n_epochs, "batch_size": batch_size})

                plt.close('all')

def evaluate_maxima(dims, function_names, n_trials):
    bounds = [np.array([[-1, 1]]).T, np.array([[0, 10]]).T]
    for i, function in enumerate(function_names):
        for dim in dims:
            bound = (np.ones((2, dim)) * bounds[i])
            path = os.path.join("D:", os.sep, "UBAS", "projects", "opt_tests",
                                function_names[i], str(dim) + "D")
            adap_solutions = []
            unif_solutions = []

            for trial in range(n_trials):
                adap_path = os.path.join(path, "adap", f"trial_{trial + 1}", "sampler_data.pkl")
                unif_path = os.path.join(path, "unif", f"trial_{trial + 1}", "sampler_data.pkl")

                with open(adap_path, 'rb') as f:
                    adap_samp = pickle.load(f)

                with open(unif_path, 'rb') as f:
                    unif_samp = pickle.load(f)

                adap_surrogate = adap_samp.surrogate
                unif_surrogate = unif_samp.surrogate

                unif_extrema, unif_extrema_locations = find_local_maxima(unif_surrogate, bound, n_extrema=5)
                adap_extrema, adap_extrema_locations = find_local_maxima(adap_surrogate, bound, n_extrema=5)
                unif_solutions.append([unif_extrema, unif_extrema_locations])
                adap_solutions.append([adap_extrema, adap_extrema_locations])
                print(f"{unif_solutions = }")
                print(f"{adap_solutions = }")


if __name__ == "__main__":
    dims = [2, 4]
    function_names = ["central_peak_exp_15", "Alpine02"]
    n_trials = 1
    evaluate_maxima(dims, function_names, n_trials)
