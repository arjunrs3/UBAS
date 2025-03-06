import numpy as np
from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.generators.benchmark_function_generator import BenchmarkFunctionGenerator
from UBAS.samplers.adap_exp_adaptive_sampler import AdapExpAdaptiveSampler
from UBAS.regressors.quant_nn_regressor import QuantNNRegressor
from UBAS.generators.input_generator import InputGenerator
from UBAS.plotters.base_plotter import BasePlotter
from UBAS.samplers.adaptive_sampler import AdaptiveSampler
from UBAS.estimators.k_fold_quantile_estimator import KFoldQuantileRegressor
from UBAS.generators.twin_peaks_generator import TwinPeakGenerator
from UBAS.utils.plotting import load_performance_data
from copy import deepcopy
from sklearn.model_selection import KFold
from UBAS.regressors.gaussian_process_quantile_regressor import GaussianProcessQuantileRegressor
import matplotlib.pyplot as plt
import os
import pickle
plt.rcParams.update({"font.size": 20})


def test_central_peak_functions():
    alpha = 0.2
    n_iterations = 40
    n_batch_points = [[15, 15], [10, 15]]#[[10, 15, 15], [5, 10, 15]]
    n_initial_points = [[100, 200], [50, 100]]#[[50, 100, 200], [25, 50, 100]]
    test_points = 100000
    n_p_samples = 500000
    track_values = ['mse', 'coverage', 'exponent', 'mean_relative_error']
    functions = [CentralPeakGenerator(-15), CentralPeakGenerator(-1)]
    function_names = ["central_peak_exp_15", "central_peak_exp_1"]
    dims = [4, 6]
    num_trials = 1
    bounds = [np.array([[-1, 1]]).T, np.array([[-1, 1]]).T]
    test_bounds = [np.array([[-0.1, 0.1]]).T, np.array([[-0.5, 0.5]]).T]
    plotter = BasePlotter("", plotting_interval=5, plots=["scatter_matrix"],
                          input_names=None, target_name="Y", save_type="png")

    n_epochs = [[1933, 3000], [3000, 1933]]#[[866, 1933, 3000], [1666, 3000, 1933]]
    n_layers = [[7, 8], [5, 5]] #[[7, 7, 8], [8, 5, 5]]
    neurons = [[128, 128], [256, 224]]#[[224, 128, 128], [128, 256, 224]]

    for i, function in enumerate(functions):
        for j, dim in enumerate(dims):
            bound = (np.ones((2, dim)) * bounds[i])
            test_bound = (np.ones((2, dim)) * test_bounds[i])
            input_sampler = InputGenerator(bound, dim)
            test_sampler = InputGenerator(test_bound, dim)
            n_initial_point = n_initial_points[i][j]
            batch_points = n_batch_points[i][j]
            plot_sample = None
            mean_relative_error = True

            n_epoch = n_epochs[i][j]
            n_layer = n_layers[i][j]
            neuron = neurons[i][j]

            nn = QuantNNRegressor(quantiles=[alpha / 2, 1 - alpha / 2], layers=n_layer, neurons=neuron, activation="relu",
                                  n_epochs=n_epoch)

            nn_surrogate = KFoldQuantileRegressor(nn, method="plus", cv=KFold(n_splits=5, shuffle=True), alpha=alpha,
                                               n_jobs=7)

            gp_surrogate = GaussianProcessQuantileRegressor(quantiles=[alpha / 2, 1 - alpha / 2])

            for trial in range(num_trials):
                init_inputs, init_targets = function.generate(input_sampler.uniformly_sample(n_initial_point))
                test_inputs, test_targets = function.generate(test_sampler.uniformly_sample(int(test_points)))

                path = os.path.join("D:", os.sep, "UBAS", "projects", "nn_gp",
                                    function_names[i], str(dim) + "D")

                sampler = AdapExpAdaptiveSampler(os.path.join(path, "nn", f"trial_{trial + 1}"), dim,
                                                 deepcopy(nn_surrogate), function, bound, n_iterations, batch_points,
                                                 init_inputs, init_targets, test_inputs, test_targets,
                                                 intermediate_training=True, plotter=plotter, save_interval=5,
                                                 mean_relative_error=mean_relative_error, adaptive_batch_size=True,
                                                 n_p_samples=n_p_samples,
                                                 width_scaling='linear', starting_exponent=dim,
                                                 learning_rate=0.1, momentum_decay=0.25, adaptive_exponent_method="mom",
                                                 max_step=dim * 2, min_exp=1, max_exp=100)

                sampler.plotter.input_names = ["X" + str(j + 1) for j in range(dim)]
                sampler.sample(track_values=track_values, plot_kwargs={"samples": plot_sample})

                sampler = AdapExpAdaptiveSampler(os.path.join(path, "gp", f"trial_{trial + 1}"), dim,
                                                 deepcopy(gp_surrogate), function, bound, n_iterations, batch_points,
                                                 init_inputs, init_targets, test_inputs, test_targets,
                                                 intermediate_training=True, plotter=plotter, save_interval=5,
                                                 mean_relative_error=mean_relative_error, adaptive_batch_size=False,
                                                 n_p_samples=n_p_samples,
                                                 width_scaling='linear', starting_exponent=dim,
                                                 learning_rate=0.1, momentum_decay=0.25, adaptive_exponent_method="mom",
                                                 max_step=dim * 2, min_exp=1, max_exp=100)

                sampler.plotter.input_names = ["X" + str(j + 1) for j in range(dim)]

                sampler.sample(track_values=track_values, plot_kwargs={"samples": plot_sample})

                plt.close('all')
def test_twin_peaks():
    alpha = 0.2
    n_iterations = 40
    n_batch_points = [10, 15, 60]
    n_initial_points = [50, 100, 200]
    test_points = 100000
    n_p_samples = 500000
    track_values_mre = ["mse", "coverage", "exponent", "mean_relative_error"]
    track_values_no_mre = ["mse", "coverage", "exponent"]
    plot_samples = [None, None, None]
    functions = [TwinPeakGenerator()]
    function_names = ["TwinPeak"]
    #dims = [2, 4, 8]
    dims = [2, 4, 8]
    num_trials = 3
    bounds = [np.array([[-0.75, 0.95]]).T]
    test_bounds_1 = [np.array([[-0.75, -0.25]]).T]
    test_bounds_2 = [np.array([[0.25, 0.95]]).T]
    mean_relative_errors = [True]

    plotter = BasePlotter("", plotting_interval=5, plots=["scatter_matrix"],
                          input_names=None, target_name="Y", save_type="png")

    n_epochs = [2555, 2555, 1805]
    n_layers = [5, 5, 8]
    neurons = [160, 160, 256]

    for i, function in enumerate(functions):
        mean_relative_error = mean_relative_errors[i]
        if mean_relative_error is True:
            track_values = track_values_mre
        else:
            track_values = track_values_no_mre

        for j, dim in enumerate(dims):
            if dim == 4:
                bound = (np.ones((2, dim)) * bounds[i])
                test_bound_1 = (np.ones((2, dim)) * test_bounds_1[i])
                test_bound_2 = (np.ones((2, dim)) * test_bounds_2[i])
                input_sampler = InputGenerator(bound, dim)
                test_sampler_1 = InputGenerator(test_bound_1, dim)
                test_sampler_2 = InputGenerator(test_bound_2, dim)
                n_initial_point = n_initial_points[j]
                batch_points = n_batch_points[j]
                plot_sample = plot_samples[j]
                n_epoch = n_epochs[j]
                n_layer = n_layers[j]
                neuron = neurons[j]

                nn = QuantNNRegressor(quantiles=[alpha / 2, 1 - alpha / 2], layers=n_layer, neurons=neuron, activation="relu",
                                      n_epochs=n_epoch)

                nn_surrogate = KFoldQuantileRegressor(nn, method="plus", cv=KFold(n_splits=5, shuffle=True), alpha=alpha,
                                                   n_jobs=7)

                gp_surrogate = GaussianProcessQuantileRegressor(quantiles=[alpha / 2, 1 - alpha / 2])

                for trial in range(num_trials):
                    init_inputs, init_targets = function.generate(input_sampler.uniformly_sample(n_initial_point))
                    test_inputs_1, test_targets_1 = function.generate(test_sampler_1.uniformly_sample(int(test_points / 2)))
                    test_inputs_2, test_targets_2 = function.generate(test_sampler_2.uniformly_sample(int(test_points / 2)))
                    print (test_inputs_1)
                    test_inputs = np.append(test_inputs_1, test_inputs_2, axis=0)
                    print (test_inputs)
                    test_targets = np.append(test_targets_1, test_targets_2)
                    print (test_targets)

                    path = os.path.join("D:", os.sep, "UBAS", "projects", "nn_gp",
                                        function_names[i], str(dim) + "D")

                    sampler = AdapExpAdaptiveSampler(os.path.join(path, "nn", f"trial_{trial + 1}"), dim,
                                                     deepcopy(nn_surrogate), function, bound, n_iterations, batch_points,
                                                     init_inputs, init_targets, test_inputs, test_targets,
                                                     intermediate_training=True, plotter=plotter, save_interval=5,
                                                     mean_relative_error=mean_relative_error, adaptive_batch_size=True,
                                                     n_p_samples=n_p_samples,
                                                     width_scaling='linear', starting_exponent=dim,
                                                     learning_rate=0.1, momentum_decay=0.9, adaptive_exponent_method="mom",
                                                     max_step=dim * 2, min_exp=1, max_exp=100)

                    sampler.plotter.input_names = ["X" + str(j + 1) for j in range(dim)]
                    sampler.sample(track_values=track_values, plot_kwargs={"samples": plot_sample})

                    if dim != 8:
                        sampler = AdapExpAdaptiveSampler(os.path.join(path, "gp", f"trial_{trial + 1}"), dim,
                                                         deepcopy(gp_surrogate), function, bound, n_iterations, batch_points,
                                                         init_inputs, init_targets, test_inputs, test_targets,
                                                         intermediate_training=True, plotter=plotter, save_interval=5,
                                                         mean_relative_error=mean_relative_error, adaptive_batch_size=False,
                                                         n_p_samples=n_p_samples,
                                                         width_scaling='linear', starting_exponent=dim,
                                                         learning_rate=0.1, momentum_decay=0.9, adaptive_exponent_method="mom",
                                                         max_step=dim * 2, min_exp=1, max_exp=100)

                        sampler.plotter.input_names = ["X" + str(j + 1) for j in range(dim)]

                        sampler.sample(track_values=track_values, plot_kwargs={"samples": plot_sample})

                    plt.close('all')


def plot_results(results, dims, function_names, n_trials):
    n_samples = []
    nn_solution_dict = {}
    gp_solution_dict = {}
    for result in results:
        nn_solution_dict[result] = []
        gp_solution_dict[result] = []

    BASE_PATH = os.path.join("D:", os.sep, "UBAS", "projects", "nn_gp")
    for function in function_names:
        for dim in dims:
            SAVE_PATH = os.path.join(BASE_PATH, function, str(dim)+"D")
            for result in results:
                nn_solution_dict[result] = []
                gp_solution_dict[result] = []

            for trial in range(n_trials):
                nn_df = load_performance_data(os.path.join(SAVE_PATH, "nn",
                                                               f"trial_{trial + 1}", "performance_data.json"))
                gp_df = load_performance_data(os.path.join(SAVE_PATH, "gp",
                                                                 f"trial_{trial + 1}", "performance_data.json"))

                n_samples = nn_df["n_samples"]
                for result in results:
                    nn_solution_dict[result].append(nn_df[result])
                    gp_solution_dict[result].append(gp_df[result])

            for result in results:
                PLOT_PATH = os.path.join(SAVE_PATH, f"{function}_{dim}D_{result}_new_nn_gp.png")
                nn_solution = np.array(nn_solution_dict[result])
                gp_solution = np.array(gp_solution_dict[result])
                nn_mean = np.mean(nn_solution, axis=0)
                gp_mean = np.mean(gp_solution, axis=0)

                fig, ax = plt.subplots(figsize=(8, 8))
                for i, sol in enumerate(nn_solution):
                    ax.plot(n_samples, sol, color="black", alpha=0.2, linestyle='dashed')
                    ax.plot(n_samples, gp_solution[i], color='blue', alpha=0.2, linestyle='solid')
                ax.plot(n_samples, nn_mean, color='black', linestyle='dashed', label='Neural Network')
                ax.plot(n_samples, gp_mean, color='blue', linestyle='solid', label='Gaussian Process')
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
    results = ["mse", "mean_relative_error", "max_absolute_error", "mean_width", "max_width", "coverage", "exponent"]
    dims = [4]
    function_names = ["TwinPeak"]
    n_trials = 3
    plot_results(results, dims, function_names, n_trials)
    #test_twin_peaks()
    #test_central_peak_functions()