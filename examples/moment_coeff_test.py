import numpy as np
from UBAS.generators.neural_foil_generator import NFGenerator
from UBAS.samplers.adap_exp_adaptive_sampler import AdapExpAdaptiveSampler
from UBAS.regressors.quant_nn_regressor import QuantNNRegressor
from UBAS.generators.input_generator import InputGenerator
from UBAS.plotters.base_plotter import BasePlotter
from UBAS.estimators.k_fold_quantile_estimator import KFoldQuantileRegressor
from UBAS.generators.twin_peaks_generator import TwinPeakGenerator
from UBAS.utils.plotting import load_performance_data
from copy import deepcopy
from sklearn.model_selection import KFold
from UBAS.regressors.gaussian_process_quantile_regressor import GaussianProcessQuantileRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import pickle
plt.rcParams.update({"font.size": 20})


def moment_coeffs():
    alpha = 0.2
    n_iterations = 40
    n_batch_points = [20, 10]
    n_initial_points = [20, 10]
    test_points = 100000
    n_p_samples = 500000
    track_values = ['mse', 'coverage', 'exponent', 'mean_relative_error']
    qois = ["CM", "CL", "CD"]
    airfoils = [None, np.array([0.02, 0.4, 0.12])]
    names = ["5D", "2D"]
    input_names = [["M/100", "P/10", "TT/100", "Re", "Alpha"], ["Re", "Alpha"]]
    num_trials = 1

    Re_bounds = np.array([10 ** 6, 10 ** 8])
    alpha_bounds = np.array([0, 15])
    e_bounds = np.array([0, 0.08])
    p_bounds = np.array([0.2, 0.6])
    t_bounds = np.array([0.02, 0.15])

    b_2d = np.c_[Re_bounds, alpha_bounds]
    b_5d = np.c_[e_bounds, p_bounds, t_bounds, Re_bounds, alpha_bounds]
    bounds = [b_5d, b_2d]
    dims = [5, 2]

    plotter = BasePlotter("", plotting_interval=1, plots=["scatter_matrix"],
                          input_names=None, target_name="Y", save_type="png")

    n_epochs = [[1933, 2733, 2733], [2200, 2200, 2466]]
    n_layers = [[4, 8, 4], [6, 8, 4]]
    neurons = [[200, 256, 200], [200, 256, 200]]

    for i, airfoil in enumerate(airfoils):
        for j, qoi in enumerate(qois):
            function = NFGenerator(qoi=qoi, airfoil=airfoil)
            un_scaled_bound = bounds[i]
            un_scaled_test_bound = bounds[i]
            scaler = MinMaxScaler()
            bound = scaler.fit_transform(un_scaled_bound)
            test_bound = scaler.transform(un_scaled_test_bound)
            dim = dims[i]
            input_sampler = InputGenerator(bound, dim)
            test_sampler = InputGenerator(test_bound, dim)
            n_initial_point = n_initial_points[i]
            batch_points = n_batch_points[i]
            plot_sample = None
            mean_relative_error = False

            n_epoch = n_epochs[i][j]
            n_layer = n_layers[i][j]
            neuron = neurons[i][j]

            nn = QuantNNRegressor(quantiles=[alpha / 2, 1 - alpha / 2], layers=n_layer, neurons=neuron, activation="relu",
                                  n_epochs=n_epoch)

            nn_surrogate = KFoldQuantileRegressor(nn, method="plus", cv=KFold(n_splits=5, shuffle=True), alpha=alpha,
                                               n_jobs=7)

            gp_surrogate = GaussianProcessQuantileRegressor(quantiles=[alpha / 2, 1 - alpha / 2])

            for trial in range(num_trials):
                unscaled_inputs = scaler.inverse_transform(input_sampler.uniformly_sample(n_initial_point))
                unscaled_test_inputs = scaler.inverse_transform(test_sampler.uniformly_sample(int(test_points)))

                unscaled_inputs, init_targets = function.generate(unscaled_inputs)
                unscaled_test_inputs, test_targets = function.generate(unscaled_test_inputs)

                init_inputs = scaler.transform(unscaled_inputs)
                test_inputs = scaler.transform(unscaled_test_inputs)

                path = os.path.join("D:", os.sep, "UBAS", "projects", "nn_gp",
                                    "moment_coeff", qoi, names[i])

                sampler = AdapExpAdaptiveSampler(os.path.join(path, "nn", f"trial_{trial + 1}"), dim,
                                                 deepcopy(nn_surrogate), function, bound, n_iterations, batch_points,
                                                 init_inputs, init_targets, test_inputs, test_targets, scaler,
                                                 intermediate_training=True, plotter=plotter, save_interval=5,
                                                 mean_relative_error=mean_relative_error, adaptive_batch_size=True,
                                                 n_p_samples=n_p_samples,
                                                 width_scaling='linear', starting_exponent=dim,
                                                 learning_rate=0.1, momentum_decay=0.25, adaptive_exponent_method="mom",
                                                 max_step=dim * 2, min_exp=1, max_exp=100)

                sampler.plotter.input_names = input_names[i]
                sampler.sample(track_values=track_values, plot_kwargs={"samples": plot_sample})

                sampler = AdapExpAdaptiveSampler(os.path.join(path, "gp", f"trial_{trial + 1}"), dim,
                                                 deepcopy(gp_surrogate), function, bound, n_iterations, batch_points,
                                                 init_inputs, init_targets, test_inputs, test_targets, scaler,
                                                 intermediate_training=True, plotter=plotter, save_interval=5,
                                                 mean_relative_error=mean_relative_error, adaptive_batch_size=False,
                                                 n_p_samples=n_p_samples,
                                                 width_scaling='linear', starting_exponent=dim,
                                                 learning_rate=0.1, momentum_decay=0.25, adaptive_exponent_method="mom",
                                                 max_step=dim * 2, min_exp=1, max_exp=100)

                sampler.plotter.input_names = input_names[i]

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
    #results = ["mse", "mean_relative_error", "max_absolute_error", "mean_width", "max_width", "coverage", "exponent"]
    #dims = [4]
    #function_names = ["TwinPeak"]
    #n_trials = 3
    #plot_results(results, dims, function_names, n_trials)
    moment_coeffs()
