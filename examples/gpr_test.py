"""
gpr_test.py
===========
A simple test function to test adaptive samplers with gaussian process regression
"""

import numpy as np
from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.samplers.adap_exp_adaptive_sampler import AdapExpAdaptiveSampler
from UBAS.regressors.gaussian_process_quantile_regressor import GaussianProcessQuantileRegressor
from UBAS.generators.input_generator import InputGenerator
from UBAS.plotters.base_plotter import BasePlotter
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import pandas as pd
from mapie.metrics import regression_coverage_score
from pandas.plotting import scatter_matrix

def gpr_test():
    alpha = 0.2
    n_iterations = 10
    n_batch_points = 20
    n_initial_points = 20
    test_points = 100000
    n_p_samples = 500000
    track_values = ["mse", "coverage", "exponent", "mean_relative_error"]
    function = CentralPeakGenerator(-15)
    function_name = "central_peak_exp_15"
    dim = 2
    bounds = np.ones((2, dim)) * np.array([[-1, 1]]).T
    test_bounds = np.ones((2, dim)) * np.array([[-0.1, 0.1]]).T
    input_sampler = InputGenerator(bounds, dim)
    test_sampler = InputGenerator(test_bounds, dim)
    surrogate = GaussianProcessQuantileRegressor(quantiles=[alpha / 2, 1 - alpha / 2])
    path = os.path.join("D:", os.sep, "UBAS", "projects", "testing", "GPR_new", function_name, str(dim) + "D")

    init_inputs, init_targets = function.generate(input_sampler.uniformly_sample(n_initial_points))
    test_inputs, test_targets = function.generate(test_sampler.uniformly_sample(test_points))

    plotter = BasePlotter("", plotting_interval=2, plots=["scatter_matrix"], input_names=None,
                          target_name="Y", save_type="png")
    sampler = AdapExpAdaptiveSampler(path, dim,
                                     deepcopy(surrogate), function, bounds, n_iterations, n_batch_points,
                                     init_inputs, init_targets, test_inputs, test_targets,
                                     intermediate_training=True, plotter=plotter, save_interval=5,
                                     mean_relative_error=True, n_p_samples=n_p_samples,
                                     width_scaling='linear', starting_exponent=dim ** 2 / 2, mode='min_variance',
                                     learning_rate=0.1, momentum_decay=0.1, adaptive_exponent_method="mom",
                                     max_step=dim * 2, min_exp=1, max_exp=100)

    sampler.plotter.input_names = ["X" + str(j + 1) for j in range(dim)]
    sampler.sample(track_values=track_values)

    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    x_grid, y_grid = np.meshgrid(x, y)
    z_preds, z_bounds = sampler.surrogate.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
    z_grid = z_preds.reshape(x_grid.shape)

    x_exact, z_exact = sampler.generator.generate(np.c_[x_grid.ravel(), y_grid.ravel()])
    z_exact = z_exact.reshape(x_grid.shape)
    norm = plt.Normalize(vmin=min(z_grid.min(), z_exact.min()), vmax=max(z_grid.max(), z_exact.max()))

    fig, axes = plt.subplots(2, 1)
    axes[0].contourf(x_grid, y_grid, z_grid, cmap='viridis', norm=norm)
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].set_title('Approximated Solution')
    contour2 = axes[1].contourf(x_grid, y_grid, z_exact, cmap='viridis', norm=norm)
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].set_title('Exact Solution')
    fig.colorbar(contour2, ax=axes, shrink=0.8)
    plt.ioff()
    plt.show()

def gpr_test2():

    alpha = 0.2
    dim = 2
    n_points = 100
    x_train = np.random.rand(n_points, dim) * 2 - 1
    y_train = 1/np.linalg.norm(x_train, axis=1) ** 2
    gpr = GaussianProcessQuantileRegressor(quantiles=[0.1, 0.9])
    gpr.fit(x_train, y_train)

    x_test = np.random.rand(1000, dim) * 2 - 1
    z_exact = 1/np.linalg.norm(x_test, axis=1) ** 2

    preds, bounds = gpr.predict(x_test)
    widths = bounds[:, 1] - bounds[:, 0]
    df = pd.DataFrame(np.c_[x_test, preds, z_exact, widths])
    cols = ["X" + str(x) for x in range(dim)]
    cols.append("preds")
    cols.append("exact")
    cols.append("widths")
    df.columns = cols
    scatter_matrix(df, alpha=0.3)

    coverage = regression_coverage_score(z_exact, bounds[:, 0], bounds[:, 1])
    print (f"{coverage = }")
    plt.show()

def test_1D():
    alpha = 0.2
    n_points = 10
    x = np.random.rand(n_points, 1) * 2 - 1
    y = np.e ** (-np.abs(x)).ravel()
    gp = GaussianProcessQuantileRegressor(quantiles=[alpha / 2, 1 - alpha / 2])
    gp.fit(x, y)
    x_vals = np.linspace(-1, 1, 100).reshape(-1, 1)
    preds, bounds = gp.predict(x_vals)

    fig, ax = plt.subplots()
    ax.fill_between(x_vals.ravel(), bounds[:, 0], bounds[:, 1], interpolate=True, facecolor='gray')
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.plot(x_vals.ravel(), preds, linestyle='dashed', color='red', label='Predicted Mean')
    ax.scatter(x, y, color='black', label='Training Points')
    plt.show()


if __name__ == "__main__":
    test_1D()
