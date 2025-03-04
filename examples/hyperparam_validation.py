"""
hyperparameter_validation.py
============================
A script to validate the hyperparameters chosen for the various functions by plotting the exact and predicted solutions,
and comparing the mean squared error of the neural nets.
"""

import os
import json
from UBAS.regressors.quant_nn_regressor import QuantNNRegressor
from UBAS.estimators.k_fold_quantile_estimator import KFoldQuantileRegressor
from sklearn.model_selection import KFold
from UBAS.generators.input_generator import InputGenerator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.generators.benchmark_function_generator import BenchmarkFunctionGenerator
from UBAS.generators.twin_peaks_generator import TwinPeakGenerator


def validate_hyperparams(function, function_name, dim, train_bounds, test_bounds, n_train_points, n_test_points):
    alpha = 0.2
    if dim < 3:
        batch_size = 64
    elif dim < 5:
        batch_size = 128
    else:
        batch_size = 512
    hyperparam_path = os.path.join("D:", os.sep, "UBAS", "projects", "param_opt", function_name, str(dim) + "D",
                                   "hyperparams.json")
    save_path = os.path.dirname(hyperparam_path)
    with open(hyperparam_path, 'r') as f:
        hyperparams = json.load(f)
    model = QuantNNRegressor(quantiles=[alpha / 2, 1 - alpha / 2], batch_size=batch_size, activation="relu",
                             **hyperparams)
    surrogate = KFoldQuantileRegressor(model, method="plus", cv=KFold(n_splits=5, shuffle=True), alpha=alpha, n_jobs=-1)

    input_sampler = InputGenerator(train_bounds, dim)
    inputs, targets = function.generate(input_sampler.uniformly_sample(n_train_points))

    test_sampler = InputGenerator(test_bounds, dim)
    test_inputs, test_targets = function.generate(test_sampler.uniformly_sample(n_test_points))

    surrogate.fit(inputs, targets, fit_median_estimator=False)
    test_preds, test_lb_ub = surrogate.predict(test_inputs)

    abs_error = np.abs(test_preds - test_targets)
    mse = np.mean(abs_error ** 2)

    print(f"{mse = }")

    combined_data = np.c_[test_inputs, test_targets, test_preds, abs_error]
    columns = ["X" + str(i+1) for i in range(dim)]
    columns.append("Exact")
    columns.append("Pred")
    columns.append("Abs. Error")
    fig, ax = plt.subplots(dim + 3, dim + 3, figsize=((dim+3) * 2, (dim+3) * 2))
    plt.title(f"{function_name} {dim}D hyperparameter_validation")
    pd.plotting.scatter_matrix(pd.DataFrame(combined_data, columns=columns), range_padding=0.25, ax=ax, edgecolors='white', s=20)
    plt.savefig(os.path.join(save_path, f"{n_train_points}_train_points_scatter_matrix.png"))


def batch_validate(functions, function_names, dims, train_bounds, test_bounds, n_train_points, n_test_points):
    for i, function in enumerate(functions):
        for j, dim in enumerate(dims):
            train_bound = np.ones((2, dim)) * train_bounds[i]
            test_bound = np.ones((2, dim)) * test_bounds[i]
            validate_hyperparams(function, function_names[i], dim, train_bound, test_bound, n_train_points[j],
                                 n_test_points[j])


def test_all():
    functions = [CentralPeakGenerator(-15), CentralPeakGenerator(-1),
                 BenchmarkFunctionGenerator("AMGM"), BenchmarkFunctionGenerator("Alpine02")]
    function_names = ["central_peak_exp_15", "central_peak_exp_1", "AMGM", "Alpine02"]

    dims = [2, 4, 8]
    train_bounds = [np.array([[-1, 1]]).T, np.array([[-1, 1]]).T, np.array([[0, 10]]).T, np.array([[0, 10]]).T]
    test_bounds = [np.array([[-0.5, 0.5]]).T, np.array([[-1, 1]]).T, np.array([[0, 10]]).T, np.array([[0, 10]]).T]
    n_train_points = [[50, 200, 700], [175, 400, 2500], [300, 600, 5000]]
    n_test_points = [1000, 10000, 10000]
    for train_points in n_train_points:
        batch_validate(functions, function_names, dims, train_bounds, test_bounds, train_points, n_test_points)


def test_alpine():
    functions = [BenchmarkFunctionGenerator("Alpine02")]
    function_names = ["Alpine02"]
    dims = [8]
    train_bounds = [np.array([[0, 10]]).T]
    test_bounds = [np.array([[0, 10]]).T]
    train_points = [5000]
    n_test_points = [10000]
    batch_validate(functions, function_names, dims, train_bounds, test_bounds, train_points, n_test_points)


def test_twin_peaks(dims, train_points, n_test_points):
    functions = [TwinPeakGenerator()]
    function_names = ["TwinPeak"]
    train_bounds = [np.array([[-0.75, 0.95]]).T]
    test_bounds = [np.array([[-0.75, 0.95]]).T]
    batch_validate(functions, function_names, dims, train_bounds, test_bounds, train_points, n_test_points)


if __name__ == "__main__":
    test_twin_peaks(dims=[4], train_points=[1000], n_test_points=[10000])
