"""
hyperparam_optimization.py
==========================
A script to determine the optimal hyperparameters for neural networks being trained on several different function
generators.
"""

from sklearn.model_selection import RandomizedSearchCV
from UBAS.regressors.quant_nn_regressor import QuantNNRegressor
from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.generators.benchmark_function_generator import BenchmarkFunctionGenerator
from UBAS.generators.input_generator import InputGenerator
from rich.progress import track
import os
import numpy as np
import json


param_dist = {"n_epochs": np.linspace(250, 2000, 10, dtype=int),
              "layers": np.linspace(5, 8, 4, dtype=int),
              "neurons": np.linspace(128, 256, 5, dtype=int)}

model = QuantNNRegressor(quantiles=[0.1, 0.9], layers=4, neurons=128, activation="relu", n_epochs=15, batch_size=512)

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=7, verbose=3)

functions = [CentralPeakGenerator(-15), CentralPeakGenerator(-1),
                 BenchmarkFunctionGenerator("AMGM"), BenchmarkFunctionGenerator("Alpine02")]

function_names = ["central_peak_exp_15", "central_peak_exp_1", "AMGM", "Alpine02"]
# dims = [2, 4]
dims = [8]
bounds = [np.array([[-1, 1]]).T, np.array([[-1, 1]]).T, np.array([[0, 10]]).T, np.array([[0, 10]]).T]
#n_points = [600, 1500]
n_points = [5000]
for i, function in enumerate(track(functions, description="Function Progress")):
    for j, dim in enumerate(dims):
        SAVE_PATH = os.path.join("D:", os.sep, "UBAS", "projects", "param_opt", function_names[i], str(dim) + "D")
        os.makedirs(SAVE_PATH, exist_ok=True)
        path = os.path.join(SAVE_PATH, "hyperparams.json")
        bound = np.ones((2, dim)) * bounds[i]
        input_sampler = InputGenerator(bound, dim)

        inputs, target = function.generate(input_sampler.uniformly_sample(n_points[j]))
        random_search.fit(inputs, target)

        with open(path, 'w') as f:
            json.dump({k: v.tolist() for k, v in random_search.best_params_.items()}, f)



