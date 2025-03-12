"""
hyperparam_optimization.py
==========================
A script to determine the optimal hyperparameters for neural networks being trained on several different function
generators.
"""

from sklearn.model_selection import RandomizedSearchCV
from UBAS.regressors.quant_nn_regressor import QuantNNRegressor
from UBAS.generators.neural_foil_generator import NFGenerator
from UBAS.generators.input_generator import InputGenerator
from rich.progress import track
import os
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler

param_dist = {"n_epochs": np.linspace(600, 3000, 10, dtype=int),
              "layers": np.linspace(2, 8, 4, dtype=int),
              "neurons": np.linspace(32, 256, 5, dtype=int)}

model = QuantNNRegressor(quantiles=[0.1, 0.9], layers=4, neurons=128, activation="relu", n_epochs=15, batch_size=250)

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=30, cv=3, n_jobs=7, verbose=3)

#qois = ["CM", "CL", "CD"]
qois = ["CL"]
airfoil = np.array([0.02, 0.4, 0.12])

Re_bounds = np.array([10 ** 6, 10 ** 8])
alpha_bounds = np.array([0, 15])
e_bounds = np.array([0, 0.08])
p_bounds = np.array([0.2, 0.6])
t_bounds = np.array([0.02, 0.15])

b_2d = np.c_[Re_bounds, alpha_bounds]
b_5d = np.c_[e_bounds, p_bounds, t_bounds, Re_bounds, alpha_bounds]

scaler_2d = MinMaxScaler()
scaler_5d = MinMaxScaler()
b_2d = scaler_2d.fit_transform(b_2d)
b_5d = scaler_5d.fit_transform(b_5d)

n_points = 750
sampler_2d = InputGenerator(b_2d, 2)
sampler_5d = InputGenerator(b_5d, 5)

for i, qoi in enumerate(track(qois, description="Function Progress")):
    # 5D
    #generator = NFGenerator(qoi, airfoil=None)
    #SAVE_PATH = os.path.join("D:", os.sep, "UBAS", "projects", "param_opt", "Moment_coeff", qoi, "5D")
    #os.makedirs(SAVE_PATH, exist_ok=True)
    #path = os.path.join(SAVE_PATH, "hyperparams.json")

    #inputs, target = generator.generate(sampler_5d.uniformly_sample(n_points))
    #random_search.fit(inputs, target)
    #print (f"{random_search.best_score_}")
    #with open(path, 'w') as f:
    #    json.dump({k: v.tolist() for k, v in random_search.best_params_.items()}, f)

    # 2D
    generator = NFGenerator(qoi, airfoil=airfoil)
    SAVE_PATH = os.path.join("D:", os.sep, "UBAS", "projects", "param_opt", "Moment_coeff", qoi, "2D")
    os.makedirs(SAVE_PATH, exist_ok=True)
    path = os.path.join(SAVE_PATH, "hyperparams.json")

    inputs, target = generator.generate(scaler_2d.inverse_transform(sampler_2d.uniformly_sample(n_points)))
    inputs = scaler_2d.transform(inputs)
    print (inputs)

    random_search.fit(inputs, target)
    print(f"{random_search.best_score_}")
    with open(path, 'w') as f:
        json.dump({k: v.tolist() for k, v in random_search.best_params_.items()}, f)

