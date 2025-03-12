import numpy as np
import pickle
import os
import json
from UBAS.regressors.quant_nn_regressor import QuantNNRegressor
from UBAS.generators.input_generator import InputGenerator
from UBAS.estimators.k_fold_quantile_estimator import KFoldQuantileRegressor
from sklearn.model_selection import KFold
from UBAS.utils.evaluation import evaluate_performance
from dataclasses import asdict

def test_uniform_sampling():
    functions = ["CD", "CL", "CM"]
    base_path = os.path.join("D:", os.sep, "UBAS", "projects", "nn_gp", "moment_coeff")
    for function in functions:
        dim = "2D"
        SAVE_PATH = os.path.join(base_path, function, dim)
        param_path = os.path.join("D:", os.sep, "UBAS", "projects", "param_opt", "Moment_coeff", function, dim, "hyperparams.json")
        with open(param_path, 'r') as f:
            params = json.load(f)

        reg = QuantNNRegressor(quantiles=[0.1, 0.9], batch_size=32, **params)

        sampler_path = os.path.join(SAVE_PATH, "nn", "trial_1", "sampler_data.pkl")
        with open(sampler_path, 'rb') as f:
            sampler = pickle.load(f)

        test_inputs = sampler.test_inputs
        test_targets = sampler.test_targets

        n_points = sampler.x_exact.shape[0]
        dimension = sampler.dimension

        batch_size = int(n_points / 3)
        bounds = sampler.bounds

        gen = InputGenerator(bounds, dimension)
        scaler = sampler.scaler
        unscaled_x = scaler.inverse_transform(gen.uniformly_sample(n_points))
        unscaled_x_train, y_train = sampler.generator.generate(unscaled_x)
        X_train = scaler.transform(unscaled_x_train)

        surrogate = KFoldQuantileRegressor(reg, method="plus", cv=KFold(n_splits=5, shuffle=True), alpha=0.2, n_jobs=7)

        surrogate.fit(X_train, y_train, batch_size=batch_size)

        y_preds, y_bounds = surrogate.predict(test_inputs)

        performance = evaluate_performance(y_preds, y_bounds, test_targets, include_mre=False)
        perf_dict = asdict(performance)

        with open(os.path.join(SAVE_PATH, "nn", "uniform_sampling_performance.json"), 'w') as f:
            json.dump(perf_dict, f)


if __name__ == "__main__":
    test_uniform_sampling()

