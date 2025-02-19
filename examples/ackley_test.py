"""
ackley_test.py
==================
This test script constructs a surrogate model for the Ackley01 function using
uniform and adaptive sampling.
"""
import numpy as np
import matplotlib.pyplot as plt
from UBAS.generators.input_generator import InputGenerator
from UBAS.generators.benchmark_function_generator import BenchmarkFunctionGenerator
from UBAS.regressors.quant_nn import QuantNN
from mpl_toolkits.axes_grid1 import make_axes_locatable
from UBAS.utils.evaluation import evaluate_performance
# from UBAS.estimators.k_fold_quantile_regressor import KFoldQuantileRegressor
from copy import deepcopy
from rich.progress import track

n_sampling_iterations = 10
n_batch_points = 30
n_initial_points = 100
dimension = 2
alpha = 0.2
p_samples = 100000

# Allocate input array
u_x = np.zeros((n_sampling_iterations * n_batch_points + n_initial_points, dimension))
a_x = np.zeros_like(u_x)
u_y = np.zeros(u_x.shape[0])
a_y = np.zeros_like(u_y)


# Initialize generators
ackley_gen = BenchmarkFunctionGenerator("Ackley01")

# construct bounds
bounds = (np.ones((2, dimension)) * np.array([[-35, 35]]).T)

# Initialize samplers
sampler = InputGenerator(bounds, dimension)


# Initialize surrogates
u_nn = QuantNN(input_dim=dimension, hidden_dim=64, output_dim=2, num_layers=3,
               quantiles=np.array([alpha, 1-alpha]), max_epochs=500, batch_size=500)
a_nn = deepcopy(u_nn)

# Generate initial points
u_x[:n_initial_points], u_y[:n_initial_points] = ackley_gen.generate(sampler.uniformly_sample(n_initial_points))
a_x[:n_initial_points], a_y[:n_initial_points] = ackley_gen.generate(sampler.uniformly_sample(n_initial_points))

# Run sampling loop
for i in track(range(n_sampling_iterations)):
    start_index = n_initial_points + i*n_batch_points

    # Uniformly sample points
    new_u_x = sampler.uniformly_sample(n_batch_points)
    new_u_x, new_u_y = ackley_gen.generate(new_u_x)

    u_x[start_index:start_index+n_batch_points] = new_u_x
    u_y[start_index:start_index+n_batch_points] = new_u_y

    # re-train adaptive neural net
    a_nn.fit(a_x[:start_index], a_y[:start_index])

    # re-generate points with which to sample the width of the adaptive neural points
    probability_eval_points = sampler.uniformly_sample(p_samples)
    p_preds = a_nn.predict(probability_eval_points)

    # Construct probability distribution and use adaptive sampler
    widths = p_preds[:, 1]-p_preds[:, 0]
    p = np.where(widths < 0, 0, widths)
    p = p / np.sum(p)

    new_a_x = sampler.adaptively_mc_sample(probability_eval_points, p, n_batch_points)
    new_a_x, new_a_y = ackley_gen.generate(new_a_x)

    # Update array:
    a_x[start_index:start_index + n_batch_points] = new_a_x
    a_y[start_index:start_index + n_batch_points] = new_a_y

# Fit uniform and adaptive surrogates
u_nn.fit(u_x, u_y)
a_nn.fit(a_x, a_y)

print("Saving Neural Networks")
u_nn.save("ackley_test/u_nn")
a_nn.save("ackley_test/a_nn")

# Plot results on a scatter plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(32, 8))
x1 = np.linspace(bounds[0, 0], bounds[1, 0], 50)
x2 = np.linspace(bounds[0, 1], bounds[1, 1], 50)
X1, X2 = np.meshgrid(x1, x2)
raveled_inputs = np.c_[np.ravel(X1), np.ravel(X2)]
exact_Z = ackley_gen.generate(raveled_inputs)[1].reshape(X1.shape[0], X2.shape[0])

u_preds = u_nn.predict(raveled_inputs)
a_preds = a_nn.predict(raveled_inputs)

u_Z = np.reshape(np.mean(u_preds, axis=1), (X1.shape[0], X2.shape[0]))
a_Z = np.reshape(np.mean(a_preds, axis=1), (X1.shape[0], X2.shape[0]))
a_width = np.reshape(a_preds[:, 1] - a_preds[:, 0], (X1.shape[0], X2.shape[0]))

u_performance = evaluate_performance(np.ravel(u_Z), u_preds, np.ravel(exact_Z))
a_performance = evaluate_performance(np.ravel(a_Z), a_preds, np.ravel(exact_Z))

print (f"{u_performance.mse = }, {a_performance.mse = }")

ax1.set_title("Exact Solution")
clev = np.arange(exact_Z.min(), exact_Z.max(), 0.001)
cm1 = ax1.contourf(X1, X2, exact_Z, clev)
ax1.set_aspect("equal")

ax2.set_title("Uniformly Sampled Solution")
cm2 = ax2.contourf(X1, X2, u_Z, clev)
ax2.set_aspect("equal")

ax3.set_title("Adaptively Sampled Solution")
cm3 = ax3.contourf(X1, X2, a_Z, clev)
ax3.set_aspect("equal")
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size = "5%", pad=0.05)
fig.colorbar(cm3, cax=cax)

ax4.set_title("Adaptive Sampling Widths")
cm4 = ax4.contourf(X1, X2, a_width)
ax4.set_aspect("equal")
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size = "5%", pad=0.05)
fig.colorbar(cm4, cax=cax)

ax4.scatter(a_x[:, 0], a_x[:, 1], color='black')

plt.show()



