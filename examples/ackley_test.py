"""
ackley_test.py
==================
This test script constructs a surrogate model for the Ackley01 function using
uniform and adaptive sampling.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from UBAS.samplers.uniform_sampler import UniformSampler
from examples.generators.benchmark_function_generator import BenchmarkFunctionGenerator
from UBAS.regressors.k_fold_quantile_regressor import KFoldQuantileRegressor


n_sampling_iterations = 100
n_batch_points = 100
n_initial_points = 10
dimension = 2

# Allocate input array
u_x = np.zeros((n_sampling_iterations * n_batch_points + n_initial_points, dimension))

# Initialize generators
ackley_gen = BenchmarkFunctionGenerator("Ackley01")

# Initialize samplers
u_sampler = UniformSampler()

# Initialize surrogates
#surrogate = KFoldQuantileRegressor()

# construct bounds
bounds = (np.ones((2, dimension)) * np.array([-35, 35])).T

# Generate initial points
u_x[:n_initial_points] = u_sampler.sample(bounds, n_initial_points)

# Run sampling loop
for i in range(n_sampling_iterations):
    start_index = n_initial_points + i*n_batch_points
    u_x[start_index:start_index+n_batch_points] = u_sampler.sample(bounds, n_batch_points)

# Generate y values:
u_x, u_y = ackley_gen.generate(u_x)

# Plot results on a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter3D(u_x[:, 0], u_x[:, 1], u_y, c=u_y, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Ackley01 Function')

plt.show()



