import numpy as np
import unittest
from UBAS.samplers.uniform_sampler import UniformSampler


class TestBenchmarkFunctionGenerator(unittest.TestCase):
    def test_uniform_sampling(self):
        u_sampler = UniformSampler()
        dimension = 2
        lb = -1
        ub = 1
        batch_points = 10
        bounds = (np.ones((2, dimension)) * np.array([lb, ub])).T
        x = u_sampler.sample(bounds, batch_points)
        self.assertEqual(x.shape[0], batch_points)
        self.assertEqual(x.shape[1], dimension)
        self.assertTrue(all(isinstance(val, float) for val in np.ravel(x)))
        self.assertTrue(all(val >= lb) for val in np.ravel(x))
        self.assertTrue(all(val <= ub) for val in np.ravel(x))


if __name__ == "__main__":
    unittest.main()
