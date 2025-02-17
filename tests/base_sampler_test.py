import numpy as np
import unittest
from UBAS.samplers.base_sampler import BaseSampler


class TestBenchmarkFunctionGenerator(unittest.TestCase):
    def test_uniform_sampling(self):
        dimension = 3
        lb = -1
        ub = 1
        batch_points = 10
        bounds = (np.ones((2, dimension)) * np.array([[lb, ub]]).T)
        sampler = BaseSampler(bounds, dimension, batch_points)
        x = sampler.uniformly_sample()
        self.assertEqual(x.shape[0], batch_points)
        self.assertEqual(x.shape[1], dimension)
        self.assertTrue(all(isinstance(val, float) for val in np.ravel(x)))
        self.assertTrue(all(val >= lb) for val in np.ravel(x))
        self.assertTrue(all(val <= ub) for val in np.ravel(x))

    def test_invalid_bounds_shape(self):
        dimension = 5
        lb = -1
        ub = 1
        bounds = (np.ones((dimension, 2)) * np.array([lb, ub]))
        batch_points = 10
        with self.assertRaises(ValueError):
            sampler = BaseSampler(bounds, dimension, batch_points)

    def test_lb_above_hb(self):
        dimension = 5
        lb = 1
        ub = -1
        bounds = (np.ones((2, dimension)) * np.array([[lb, ub]]).T)
        batch_points = 10
        with self.assertRaises(ValueError):
            sampler = BaseSampler(bounds, dimension, batch_points)

    def test_adaptively_mc_sample_no_bin_width(self):
        dimension = 3
        lb = -1
        ub = 1
        batch_points = 10
        bounds = (np.ones((2, dimension)) * np.array([[lb, ub]]).T)
        x_candidates = np.array([[2, 1, 3], [0.3, 0.2, 0.3], [0.5, 0.3, 0.5], [0.1, 0.1, 0.2]])
        p = np.array([1, 0, 0, 0])
        sampler = BaseSampler(bounds, dimension, batch_points)
        bin_width = None
        x = sampler.adaptively_mc_sample(x_candidates, p, bin_width)
        self.assertEqual(x.shape[0], batch_points)
        self.assertEqual(x.shape[1], dimension)
        self.assertTrue(all(isinstance(val, float) for val in np.ravel(x)))
        self.assertTrue(all(val >= lb) for val in np.ravel(x))
        self.assertTrue(all(val <= ub) for val in np.ravel(x))
        self.assertTrue(all(val == np.array([1, 1, 1])) for val in x)

    def test_adaptively_mc_sample_array_bin_width(self):
        dimension = 3
        lb = -1
        ub = 1
        batch_points = 10
        bounds = (np.ones((2, dimension)) * np.array([[lb, ub]]).T)
        x_candidates = np.array([[0.1, 0.2, 0.4], [0.3, 0.2, 0.3], [0.5, 0.3, 0.5], [0.1, 0.1, 0.2]])
        p = np.array([1, 0, 0, 0])
        sampler = BaseSampler(bounds, dimension, batch_points)
        bin_width = np.array([0.1, 0.2, 0.2])
        x = sampler.adaptively_mc_sample(x_candidates, p, bin_width)
        self.assertEqual(x.shape[0], batch_points)
        self.assertEqual(x.shape[1], dimension)
        self.assertTrue(all(isinstance(val, float) for val in np.ravel(x)))
        self.assertTrue(all(val >= lb) for val in np.ravel(x))
        self.assertTrue(all(val <= ub) for val in np.ravel(x))
        self.assertTrue(all(val - x_candidates[0] - bin_width >= 0) for val in x)

    def test_adaptively_mc_sample_invalid_x_candidates(self):
        dimension = 3
        lb = -1
        ub = 1
        batch_points = 10
        bounds = (np.ones((2, dimension)) * np.array([[lb, ub]]).T)
        x_candidates = np.array([[2, 2], [0.3, 0.3], [0.5, 0.5], [0.1, 0.1]])
        p = np.array([1, 0, 0, 0])
        sampler = BaseSampler(bounds, dimension, batch_points)
        bin_width = None
        with self.assertRaises(ValueError):
            x = sampler.adaptively_mc_sample(x_candidates, p, bin_width)

    def test_adaptively_mc_sample_invalid_p(self):
        dimension = 3
        lb = -1
        ub = 1
        batch_points = 10
        bounds = (np.ones((2, dimension)) * np.array([[lb, ub]]).T)
        x_candidates = np.array([[2, 1, 3], [0.3, 0.2, 0.3], [0.5, 0.3, 0.5], [0.1, 0.1, 0.2]])
        p = np.array([1, 0, 0, 0, 0])
        sampler = BaseSampler(bounds, dimension, batch_points)
        bin_width = None
        with self.assertRaises(ValueError):
            x = sampler.adaptively_mc_sample(x_candidates, p, bin_width)

    def test_adaptively_mc_sample_invalid_bin_width(self):
        dimension = 3
        lb = -1
        ub = 1
        batch_points = 10
        bounds = (np.ones((2, dimension)) * np.array([[lb, ub]]).T)
        x_candidates = np.array([[2, 1, 3], [0.3, 0.2, 0.3], [0.5, 0.3, 0.5], [0.1, 0.1, 0.2]])
        p = np.array([1, 0, 0, 0])
        sampler = BaseSampler(bounds, dimension, batch_points)
        bin_width = np.array([0.1, 0.1, 0.3, 0.5])
        with self.assertRaises(ValueError):
            x = sampler.adaptively_mc_sample(x_candidates, p, bin_width)


if __name__ == "__main__":
    unittest.main()
