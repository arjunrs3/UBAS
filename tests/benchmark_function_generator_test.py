import numpy as np
import unittest
from examples.generators.benchmark_function_generator import BenchmarkFunctionGenerator


class TestBenchmarkFunctionGenerator(unittest.TestCase):
    def test_valid_function(self):
        x_data = np.array([[0, 0], [1, 1]])
        generator = BenchmarkFunctionGenerator("Ackley03")
        x, results = list(generator.generate(x_data))
        self.assertEqual(x_data.all(), x.all())
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(y, float) for y in results))

    def test_invalid_function(self):
        with self.assertRaises(ValueError):
            BenchmarkFunctionGenerator("invalid_function")


if __name__ == "__main__":
    unittest.main()
