import unittest
import numpy as np
from UBAS.utils.evaluation import evaluate_performance, EvalObject


class TestEvaluatePerformance(unittest.TestCase):
    def setUp(self):
        # Sample test data
        self.y_preds = np.array([2.0, 4.0, 6.0, 8.0])
        self.y_bounds = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [5.6, 9.0]])
        self.y_true = np.array([2.1, 3.2, 6.2, 8.0])

    def test_evaluate_performance_with_y_true(self):
        result = evaluate_performance(self.y_preds, self.y_bounds, self.y_true)

        self.assertIsInstance(result, EvalObject)
        self.assertIsNotNone(result.mse)
        self.assertIsNotNone(result.max_absolute_error)
        self.assertIsNotNone(result.mean_width)
        self.assertIsNotNone(result.max_width)
        self.assertIsNotNone(result.coverage)
        self.assertAlmostEqual(result.mse, 0.1725)
        self.assertAlmostEqual(result.max_width, 3.4)
        self.assertAlmostEqual(result.max_absolute_error, 0.8)
        self.assertAlmostEqual(result.coverage, 0.75)
        self.assertAlmostEqual(result.mean_width, 1.6)

    def test_evaluate_performance_without_y_true(self):
        result = evaluate_performance(self.y_preds, self.y_bounds)

        self.assertIsInstance(result, EvalObject)
        self.assertIsNone(result.mse)
        self.assertIsNone(result.max_absolute_error)
        self.assertIsNotNone(result.mean_width)
        self.assertIsNotNone(result.max_width)
        self.assertIsNone(result.coverage)

    def test_invalid_y_bounds_shape(self):
        y_bounds_invalid = np.array([[1.5, 3.5, 5.5]])  # Invalid shape
        with self.assertRaises(ValueError):
            evaluate_performance(self.y_preds, y_bounds_invalid, self.y_true)

    def test_invalid_y_true_shape(self):
        y_true_invalid = np.array([2.1, 3.9])  # Mismatched shape
        with self.assertRaises(ValueError):
            evaluate_performance(self.y_preds, self.y_bounds, y_true_invalid)


if __name__ == "__main__":
    unittest.main()
