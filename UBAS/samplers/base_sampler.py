"""
base_sampler.py
==================
Parent class for all samplers: functions which sample from data generators, typically for the purpose of constructing
a surrogate model
"""
from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from rich.progress import track
from UBAS.generators.input_generator import InputGenerator
from UBAS.utils.evaluation import evaluate_performance


class BaseSampler:
    """Base class for data samplers that samples uniformly within the given bounds"""
    def __init__(self, dimension, surrogate, generator, bounds, n_iterations, n_batch_points,
                 initial_inputs, initial_targets, test_inputs=None, test_targets=None, intermediate_training=False,
                 plotting_interval=5, save_interval=5, mean_relative_error=False):
        """
        Sampler Initialization

        Parameters
        ----------
        dimension : int
            Dimension of the problem
        surrogate : Object
            A surrogate model which predicts confidence intervals with the scikit-learn 'fit' and 'predict' API
        generator : BaseGenerator
            A generator model that generates target values from the inputs
        bounds : NDArray
            The bounds within which to sample. Should be an NDArray of shape (2, n_dimensions) where
            the first element of the 0th axis are the lower bounds and the second element are the upper
            bounds for each dimension.
        n_iterations : int
            Number of sampling iterations to perform
        n_batch_points : int
            Number of points to add to the training set per sampling iteration
        initial_inputs : NDArray
            Starting inputs, should be of shape (n_initial_samples, dimension)
        initial_targets : NDArray
            Initial targets, should be of shape (n_initial_samples, )
        test_inputs : NDArray default=None
            Inputs of a validation set to test the surrogate model against. If not provided, no evaluation will occur
            Should be of shape (n_test_samples, dimension)
        test_targets : NDArray default=None
            Targets of a validation set to test the surrogate model against. If not provided, no evaluation will occur
            Should be of shape (n_test_samples, )
        intermediate_training : Bool default=False
            If True, the surrogate model will be retrained after every sampling iteration. If False, the surrogate
            model gets trained once at the end of sampling
        plotting_interval : int default=5
            Not yet implemented
        save_interval : int default=5
            Not yet implemented
        mean_relative_error : Bool, default=False
            If True, mean relative error metrics will be computed (as long as test_inputs and test_targets are
            provided). If false, the mean relative error will be none in the error evaluation objects.
        """
        self.dimension = dimension
        self.surrogate = surrogate
        self.generator = generator
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.n_batch_points = n_batch_points
        self.initial_inputs = initial_inputs
        self.initial_targets = initial_targets
        self.n_initial_points = self.initial_inputs.shape[0]
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.intermediate_training = intermediate_training
        self.plotting_interval = plotting_interval
        self.save_interval = save_interval
        self.mean_relative_error = mean_relative_error

        # Initialize sampler
        self.sampler = InputGenerator(bounds, dimension)

        # Allocate arrays
        x_exact = np.zeros((self.n_iterations * self.n_batch_points + self.n_initial_points, self.dimension))
        y_exact = np.zeros(x_exact.shape[0])

        if intermediate_training is True:
            self.model_performance = np.empty(self.n_iterations + 1, dtype=object)
        else:
            self.model_performance = np.empty(1, dtype=object)

        # Populate arrays with initial values
        x_exact[:self.n_initial_points] = self.initial_inputs
        y_exact[:self.n_initial_points] = self.initial_targets

        # Create instance variables with the exact arrays:
        self.x_exact = x_exact
        self.y_exact = y_exact

        # Perform checks on test_inputs and test_targets and set evaluation flag accordingly
        if test_inputs is not None:
            self.evaluate_surrogates = True
        else:
            self.evaluate_surrogates = False

    def sample(self, filename, *args, **fit_kwargs):
        """
        Method to perform iterations of sample_step while handling re-training and evaluation the surrogate. Usually
        does not need to be overriden in subclasses.

        Parameters
        ----------
        filename : str
            Not yet Implemented
        *args
            Extra arguments to the surrogate model fit method should be passed as keyword arguments
        **fit_kwargs
            Extra keyword arguments are passed to the fit method of the surrogate model.
        """
        n_iterations = self.n_iterations
        n_initial_points = self.n_initial_points
        n_batch_points = self.n_batch_points

        # Run sampling loop:
        for i in track(range(n_iterations), description="Running Main Sampling Loop"):
            start_index = n_initial_points + i * n_batch_points

            # Re-train surrogate
            if self.intermediate_training is True:
                self.surrogate.fit(self.x_exact[:start_index], self.y_exact[:start_index], **fit_kwargs)

                # if test data is available, evaluate the surrogate
                if self.evaluate_surrogates is True:
                    y_preds, y_bounds = self.predict(self.test_inputs)
                    eval_obj = evaluate_performance(y_preds, y_bounds, self.test_targets, self.mean_relative_error)
                    self.model_performance[i] = eval_obj

            # Sample new points with sampling_step
            new_x = self.sampling_step(n_batch_points)
            new_x, new_y = self.generator.generate(new_x)
            self.x_exact[start_index:start_index+n_batch_points] = new_x
            self.y_exact[start_index:start_index+n_batch_points] = new_y

        # Re-train surrogate on full training data
        self.surrogate.fit(self.x_exact, self.y_exact, **fit_kwargs)

        # Re-evaluate_surrogate on full training data
        if self.evaluate_surrogates is True:
            y_preds, y_bounds = self.predict(self.test_inputs)
            eval_obj = evaluate_performance(y_preds, y_bounds, self.test_targets, self.mean_relative_error)
            if self.intermediate_training is True:
                self.model_performance[n_iterations] = eval_obj
            else:
                self.model_performance[0] = eval_obj

        print("Sampling has finished")

    def sampling_step(self, n_batch_points) -> NDArray:
        """
        One step of uniform sampling. This is usually overriden by subclasses

        Parameters
        ----------
        n_batch_points
            The number of points to sample with one call to sampling_step

        Returns
        -------
        NDArray
            A set of new training points of shape (n_batch_points, dimension)
        """
        new_x = self.sampler.uniformly_sample(n_batch_points)
        return new_x

    def predict(self, inputs):
        """
        A helper method to transform different types of predictions arising from surrogate models into a usable form.

        Parameters
        ----------
        inputs : NDArray
            Inputs for which to predict values

        Returns
        -------
        Y_preds : NDArray
            Array of shape (length of inputs, )
            The mean predictions of the surrogate model

        Y_bounds : NDArray
            Array of shape (length of inputs, 2)
            The lower and upper bound predictions of the surrogate model.
        """
        pred_results = self.surrogate.predict(inputs)
        if isinstance(pred_results, Tuple):
            y_preds, y_bounds = Tuple
        else:
            y_bounds = pred_results
            y_preds = np.mean(pred_results, axis=1)
        return y_preds, y_bounds
