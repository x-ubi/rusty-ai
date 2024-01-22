use crate::{
    data::dataset::{Dataset, RealNumber},
    metrics::errors::RegressionMetrics,
};
use nalgebra::{DMatrix, DVector};
use std::error::Error;

/// Represents a linear regression model.
///
/// The `LinearRegression` struct implements a linear regression model for predicting a target variable based on one or more input features.
/// It uses the least squares method to estimate the weights of the linear model.
///
/// # Type Parameters
///
/// * `T`: The numeric type used for calculations. Must implement the `RealNumber` trait.
///
/// # Fields
///
/// * `weights`: The weights of the logistic regression model, with the first being the bias weight.
///
/// # Examples
///
/// ```
/// use rusty_ai::regression::linear::LinearRegression;
/// use rusty_ai::data::dataset::Dataset;
/// use nalgebra::{DMatrix, DVector};
///
/// // Create a new linear regression model
/// let mut model = LinearRegression::<f64>::new();
///
/// // Fit the model to a dataset
/// let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let y = DVector::from_vec(vec![1.5, 2.5, 3.5]);
/// let dataset = Dataset::new(x, y);
/// let learning_rate = 0.01;
/// let max_steps = 1000;
/// let epsilon = Some(0.001);
/// let progress = Some(100);
/// let result = model.fit(&dataset, learning_rate, max_steps, epsilon, progress);
///
/// // Make predictions using the trained model
/// let x_test = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
/// let predictions = model.predict(&x_test);
/// assert!(predictions.is_ok());
/// ```

#[derive(Clone, Debug)]
pub struct LinearRegression<T: RealNumber> {
    weights: DVector<T>,
}

impl<T: RealNumber> RegressionMetrics<T> for LinearRegression<T> {}

impl<T: RealNumber> Default for LinearRegression<T> {
    /// Creates a new `LinearRegression` model with default weights.
    ///
    /// The default weights are initialized to 1.0 for each feature, including the bias weight.
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealNumber> LinearRegression<T> {
    /// Creates a new `LinearRegression` model with default weights.
    ///
    /// The default weights are initialized to 1.0 for each feature, including the bias weight.
    pub fn new() -> Self {
        Self {
            weights: DVector::<T>::from_element(3, T::from_f64(1.0).unwrap()),
        }
    }

    /// Creates a new `LinearRegression` model with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `dimension`: The dimension of the input features. If `None`, the dimension will be inferred from the provided weights.
    /// * `weights`: The initial weights for the linear regression model. If `None`, default weights will be used.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `LinearRegression` model if the parameters are valid, or an error message if the parameters are invalid.
    ///
    /// # Errors
    ///
    /// An error will be returned if:
    /// * Both `dimension` and `weights` are `None`.
    /// * The length of `weights` is not equal to `dimension + 1` to account for the bias weight.
    pub fn with_params(
        dimension: Option<usize>,
        weights: Option<DVector<T>>,
    ) -> Result<Self, Box<dyn Error>> {
        match (dimension, &weights) {
            (None, None) => Err("Please input the dimension or starting weights.".into()),

            (Some(dim), Some(w)) if dim != w.len() - 1 => {
                Err("The weights should be longer by 1 than the dimension to account for the bias weight.".into())
            }
            _ => Ok(Self {
                weights: weights.unwrap_or_else(|| {
                    DVector::<T>::from_element(dimension.unwrap() + 1, T::from_f64(1.0).unwrap())
                }),
            }),
        }
    }

    /// Returns a reference to the weights of the linear regression model.
    pub fn weights(&self) -> &DVector<T> {
        &self.weights
    }

    /// Makes predictions using the trained linear regression model.
    ///
    /// # Arguments
    ///
    /// * `x_pred`: The input features for which to make predictions.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the predicted target values if successful, or an error message if an error occurs during prediction.
    pub fn predict(&self, x_pred: &DMatrix<T>) -> Result<DVector<T>, Box<dyn Error>> {
        let x_pred_with_bias = x_pred.clone().insert_column(0, T::from_f64(1.0).unwrap());
        Ok(self.h(&x_pred_with_bias))
    }

    /// Fits the linear regression model to a dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset`: The dataset containing the input features and target values.
    /// * `lr`: The learning rate for gradient descent.
    /// * `max_steps`: The maximum number of steps to perform during training.
    /// * `epsilon`: The convergence threshold. If the change in weights is below this threshold, training will stop.
    /// * `progress`: The number of steps at which to display progress information. If `None`, no progress information will be displayed.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a success message if training is successful, or an error message if an error occurs during training.
    ///
    /// # Errors
    ///
    /// An error will be returned if:
    /// * The number of steps for progress visualization is 0.
    /// * The gradient turns to NaN during training.
    pub fn fit(
        &mut self,
        dataset: &Dataset<T, T>,
        lr: T,
        mut max_steps: usize,
        epsilon: Option<T>,
        progress: Option<usize>,
    ) -> Result<String, Box<dyn Error>> {
        if progress.is_some_and(|steps| steps == 0) {
            return Err(
                "The number of steps for progress visualization must be greater than 0.".into(),
            );
        }

        let (x, y) = dataset.into_parts();

        let epsilon = epsilon.unwrap_or_else(|| T::from_f64(1e-6).unwrap());
        let initial_max_steps = max_steps;
        let x_with_bias = x.clone().insert_column(0, T::from_f64(1.0).unwrap());
        while max_steps > 0 {
            let weights_prev = self.weights.clone();

            let gradient = self.gradient(&x_with_bias, y);

            if gradient.iter().any(|&g| g.is_nan()) {
                return Err("Gradient turned to NaN during training.".into());
            }

            self.weights -= gradient * lr;

            if progress.is_some_and(|steps| max_steps % steps == 0) {
                println!("Step: {}", initial_max_steps - max_steps);
                println!("Weights: {:?}", self.weights);
                println!("MSE: {:?}", self.mse_training(&x_with_bias, y));
            }

            let delta = self
                .weights
                .iter()
                .zip(weights_prev.iter())
                .map(|(&w, &w_prev)| (w - w_prev) * (w - w_prev))
                .fold(T::from_f64(0.0).unwrap(), |acc, x| acc + x);

            if delta < epsilon {
                return Ok(format!(
                    "Finished training in {} steps.",
                    initial_max_steps - max_steps,
                ));
            }
            max_steps -= 1;
        }
        Ok("Reached maximum steps without converging.".into())
    }

    fn gradient(&self, x: &DMatrix<T>, y: &DVector<T>) -> DVector<T> {
        let y_pred = self.h(x);

        let errors = y_pred - y;

        x.transpose() * errors * T::from_f64(2.0).unwrap() / T::from_usize(y.len()).unwrap()
    }

    fn h(&self, x: &DMatrix<T>) -> DVector<T> {
        x * &self.weights
    }

    fn mse_training(&self, x: &DMatrix<T>, y: &DVector<T>) -> T {
        let m = T::from_usize(y.len()).unwrap();
        let y_pred = self.h(x);

        let errors = y_pred - y;

        let errors_sq = errors.component_mul(&errors);
        errors_sq.sum() / m
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_new() {
        let model = LinearRegression::<f32>::new();
        assert_eq!(model.weights().len(), 3);
        assert!(model.weights().iter().all(|&w| w == 1.0));
    }

    #[test]
    fn test_with_params() {
        // Test with valid dimensions and weights
        let weights = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let model = LinearRegression::with_params(Some(2), Some(weights.clone()));
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.weights, weights);
    }

    #[test]
    fn test_with_params_incorrect() {
        let weights = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let model = LinearRegression::with_params(Some(4), Some(weights));
        assert!(model.is_err());
    }

    #[test]
    fn test_with_dimension() {
        let model = LinearRegression::<f64>::with_params(Some(3), None);
        assert!(model.is_ok());
        assert_eq!(model.as_ref().unwrap().weights().len(), 4);
        assert!(model.unwrap().weights().iter().all(|&w| w == 1.0));
    }

    #[test]
    fn test_with_weights() {
        let weights = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let model = LinearRegression::with_params(None, Some(weights.clone()));
        assert!(model.is_ok());
        assert_eq!(model.unwrap().weights, weights);
    }

    #[test]
    fn test_with_nothing_provided() {
        // Test with no dimensions and no weights
        let model = LinearRegression::<f64>::with_params(None, None);
        assert!(model.is_err());
    }

    #[test]
    fn test_weights() {
        // Create a LinearRegression model with known weights
        let weights = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let model = LinearRegression::with_params(Some(2), Some(weights.clone())).unwrap();
        let model_weights = model.weights();
        assert_eq!(model_weights, &weights);
    }

    #[test]
    fn test_predict() {
        let weights = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let model = LinearRegression::with_params(None, Some(weights)).unwrap();
        let x_pred = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let prediction = model.predict(&x_pred);
        assert!(prediction.is_ok());

        let expected = DVector::from_vec(vec![9.0, 19.0]);
        assert_eq!(prediction.unwrap(), expected);
    }

    #[test]
    fn test_gradient() {
        // Create a LinearRegression instance
        let model =
            LinearRegression::<f64>::with_params(None, Some(DVector::from(vec![1.0, 2.0, 3.0])))
                .unwrap();

        // Create input matrix and target vector
        let x = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let y = DVector::from_vec(vec![7.0, 8.0]);
        let x_with_bias = x.clone().insert_column(0, 1.0);

        // Calculate the gradient
        let gradient = model.gradient(&x_with_bias, &y);

        // Define the expected gradient
        let expected_gradient = DVector::from_vec(vec![13.0, 35.0, 48.0]);

        // Check if the calculated gradient matches the expected gradient
        assert_eq!(gradient, expected_gradient);
    }

    #[test]
    fn test_mse_training() {
        let model =
            LinearRegression::<f64>::with_params(None, Some(DVector::from(vec![1.0, 2.0, 3.0])))
                .unwrap();
        let x = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let y = DVector::from_vec(vec![7.0, 8.0]);

        let x_with_bias = x.clone().insert_column(0, 1.0);

        let mse = model.mse_training(&x_with_bias, &y);

        assert_relative_eq!(mse, 62.5, epsilon = 1e-6);
    }

    #[test]
    fn test_fit_with_progress_set_to_zero() {
        let mut model = LinearRegression::<f64>::new();

        // Create a dummy dataset
        let x = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let y = DVector::from_vec(vec![1.0, 2.0]);
        let dataset = Dataset::new(x, y);

        let lr = 0.1;
        let max_steps = 100;
        let epsilon = Some(0.0001);
        let progress = Some(0);

        let result = model.fit(&dataset, lr, max_steps, epsilon, progress);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "The number of steps for progress visualization must be greater than 0."
        );
    }

    #[test]
    fn test_fit_no_convergence() {
        let mut logistic_regression = LinearRegression::<f64>::new();
        let dataset = Dataset::new(
            DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        );
        let result = logistic_regression.fit(&dataset, 0.1, 100, Some(1e-6), None);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            "Reached maximum steps without converging.".to_string()
        );
    }

    #[test]
    fn test_fit_with_convergence() {
        let mut logistic_regression = LinearRegression::<f64>::new();
        let dataset = Dataset::new(
            DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        );
        let result = logistic_regression.fit(&dataset, 0.01, 100, Some(1e-2), Some(1));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Finished training in 4 steps.".to_string());
    }
}
