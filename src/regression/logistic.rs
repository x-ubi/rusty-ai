use std::{error::Error, marker::PhantomData};

use crate::{
    data::dataset::{Dataset, RealNumber, WholeNumber},
    metrics::confusion::ClassificationMetrics,
};
use nalgebra::{DMatrix, DVector};

/// Logistic regression model for binary classification.
///
/// This struct represents a logistic regression model for binary classification. It uses the sigmoid function to map the input features to a probability between 0 and 1, and makes predictions based on a threshold of 0.5.
///
/// # Type Parameters
///
/// * `XT`: The type of the input features.
/// * `YT`: The type of the target labels.
///
/// # Fields
///
/// * `weights`: The weights of the logistic regression model, with the first being the bias weight.
/// * `_marker`: A marker field to indicate the target label type.
///
/// # Examples
///
/// ```
/// use rusty_ai::regression::logistic::LogisticRegression;
/// use rusty_ai::data::dataset::Dataset;
/// use nalgebra::{DMatrix, DVector};
///
/// // Create a new logistic regression model
/// let mut model: LogisticRegression<f64, u8> = LogisticRegression::new();
///
/// // Fit the model to a dataset
/// let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let y = DVector::from_vec(vec![0, 1, 0]);
/// let dataset = Dataset::new(x, y);
/// let lr = 0.01;
/// let max_steps = 1000;
/// let epsilon = Some(0.001);
/// let progress = Some(100);
/// let result = model.fit(&dataset, lr, max_steps, epsilon, progress);
///
/// // Make predictions using the trained model
/// let x_pred = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
/// let predictions = model.predict(&x_pred);
/// ```

#[derive(Clone, Debug)]
pub struct LogisticRegression<XT: RealNumber, YT: WholeNumber> {
    weights: DVector<XT>,

    _marker: PhantomData<YT>,
}

impl<XT: RealNumber, YT: WholeNumber> ClassificationMetrics<YT> for LogisticRegression<XT, YT> {}

impl<XT: RealNumber, YT: WholeNumber> Default for LogisticRegression<XT, YT> {
    /// Creates a new instance of `LogisticRegression` with default values.
    ///
    /// # Returns
    ///
    /// A new `LogisticRegression` instance.
    fn default() -> Self {
        Self::new()
    }
}

impl<XT: RealNumber, YT: WholeNumber> LogisticRegression<XT, YT> {
    /// Creates a new instance of `LogisticRegression` with default values.
    ///
    /// # Returns
    ///
    /// A new `LogisticRegression` instance.
    pub fn new() -> Self {
        Self {
            weights: DVector::<XT>::from_element(3, XT::from_f64(1.0).unwrap()),
            _marker: PhantomData,
        }
    }

    /// Creates a new instance of `LogisticRegression` with custom parameters.
    ///
    /// # Parameters
    ///
    /// * `dimension`: The dimension of the input features. If `None`, it will be inferred from the starting weights.
    /// * `weights`: The starting weights for the logistic regression model. If `None`, default weights will be used.
    ///
    /// # Returns
    ///
    /// A new `LogisticRegression` instance.
    ///
    /// # Errors
    ///
    /// An error is returned if the dimension and weights are incompatible.
    pub fn with_params(
        dimension: Option<usize>,
        weights: Option<DVector<XT>>,
    ) -> Result<Self, Box<dyn Error>> {
        match (dimension, &weights) {
            (None, None) => Err("Please input the dimension or starting weights.".into()),

            (Some(dim), Some(w)) if dim != w.len() - 1 => {
                Err("The weights should be longer by 1 than the dimension to account for the bias weight.".into())
            }
            _ => Ok(Self {
                weights: weights.unwrap_or_else(|| {
                    DVector::<XT>::from_element(dimension.unwrap() + 1, XT::from_f64(1.0).unwrap())
                }),
                _marker: PhantomData,
            }),
        }
    }

    /// Predicts the target labels for the given input features.
    ///
    /// # Parameters
    ///
    /// * `x_pred`: The input features to make predictions for.
    ///
    /// # Returns
    ///
    /// A `Result` containing the predicted target labels if successful, or an error message if an error occurs during prediction.
    pub fn predict(&self, x_pred: &DMatrix<XT>) -> Result<DVector<YT>, Box<dyn Error>> {
        let x_pred_with_bias = x_pred.clone().insert_column(0, XT::from_f64(0.0).unwrap());

        Ok(self.h(&x_pred_with_bias).map(|val| {
            if val > XT::from_f64(0.5).unwrap() {
                YT::from_usize(1).unwrap()
            } else {
                YT::from_usize(0).unwrap()
            }
        }))
    }

    /// Fits the logistic regression model to a dataset.
    ///
    /// # Parameters
    ///
    /// * `dataset`: The dataset to fit the model to.
    /// * `lr`: The learning rate for gradient descent.
    /// * `max_steps`: The maximum number of steps for gradient descent.
    /// * `epsilon`: The convergence threshold for gradient descent. If `None`, a default value is used.
    /// * `progress`: The number of steps to display progress information. If `None`, no progress is displayed.
    ///
    /// # Returns
    ///
    /// A string indicating the result of the training process.
    ///
    /// # Errors
    ///
    /// An error is returned if the progress steps value is 0.
    pub fn fit(
        &mut self,
        dataset: &Dataset<XT, YT>,
        lr: XT,
        mut max_steps: usize,
        epsilon: Option<XT>,
        progress: Option<usize>,
    ) -> Result<String, Box<dyn Error>> {
        if progress.is_some_and(|steps| steps == 0) {
            return Err(
                "The number of steps for progress visualization must be greater than 0.".into(),
            );
        }
        let (x, y) = dataset.into_parts();

        let epsilon = epsilon.unwrap_or_else(|| XT::from_f64(1e-6).unwrap());
        let initial_max_steps = max_steps;
        let x_with_bias = x.clone().insert_column(0, XT::from_f64(1.0).unwrap());
        while max_steps > 0 {
            let weights_prev = self.weights.clone();

            let gradient = self.gradient(&x_with_bias, y);

            self.weights -= gradient * lr;

            if progress.is_some_and(|steps| max_steps % steps == 0) {
                println!("Step: {:?}", initial_max_steps - max_steps);
                println!("Weights: {:?}", self.weights);
                println!(
                    "Cross entropy: {:?}",
                    self.cross_entropy(&x_with_bias, y, false)
                );
            }

            let delta = self
                .weights
                .iter()
                .zip(weights_prev.iter())
                .map(|(&w, &w_prev)| (w - w_prev) * (w - w_prev))
                .fold(XT::from_f64(0.0).unwrap(), |acc, x| acc + x);

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

    pub fn weights(&self) -> &DVector<XT> {
        &self.weights
    }

    fn gradient(&self, x: &DMatrix<XT>, y: &DVector<YT>) -> DVector<XT> {
        let y_pred = self.h(x);

        let y_xt_vec = y
            .iter()
            .map(|&y_i| XT::from(y_i).unwrap())
            .collect::<Vec<_>>();

        let y_xt = DVector::from_vec(y_xt_vec);
        let errors = y_pred - y_xt;

        x.transpose() * errors / XT::from_usize(y.len()).unwrap()
    }

    pub fn cross_entropy(
        &self,
        x: &DMatrix<XT>,
        y: &DVector<YT>,
        testing: bool,
    ) -> Result<XT, Box<dyn Error>> {
        let x = match testing {
            true => x.clone().insert_column(0, XT::from_f64(0.0).unwrap()),
            false => x.clone(),
        };
        let y_pred: DVector<XT> = self.h(&x);
        let one = XT::from_f64(1.0).unwrap();

        let cross_entropy = y
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_i, &y_pred_i)| {
                let y_i_xt = XT::from(y_i).unwrap();
                -y_i_xt * (y_pred_i + XT::from_f64(f64::EPSILON).unwrap()).ln()
                    - (one - y_i_xt) * (one - y_pred_i + XT::from_f64(f64::EPSILON).unwrap()).ln()
            })
            .fold(XT::from_f64(0.0).unwrap(), |acc, x| acc + x)
            / XT::from_usize(y.len()).unwrap();

        Ok(cross_entropy)
    }

    fn h(&self, x: &DMatrix<XT>) -> DVector<XT> {
        let z = x * &self.weights;
        z.map(|val| Self::sigmoid(val))
    }

    fn sigmoid(z: XT) -> XT {
        let one = XT::from_f64(1.0).unwrap();

        match z {
            z if z < XT::from_f64(-10.0).unwrap() => XT::from_f64(0.0).unwrap(),
            z if z > XT::from_f64(10.0).unwrap() => one,
            _ => one / (one + (-z).exp()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let model = LogisticRegression::<f64, u8>::default();
        assert_eq!(model.weights().len(), 3);
        assert!(model.weights().iter().all(|&w| w == 1.0));
    }

    // Test the creation of a new LogisticRegression model
    #[test]
    fn test_with_dimension() {
        let model = LogisticRegression::<f64, u8>::with_params(Some(3), None);
        assert!(model.is_ok());
        assert_eq!(model.as_ref().unwrap().weights().len(), 4);
        assert!(model.unwrap().weights().iter().all(|&w| w == 1.0));
    }

    // Test when only starting weights are provided
    #[test]
    fn test_with_weights() {
        let weights = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let model = LogisticRegression::<f64, u8>::with_params(None, Some(weights.clone()));
        assert!(model.is_ok());
        assert_eq!(model.unwrap().weights, weights);
    }

    #[test]
    fn test_with_params_nothing_provided() {
        let model = LogisticRegression::<f64, u8>::with_params(None, None);
        assert!(model.is_err());
    }

    // Test when both dimension and starting weights are provided correctly
    #[test]
    fn test_dimension_and_weights_provided_correct() {
        let weights = DVector::from_vec(vec![0.5, -0.5, 1.0]);
        let model = LogisticRegression::<f64, u8>::with_params(Some(2), Some(weights.clone()));
        assert!(model.is_ok());
        assert_eq!(model.unwrap().weights, weights);
    }

    // Test when both dimension and starting weights are provided incorrectly
    #[test]
    fn test_dimension_and_weights_provided_incorrect() {
        let weights = DVector::from_vec(vec![0.5, -0.5]);
        let model = LogisticRegression::<f64, u8>::with_params(Some(2), Some(weights));
        assert!(model.is_err());
    }

    #[test]
    fn test_h_function() {
        let mut model = LogisticRegression::<f64, u8>::with_params(Some(2), None).unwrap();

        // Set model weights to known values
        model.weights = DVector::from_vec(vec![0.0, 0.5, -0.5]);

        // Create features for testing
        let features = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        // Expected sigmoid values for the given features and weights
        // Sigmoid(0.5*1.0 - 0.5*2.0) and Sigmoid(0.5*3.0 - 0.5*4.0)
        let expected_sigmoid_values = DVector::from_vec(vec![
            1.0 / (1.0 + f64::exp(0.5)), // Sigmoid(0.5*1 - 0.5*2 + 0.0*bias)
            1.0 / (1.0 + f64::exp(0.5)), // Sigmoid(0.5*3 - 0.5*4 + 0.0*bias)
        ]);
        let features_with_bias = features.clone().insert_column(0, 1.0);
        // Compute predictions using the 'h' function
        let predictions = model.h(&features_with_bias);

        // Check if the computed predictions are close to the expected values
        for (predicted, expected) in predictions.iter().zip(expected_sigmoid_values.iter()) {
            assert!((predicted - expected).abs() < f64::EPSILON);
        }
    }

    // Test the prediction functionality
    #[test]
    fn test_predict() {
        let model = LogisticRegression::<f64, u8>::with_params(
            None,
            Some(DVector::from_vec(vec![0.0, 0.5, -0.5])),
        )
        .unwrap();

        let features = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let predictions = model.predict(&features).unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|&p| p == 0 || p == 1));
    }

    // Add more tests for fit, weights update, gradient calculation, etc.

    // Test sigmoid function

    #[test]
    fn test_sigmoid_less_than_negative_ten() {
        let value = LogisticRegression::<f64, u8>::sigmoid(-10.1);
        assert_eq!(value, 0.0);
    }

    #[test]
    fn test_sigmoid_zero() {
        let value = LogisticRegression::<f64, u8>::sigmoid(0.0);
        assert!((value - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sigmoid_one() {
        let value = LogisticRegression::<f64, u8>::sigmoid(1.0);
        println!("{}", f64::EPSILON);
        assert!((value - 0.7310585786300049).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sigmoid_over_ten() {
        let value = LogisticRegression::<f64, u8>::sigmoid(10.1);
        assert_eq!(value, 1.0);
    }

    #[test]
    fn test_h() {
        let model = LogisticRegression::<f64, u8>::with_params(
            None,
            Some(DVector::from_vec(vec![0.0, 0.5, -0.5])),
        )
        .unwrap();
        let features = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 5.0]);
        let features_with_bias = features.clone().insert_column(0, 1.0);
        let value = model.h(&features_with_bias);

        assert!((value[0] - 0.3775406687981454).abs() < f64::EPSILON);
        assert!((value[1] - 0.2689414213699951).abs() < f64::EPSILON);
    }

    // Test cross-entropy calculation
    #[test]
    fn test_cross_entropy() {
        let model = LogisticRegression::<f64, u8>::with_params(
            None,
            Some(DVector::from_vec(vec![0.0, 0.5, -0.5])),
        )
        .unwrap();

        // Create features and labels for testing
        let features = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let labels = DVector::from_vec(vec![1, 0]);

        // Compute cross-entropy loss
        let loss = model.cross_entropy(&features, &labels, true).unwrap();
        // Expected loss value
        let expected_loss = 0.7240769841801062;

        // Check if the computed loss is close to the expected value
        assert!((loss - expected_loss).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gradient() {
        // Create a logistic regression model
        let model = LogisticRegression::new();

        // Create a test input matrix and labels
        let x = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = DVector::from_vec(vec![0, 1]);

        // Calculate the gradient
        let gradient = model.gradient(&x, &y);
        // Assert the expected gradient shape
        assert_eq!(gradient.shape(), (3, 1));
    }

    #[test]
    fn test_fit_with_progress_set_to_zero() {
        let mut model = LogisticRegression::<f64, u8>::new();

        // Create a dummy dataset
        let x = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let y = DVector::from_vec(vec![1, 2]);
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
    fn test_fit() {
        let mut logistic_regression = LogisticRegression::<f64, u8>::new();
        let dataset = Dataset::new(
            DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            DVector::from_vec(vec![0, 1]),
        );
        let result = logistic_regression.fit(&dataset, 0.1, 100, Some(1e-6), Some(50));
        assert!(result.is_ok());
    }
}
