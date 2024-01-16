use std::{error::Error, marker::PhantomData};

use crate::dataset::{Dataset, RealNumber, WholeNumber};
use nalgebra::{DMatrix, DVector};

pub struct LogisticRegression<XT: RealNumber, YT: WholeNumber> {
    weights: DVector<XT>,

    _marker: PhantomData<YT>,
}

impl<XT: RealNumber, YT: WholeNumber> LogisticRegression<XT, YT> {
    pub fn new(
        dimension: Option<usize>,
        weights: Option<DVector<XT>>,
    ) -> Result<Self, Box<dyn Error>> {
        match (dimension, &weights) {
            (None, None) => Err("Please input the dimension or starting weights.".into()),

            (Some(dim), Some(w)) if dim != w.len() - 1 => {
                Err("The dimension isn't equal the amount of weights.".into())
            }
            _ => Ok(Self {
                weights: weights.unwrap_or_else(|| {
                    DVector::<XT>::from_element(dimension.unwrap() + 1, XT::from_f64(0.0).unwrap())
                }),
                _marker: PhantomData,
            }),
        }
    }

    pub fn predict(&self, x_pred: &DMatrix<XT>) -> DVector<YT> {
        let x_pred_with_bias = x_pred.clone().insert_column(0, XT::from_f64(0.0).unwrap());

        self.h(&x_pred_with_bias).map(|val| {
            if val > XT::from_f64(0.5).unwrap() {
                YT::from_usize(1).unwrap()
            } else {
                YT::from_usize(0).unwrap()
            }
        })
    }

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
        let initial_max_steps = max_steps.clone();
        let x_with_bias = x.clone().insert_column(0, XT::from_f64(1.0).unwrap());
        while max_steps > 0 {
            let weights_prev = self.weights.clone();

            let gradient = self.gradient(&x_with_bias, y);

            self.weights -= gradient * lr;

            if progress.is_some_and(|steps| max_steps % steps == 0) {
                println!("Step: {:?}", initial_max_steps - max_steps);
                println!("Weights: {:?}", self.weights);
                println!("Cross entropy: {:?}", self.cross_entropy(&x_with_bias, y));
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

    pub fn cross_entropy(&self, x: &DMatrix<XT>, y: &DVector<YT>) -> Result<XT, Box<dyn Error>> {
        let y_pred: DVector<XT> = self.h(x);
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

    // Test the creation of a new LogisticRegression model
    #[test]
    fn test_new_logistic_regression() {
        let model = LogisticRegression::<f64, u8>::new(Some(3), None);
        assert!(model.is_ok());
    }

    #[test]
    fn test_new_logistic_regression_nothing_provided() {
        let model = LogisticRegression::<f64, u8>::new(None, None);
        assert!(model.is_err());
    }

    // Test when only dimension is provided
    #[test]
    fn test_new_logistic_regression_only_dimension_provided() {
        let model = LogisticRegression::<f64, u8>::new(Some(3), None);
        assert!(model.is_ok());
        assert_eq!(model.unwrap().weights.len(), 4);
    }

    // Test when only starting weights are provided
    #[test]
    fn test_new_logistic_regression_only_weights_provided() {
        let weights = DVector::from_vec(vec![0.5, -0.5, 0.2]);
        let model = LogisticRegression::<f64, u8>::new(None, Some(weights.clone()));
        assert!(model.is_ok());
        assert_eq!(model.unwrap().weights, weights);
    }

    // Test when both dimension and starting weights are provided correctly
    #[test]
    fn test_new_logistic_regression_dimension_and_weights_provided_correct() {
        let weights = DVector::from_vec(vec![0.5, -0.5, 1.0]);
        let model = LogisticRegression::<f64, u8>::new(Some(2), Some(weights.clone()));
        assert!(model.is_ok());
        assert_eq!(model.unwrap().weights, weights);
    }

    // Test when both dimension and starting weights are provided incorrectly
    #[test]
    fn test_new_logistic_regression_dimension_and_weights_provided_incorrect() {
        let weights = DVector::from_vec(vec![0.5, -0.5]);
        let model = LogisticRegression::<f64, u8>::new(Some(2), Some(weights));
        assert!(model.is_err());
    }

    #[test]
    fn test_h_function() {
        let mut model = LogisticRegression::<f64, u8>::new(Some(2), None).unwrap();

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
        let model =
            LogisticRegression::<f64, u8>::new(None, Some(DVector::from_vec(vec![0.0, 0.5, -0.5])))
                .unwrap();

        let features = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let features_with_bias = features.clone().insert_column(0, 1.0);
        let predictions = model.predict(&features_with_bias);

        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|&p| p == 0 || p == 1));
    }

    // Add more tests for fit, weights update, gradient calculation, etc.

    // Test sigmoid function
    #[test]
    fn test_sigmoid() {
        let value = LogisticRegression::<f64, u8>::sigmoid(0.0);
        assert!((value - 0.5).abs() < f64::EPSILON);
    }

    // Test cross-entropy calculation
    #[test]
    fn test_cross_entropy() {
        let model =
            LogisticRegression::<f64, u8>::new(None, Some(DVector::from_vec(vec![0.0, 0.5, -0.5])))
                .unwrap();

        // Create features and labels for testing
        let features = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let labels = DVector::from_vec(vec![1, 0]);

        // Compute cross-entropy loss
        let loss = model.cross_entropy(&features, &labels).unwrap();
        // Expected loss value
        let expected_loss = 0.7240769841801067;

        // Check if the computed loss is close to the expected value
        assert!((loss - expected_loss).abs() < f64::EPSILON);
    }
}
