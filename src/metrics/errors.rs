use std::error::Error;

use nalgebra::DVector;

use crate::data::dataset::RealNumber;

/// A trait for computing regression metrics.
pub trait RegressionMetrics<T: RealNumber> {
    /// Computes the mean squared error (MSE) between the true values and the predicted values.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true values.
    /// * `y_pred` - The predicted values.
    ///
    /// # Returns
    ///
    /// The mean squared error.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of `y_true` and `y_pred` are different.
    fn mse(&self, y_true: &DVector<T>, y_pred: &DVector<T>) -> Result<T, Box<dyn Error>> {
        if y_true.len() != y_pred.len() {
            return Err("Predictions and labels are of different sizes.".into());
        }

        let n = T::from_usize(y_true.len()).unwrap();
        let errors = y_pred - y_true;
        let errors_sq = errors.component_mul(&errors);

        Ok(errors_sq.sum() / n)
    }

    /// Computes the mean absolute error (MAE) between the true values and the predicted values.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true values.
    /// * `y_pred` - The predicted values.
    ///
    /// # Returns
    ///
    /// The mean absolute error.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of `y_true` and `y_pred` are different.
    fn mae(&self, y_true: &DVector<T>, y_pred: &DVector<T>) -> Result<T, Box<dyn Error>> {
        if y_true.len() != y_pred.len() {
            return Err("Predictions and labels are of different sizes.".into());
        }
        let n = T::from_usize(y_true.len()).unwrap();
        let abs_errors_sum = y_pred
            .iter()
            .zip(y_true.iter())
            .map(|(&y_p, &y_t)| (y_p - y_t).abs())
            .fold(T::from_f64(0.0).unwrap(), |acc, x| acc + x);

        Ok(abs_errors_sum / n)
    }

    /// Computes the coefficient of determination (R^2) between the true values and the predicted values.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true values.
    /// * `y_pred` - The predicted values.
    ///
    /// # Returns
    ///
    /// The coefficient of determination (R^2).
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths of `y_true` and `y_pred` are different.
    fn r2(&self, y_true: &DVector<T>, y_pred: &DVector<T>) -> Result<T, Box<dyn Error>> {
        if y_true.len() != y_pred.len() {
            return Err("Predictions and labels are of different sizes.".into());
        }
        let n = T::from_usize(y_true.len()).unwrap();

        let y_true_mean = y_true.sum() / n;

        let y_true_mean_vec = DVector::from_element(y_true.len(), y_true_mean);

        let mse_model = self.mse(y_true, y_pred)?;
        let mse_base = self.mse(&y_true_mean_vec, y_true)?;

        Ok(T::from_f64(1.0).unwrap() - (mse_model / mse_base))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    struct MockRegressor;

    impl RegressionMetrics<f64> for MockRegressor {}

    #[test]
    fn test_mse() {
        let regressor = MockRegressor;
        let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = DVector::from_vec(vec![1.1, 1.9, 3.2]);

        let mse = regressor.mse(&y_true, &y_pred).unwrap();
        let expected_mse = ((0.1 * 0.1) + (0.1 * 0.1) + (0.2 * 0.2)) / 3.0;
        assert!((mse - expected_mse).abs() < 1e-6);
    }

    #[test]
    fn test_mae() {
        let regressor = MockRegressor;
        let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = DVector::from_vec(vec![1.1, 1.9, 3.2]);

        let mae = regressor.mae(&y_true, &y_pred).unwrap();
        let expected_mae = (0.1 + 0.1 + 0.2) / 3.0;
        assert!((mae - expected_mae).abs() < 1e-6);
    }

    #[test]
    fn test_r2() {
        let regressor = MockRegressor;
        let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = DVector::from_vec(vec![1.1, 1.9, 3.2]);

        let r2 = regressor.r2(&y_true, &y_pred).unwrap();

        let y_true_mean = y_true.mean();
        let tss: f64 = y_true.iter().map(|&y| (y - y_true_mean).powi(2)).sum();

        let rss: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
            .sum();

        // Calculate expected R2
        let expected_r2 = 1.0 - (rss / tss);

        assert!((r2 - expected_r2).abs() < 1e-6);
    }

    #[test]
    fn test_different_length_error() {
        let regressor = MockRegressor;
        let y_true = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = DVector::from_vec(vec![1.1, 1.9]);

        assert!(regressor.mse(&y_true, &y_pred).is_err());
        assert!(regressor.mae(&y_true, &y_pred).is_err());
        assert!(regressor.r2(&y_true, &y_pred).is_err());
    }
}
