use std::error::Error;

use nalgebra::DVector;

use crate::data::dataset::RealNumber;

pub trait RegressionMetrics<T: RealNumber> {
    fn mse(&self, y_true: &DVector<T>, y_pred: &DVector<T>) -> Result<T, Box<dyn Error>> {
        if y_true.len() != y_pred.len() {
            return Err("Predictions and labels are of different sizes.".into());
        }

        let n = T::from_usize(y_true.len()).ok_or("Couldn't transform from usize")?;
        let errors = y_pred - y_true;
        let errors_sq = errors.component_mul(&errors);

        Ok(errors_sq.sum() / n)
    }

    fn mae(&self, y_true: &DVector<T>, y_pred: &DVector<T>) -> Result<T, Box<dyn Error>> {
        if y_true.len() != y_pred.len() {
            return Err("Predictions and labels are of different sizes.".into());
        }
        let n = T::from_usize(y_true.len()).ok_or("Couldn't transform from usize")?;
        let abs_errors_sum = y_pred
            .iter()
            .zip(y_true.iter())
            .map(|(&y_p, &y_t)| (y_p - y_t).abs())
            .fold(
                T::from_f64(0.0).ok_or("Couldn't transform from f64")?,
                |acc, x| acc + x,
            );

        Ok(abs_errors_sum / n)
    }

    fn r2(&self, y_true: &DVector<T>, y_pred: &DVector<T>) -> Result<T, Box<dyn Error>> {
        if y_true.len() != y_pred.len() {
            return Err("Predictions and labels are of different sizes.".into());
        }
        let n = T::from_usize(y_true.len()).ok_or("Couldn't transform from usize")?;

        let y_true_mean = y_true.sum() / n;

        let y_true_mean_vec = DVector::from_element(y_true.len(), y_true_mean);

        let mse_model = self.mse(y_true, y_pred)?;
        let mse_base = self.mse(&y_true_mean_vec, y_pred)?;

        Ok(T::from_f64(1.0).ok_or("Couldn't transform from f64")? - (mse_model / mse_base))
    }
}
