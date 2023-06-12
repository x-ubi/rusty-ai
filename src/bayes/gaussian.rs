use nalgebra::{DMatrix, DVector};
use std::collections::{HashMap, HashSet};

/// For now, the implementation is set to work on integer classes and floating point data.
/// This is because I wanted to have AN implementation ready to go.
/// It will be genericized further down the line.
pub struct GaussianNB {
    class_freq: HashMap<i32, f64>,
    class_mean: HashMap<i32, DVector<f64>>,
    class_variance: HashMap<i32, DVector<f64>>,
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussianNB {
    pub fn new() -> Self {
        GaussianNB {
            class_freq: HashMap::new(),
            class_mean: HashMap::new(),
            class_variance: HashMap::new(),
        }
    }

    pub fn fit(&mut self, x: &DMatrix<f64>, y: &DVector<i32>) {
        let classes = y.iter().cloned().collect::<HashSet<_>>();

        for class in classes {
            let class_mask = y.map(|label| label == class);
            let class_indices = class_mask
                .iter()
                .enumerate()
                .filter(|&(_, &value)| value)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            let x_class = x.select_rows(class_indices.as_slice());

            let mean = DVector::from_fn(x_class.ncols(), |col, _| x_class.column(col).mean());
            let variance =
                DVector::from_fn(x_class.ncols(), |col, _| x_class.column(col).variance());

            let freq = class_indices.len() as f64 / x.nrows() as f64;

            self.class_freq.insert(class, freq);
            self.class_mean.insert(class, mean);
            self.class_variance.insert(class, variance);
        }
    }

    fn predict_single(&self, x: &DVector<f64>) -> i32 {
        let mut max_log_likelihood = f64::MIN;
        let mut max_class = 0;

        for class in self.class_freq.keys() {
            let mean = self.class_mean.get(class).unwrap();
            let variance = self.class_variance.get(class).unwrap();

            let log_likelihood = -0.5
                * ((x - mean)
                    .component_mul(&(x - mean))
                    .component_div(&(variance.scale(2.0))))
                .sum()
                - 0.5 * variance.map(|v| v.ln()).sum()
                + self.class_freq.get(class).unwrap().ln();

            if log_likelihood > max_log_likelihood {
                max_log_likelihood = log_likelihood;
                max_class = *class;
            }
        }
        max_class
    }

    pub fn predict(&self, x: &DMatrix<f64>) -> DVector<i32> {
        let mut y_pred = Vec::new();

        for i in 0..x.nrows() {
            let x_row = x.row(i).into_owned().transpose();
            let class = self.predict_single(&x_row);
            y_pred.push(class);
        }

        DVector::from(y_pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_default_initialization() {
        let clf = GaussianNB::new();

        assert!(clf.class_freq.is_empty());
        assert!(clf.class_mean.is_empty());
        assert!(clf.class_variance.is_empty());
    }

    #[test]
    fn test_model_fit() {
        let mut clf = GaussianNB::new();

        let x = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let y = DVector::from_column_slice(&[0, 0, 1, 1]);

        clf.fit(&x, &y);

        assert_abs_diff_eq!(*clf.class_freq.get(&0).unwrap(), 0.5, epsilon = 1e-7);
        assert_abs_diff_eq!(*clf.class_freq.get(&1).unwrap(), 0.5, epsilon = 1e-7);
    }

    #[test]
    fn test_predictions() {
        let mut clf = GaussianNB::new();

        let x = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let y = DVector::from_column_slice(&[0, 0, 1, 1]);
        clf.fit(&x, &y);

        let test_x = DMatrix::from_row_slice(2, 3, &[2.0, 3.0, 4.0, 6.0, 7.0, 8.0]);

        let pred_y = clf.predict(&test_x);

        assert_eq!(pred_y, DVector::from_column_slice(&[0, 1]));
    }

    #[test]
    fn test_empty_data() {
        let mut clf = GaussianNB::new();
        let empty_x = DMatrix::<f64>::zeros(0, 0);
        let empty_y = DVector::<i32>::zeros(0);
        let empty_pred_y = clf.predict(&empty_x);
        assert_eq!(empty_pred_y.len(), 0);

        clf.fit(&empty_x, &empty_y);
        assert_eq!(clf.class_freq.len(), 0);
        assert_eq!(clf.class_mean.len(), 0);
        assert_eq!(clf.class_variance.len(), 0);
    }

    #[test]
    fn test_single_class() {
        let mut clf = GaussianNB::new();

        let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 2.0, 3.0, 3.0, 4.0]);
        let y = DVector::from_column_slice(&[0, 0, 0]);
        clf.fit(&x, &y);

        assert_eq!(clf.class_freq.len(), 1);
        assert_eq!(clf.class_mean.len(), 1);
        assert_eq!(clf.class_variance.len(), 1);

        let test_x = DMatrix::from_row_slice(2, 2, &[1.5, 2.5, 2.5, 3.5]);

        let pred_y = clf.predict(&test_x);

        assert_eq!(pred_y, DVector::from_column_slice(&[0, 0]));
    }

    #[test]
    fn test_gaussian_nb() {
        let mut clf = GaussianNB::new();

        let x = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let y = DVector::from_column_slice(&[0, 0, 1, 1]);

        clf.fit(&x, &y);

        assert_abs_diff_eq!(*clf.class_freq.get(&0).unwrap(), 0.5, epsilon = 1e-7);
        assert_abs_diff_eq!(*clf.class_freq.get(&1).unwrap(), 0.5, epsilon = 1e-7);

        let test_x = DMatrix::from_row_slice(2, 3, &[2.0, 3.0, 4.0, 6.0, 7.0, 8.0]);

        let pred_y = clf.predict(&test_x);

        assert_eq!(pred_y, DVector::from_column_slice(&[0, 1]));
    }
}
