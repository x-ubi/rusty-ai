use crate::{
    data::dataset::{Dataset, RealNumber, WholeNumber},
    metrics::confusion::ClassificationMetrics,
};
use nalgebra::{DMatrix, DVector};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
};

/// Implementation of Gaussian Naive Bayes classifier.
///
/// This struct represents a Gaussian Naive Bayes classifier. It is used to fit a training dataset
/// and make predictions on new data points. The classifier assumes that the features are
/// independent and follow a Gaussian distribution.
///
/// # Type Parameters
///
/// * `XT`: The type of the input features.
/// * `YT`: The type of the target labels.
///
/// # Example
///
/// ```
/// use rusty_ai::bayes::gaussian::GaussianNB;
/// use rusty_ai::data::dataset::Dataset;
/// use nalgebra::{DMatrix, DVector};
///
/// // Create a new Gaussian Naive Bayes classifier
/// let mut classifier = GaussianNB::new();
///
/// // Create a training dataset
/// let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let y = DVector::from_vec(vec![0, 1, 0]);
/// let dataset = Dataset::new(x, y);
///
/// // Fit the classifier to the training dataset
/// let result = classifier.fit(&dataset);
/// assert!(result.is_ok());
///
/// // Make predictions on new data points
/// let x_test = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
/// let predictions = classifier.predict(&x_test);
/// assert!(predictions.is_ok());
/// ```

#[derive(Clone, Debug)]
pub struct GaussianNB<XT: RealNumber, YT: WholeNumber> {
    class_freq: HashMap<YT, XT>,
    class_mean: HashMap<YT, DVector<XT>>,
    class_variance: HashMap<YT, DVector<XT>>,
}

impl<XT: RealNumber, YT: WholeNumber> ClassificationMetrics<YT> for GaussianNB<XT, YT> {}

impl<XT: RealNumber, YT: WholeNumber> Default for GaussianNB<XT, YT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<XT: RealNumber, YT: WholeNumber> GaussianNB<XT, YT> {
    /// Creates a new Gaussian Naive Bayes classifier.
    ///
    /// This function initializes the classifier with empty class frequency, mean, and variance
    /// maps.
    ///
    /// # Returns
    ///
    /// A new instance of `GaussianNB`.
    pub fn new() -> Self {
        Self {
            class_freq: HashMap::new(),
            class_mean: HashMap::new(),
            class_variance: HashMap::new(),
        }
    }

    /// Returns a reference to the class frequency map.
    ///
    /// This function returns a reference to the map that stores the frequency of each class in
    /// the training dataset.
    ///
    /// # Returns
    ///
    /// A reference to the class frequency map.
    pub fn class_freq(&self) -> &HashMap<YT, XT> {
        &self.class_freq
    }

    /// Returns a reference to the class mean map.
    ///
    /// This function returns a reference to the map that stores the mean values of each feature
    /// for each class in the training dataset.
    ///
    /// # Returns
    ///
    /// A reference to the class mean map.
    pub fn class_mean(&self) -> &HashMap<YT, DVector<XT>> {
        &self.class_mean
    }

    /// Returns a reference to the class variance map.
    ///
    /// This function returns a reference to the map that stores the variance values of each
    /// feature for each class in the training dataset.
    ///
    /// # Returns
    ///
    /// A reference to the class variance map.
    pub fn class_variance(&self) -> &HashMap<YT, DVector<XT>> {
        &self.class_variance
    }

    /// Fits the classifier to a training dataset.
    ///
    /// This function fits the classifier to the provided training dataset. It calculates the
    /// class frequency, mean, and variance for each class in the dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The training dataset to fit the classifier to.
    ///
    /// # Returns
    ///
    /// A `Result` indicating whether the fitting process was successful or an error occurred.
    pub fn fit(&mut self, dataset: &Dataset<XT, YT>) -> Result<String, Box<dyn Error>> {
        let (x, y) = dataset.into_parts();
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

            let mean = DVector::from_fn(x_class.ncols(), |col, _| {
                self.mean(&x_class.column(col).into_owned())
            });
            let variance = DVector::from_fn(x_class.ncols(), |col, _| {
                self.variance(&x_class.column(col).into_owned())
            });

            let freq =
                XT::from_usize(class_indices.len()).unwrap() / XT::from_usize(x.nrows()).unwrap();

            self.class_freq.insert(class, freq);
            self.class_mean.insert(class, mean);
            self.class_variance.insert(class, variance);
        }
        Ok("Finished fitting".into())
    }

    fn mean(&self, x: &DVector<XT>) -> XT {
        let zero = XT::from_f64(0.0).unwrap();
        let sum: XT = x.fold(zero, |acc, x| acc + x);

        sum / XT::from_usize(x.len()).unwrap()
    }

    fn variance(&self, x: &DVector<XT>) -> XT {
        let mean = self.mean(x);
        let zero = XT::from_f64(0.0).unwrap();
        let numerator = x.fold(zero, |acc, x| acc + (x - mean) * (x - mean));

        numerator / XT::from_usize(x.len() - 1).unwrap()
    }

    fn predict_single(&self, x: &DVector<XT>) -> Result<YT, Box<dyn Error>> {
        let mut max_log_likelihood = XT::from_f64(f64::NEG_INFINITY).unwrap();
        let mut max_class = YT::from_i8(0).unwrap();

        for class in self.class_freq.keys() {
            let mean = self
                .class_mean
                .get(class)
                .ok_or(format!("Mean for class {:?} wasn't calculated.", class))?;
            let variance = self
                .class_variance
                .get(class)
                .ok_or(format!("Variance for class {:?} wasn't calculated.", class))?;
            let variance_epsilon =
                DVector::<XT>::from_element(variance.len(), XT::from_f64(1e-9).unwrap());

            let starting = XT::from_f64(-0.5).unwrap();
            let log_likelihood = starting
                * ((x - mean).component_mul(&(x - mean)).component_div(
                    &(variance.map(|v| v * XT::from_f64(2.0).unwrap()) + &variance_epsilon),
                ))
                .sum()
                + starting * (variance + &variance_epsilon).map(|v| v.ln()).sum()
                + self
                    .class_freq
                    .get(class)
                    .ok_or(format!("Frequency of class {:?} wasn't obtained.", class))?
                    .ln();

            if log_likelihood > max_log_likelihood {
                max_log_likelihood = log_likelihood;
                max_class = *class;
            }
        }
        Ok(max_class)
    }

    /// Predicts the class labels for a given matrix of feature vectors.
    ///
    /// This function predicts the class labels for each feature vector in the input matrix using
    /// the fitted Gaussian Naive Bayes classifier. It returns a vector of predicted class labels.
    ///
    /// # Arguments
    ///
    /// * `x` - The matrix of feature vectors to predict the class labels for.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of predicted class labels or an error if the prediction
    /// process fails.
    pub fn predict(&self, x: &DMatrix<XT>) -> Result<DVector<YT>, Box<dyn Error>> {
        let mut y_pred = Vec::new();

        for i in 0..x.nrows() {
            let x_row = x.row(i).into_owned().transpose();
            let class = self.predict_single(&x_row)?;
            y_pred.push(class);
        }

        Ok(DVector::from_vec(y_pred))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_new() {
        let clf = GaussianNB::<f64, i32>::new();

        assert!(clf.class_freq.is_empty());
        assert!(clf.class_mean.is_empty());
        assert!(clf.class_variance.is_empty());
    }

    #[test]
    fn test_model_fit() {
        let mut clf = GaussianNB::<f64, i32>::new();

        let x = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let y = DVector::from_column_slice(&[0, 0, 1, 1]);
        let dataset = Dataset::new(x, y);

        let _ = clf.fit(&dataset);

        assert_abs_diff_eq!(*clf.class_freq.get(&0).unwrap(), 0.5, epsilon = 1e-7);
        assert_abs_diff_eq!(*clf.class_freq.get(&1).unwrap(), 0.5, epsilon = 1e-7);
    }

    #[test]
    fn test_predictions() {
        let mut clf = GaussianNB::<f64, i32>::new();

        let x = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let y = DVector::from_column_slice(&[0, 0, 1, 1]);
        let dataset = Dataset::new(x, y);

        let _ = clf.fit(&dataset);

        let test_x = DMatrix::from_row_slice(2, 3, &[2.0, 3.0, 4.0, 6.0, 7.0, 8.0]);

        let pred_y = clf.predict(&test_x).unwrap();

        assert_eq!(pred_y, DVector::from_column_slice(&[0, 1]));
    }

    #[test]
    fn test_empty_data() {
        let mut clf = GaussianNB::<f64, i32>::new();
        let empty_x = DMatrix::<f64>::zeros(0, 0);
        let empty_y = DVector::<i32>::zeros(0);
        let empty_pred_y = clf.predict(&empty_x).unwrap();
        assert_eq!(empty_pred_y.len(), 0);
        let dataset = Dataset::new(empty_x, empty_y);

        let _ = clf.fit(&dataset);
        assert_eq!(clf.class_freq.len(), 0);
        assert_eq!(clf.class_mean.len(), 0);
        assert_eq!(clf.class_variance.len(), 0);
    }

    #[test]
    fn test_single_class() {
        let mut clf = GaussianNB::<f64, i32>::new();

        let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 2.0, 3.0, 3.0, 4.0]);
        let y = DVector::from_column_slice(&[0, 0, 0]);
        let dataset = Dataset::new(x, y);

        let _ = clf.fit(&dataset);

        assert_eq!(clf.class_freq.len(), 1);
        assert_eq!(clf.class_mean.len(), 1);
        assert_eq!(clf.class_variance.len(), 1);

        let test_x = DMatrix::from_row_slice(2, 2, &[1.5, 2.5, 2.5, 3.5]);

        let pred_y = clf.predict(&test_x).unwrap();

        assert_eq!(pred_y, DVector::from_column_slice(&[0, 0]));
    }

    #[test]
    fn test_predict_with_constant_feature() {
        let mut clf = GaussianNB::<f64, i32>::new();

        let x = DMatrix::from_row_slice(4, 2, &[0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let y = DVector::from_vec(vec![0, 0, 1, 1]);

        let x_new = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 1.0]);
        let dataset = Dataset::new(x, y);

        let _ = clf.fit(&dataset);

        let y_hat = clf.predict(&x_new).unwrap();

        assert_eq!(y_hat.len(), 2);
        assert_eq!(y_hat[0], 0);
        assert_eq!(y_hat[1], 1);
    }

    #[test]
    fn test_gaussian_nb() {
        let mut clf = GaussianNB::<f64, i32>::new();

        let x = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let y = DVector::from_column_slice(&[0, 0, 1, 1]);
        let dataset = Dataset::new(x, y);

        let _ = clf.fit(&dataset);

        assert_abs_diff_eq!(*clf.class_freq.get(&0).unwrap(), 0.5, epsilon = 1e-7);
        assert_abs_diff_eq!(*clf.class_freq.get(&1).unwrap(), 0.5, epsilon = 1e-7);

        let test_x = DMatrix::from_row_slice(2, 3, &[2.0, 3.0, 4.0, 6.0, 7.0, 8.0]);

        let pred_y = clf.predict(&test_x).unwrap();

        assert_eq!(pred_y, DVector::from_column_slice(&[0, 1]));
    }
}
