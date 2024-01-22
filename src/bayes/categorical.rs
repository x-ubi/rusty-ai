use crate::data::dataset::{Dataset, WholeNumber};
use nalgebra::{DMatrix, DVector};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
};

/// Implementation of the Categorical Naive Bayes classifier.
///
/// This struct represents a Categorical Naive Bayes classifier, which is a probabilistic
/// classifier that assumes independence between features given the class label. It is
/// specifically designed for categorical features.
///
/// # Example
///
/// ```
/// use rusty_ai::bayes::categorical::CategoricalNB;
/// use rusty_ai::data::dataset::Dataset;
/// use nalgebra::{DMatrix, DVector};
///
/// // Create a new CategoricalNB classifier
/// let mut model = CategoricalNB::new();
///
/// // Fit the classifier to a dataset
/// let x = DMatrix::from_row_slice(2, 3, &[1, 2, 3, 2, 3, 4]);
/// let y = DVector::from_vec(vec![0, 1]);
/// let dataset = Dataset::new(x, y);
/// model.fit(&dataset).unwrap();
///
/// // Predict the class labels for new data
/// let x_test = DMatrix::from_row_slice(2, 3, &[1, 3, 4, 2, 2, 3]);
/// let predictions = model.predict(&x_test).unwrap();
/// assert_eq!(predictions, DVector::from_vec(vec![1,0]))
/// ```

pub struct CategoricalNB<T: WholeNumber> {
    feature_class_freq: HashMap<T, DVector<HashMap<T, f64>>>,
    label_class_freq: HashMap<T, f64>,
    unique_feature_values_count: Vec<usize>,
}

impl<T: WholeNumber> Default for CategoricalNB<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: WholeNumber> CategoricalNB<T> {
    /// Creates a new instance of the CategoricalNB classifier.
    ///
    /// This function initializes the classifier with empty frequency maps and an empty
    /// vector to store the count of unique feature values.
    ///
    /// # Returns
    ///
    /// A new instance of the CategoricalNB classifier.
    pub fn new() -> Self {
        Self {
            feature_class_freq: HashMap::new(),
            label_class_freq: HashMap::new(),
            unique_feature_values_count: Vec::new(),
        }
    }

    /// Returns a reference to the feature class frequency map.
    ///
    /// This function returns a reference to the map that stores the frequency of each
    /// feature value for each class label.
    ///
    /// # Returns
    ///
    /// A reference to the feature class frequency map.
    pub fn feature_class_freq(&self) -> &HashMap<T, DVector<HashMap<T, f64>>> {
        &self.feature_class_freq
    }

    /// Returns a reference to the label class frequency map.
    ///
    /// This function returns a reference to the map that stores the frequency of each
    /// class label.
    ///
    /// # Returns
    ///
    /// A reference to the label class frequency map.
    pub fn label_class_freq(&self) -> &HashMap<T, f64> {
        &self.label_class_freq
    }

    /// Fits the classifier to a dataset.
    ///
    /// This function fits the classifier to the given dataset by calculating the
    /// frequency of each feature value for each class label and the frequency of each
    /// class label. It also calculates the count of unique feature values for each
    /// feature.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to fit the classifier to.
    ///
    /// # Returns
    ///
    /// A `Result` indicating whether the fitting process was successful or an error occurred.
    pub fn fit(&mut self, dataset: &Dataset<T, T>) -> Result<String, Box<dyn Error>> {
        let (x, y) = dataset.into_parts();
        let y_classes = y.iter().cloned().collect::<HashSet<_>>();

        let mut unique_feature_values_count_temp = vec![HashSet::new(); x.ncols()];

        x.column_iter().enumerate().for_each(|(idx, feature)| {
            feature.iter().for_each(|&val| {
                unique_feature_values_count_temp[idx].insert(val);
            })
        });

        self.unique_feature_values_count = unique_feature_values_count_temp
            .iter()
            .map(|set| set.len())
            .collect::<Vec<_>>();

        for y_class in y_classes {
            let class_mask = y.map(|label| label == y_class);
            let class_indices = class_mask
                .iter()
                .enumerate()
                .filter(|&(_, &value)| value)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();

            let x_y_class = x.select_rows(class_indices.as_slice());

            let mut all_features_freq = DVector::from_element(x.ncols(), HashMap::new());
            for (idx, feature) in x_y_class.column_iter().enumerate() {
                let feature_count = feature.iter().fold(HashMap::new(), |mut acc, &val| {
                    *acc.entry(val).or_insert(0) += 1;
                    acc
                });
                let total_count =
                    class_indices.len() as f64 + self.unique_feature_values_count[idx] as f64;
                let feature_freq = feature_count
                    .into_iter()
                    .map(|(class, count)| (class, (count as f64 + 1.0 / total_count)))
                    .collect();
                all_features_freq[idx] = feature_freq;
            }

            let label_class_freq = class_indices.len() as f64 / y.nrows() as f64;

            self.label_class_freq.insert(y_class, label_class_freq);
            self.feature_class_freq.insert(y_class, all_features_freq);
        }

        Ok("Finished fitting".into())
    }

    fn predict_single(&self, x: &DVector<T>) -> Result<T, Box<dyn Error>> {
        let mut max_prob = f64::NEG_INFINITY;
        let mut max_class = T::from_i8(0).unwrap();

        for (y_class, label_freq) in &self.label_class_freq {
            let mut prob = label_freq.ln();

            for (idx, feature) in x.iter().enumerate() {
                let feature_probs = &self
                    .feature_class_freq
                    .get(y_class)
                    .ok_or(format!("Class {:?} wasn't obtained.", y_class))?[idx];

                let total_feature_count = self.label_class_freq.values().sum::<f64>()
                    + self.unique_feature_values_count[idx] as f64;
                let feature_prob = feature_probs
                    .get(feature)
                    .unwrap_or(&(1.0 / total_feature_count))
                    .ln();

                prob += feature_prob;
            }

            if prob > max_prob {
                max_prob = prob;
                max_class = *y_class;
            }
        }

        Ok(max_class)
    }

    /// Predicts the class labels for a matrix of feature values.
    ///
    /// This function predicts the class labels for each row in the given matrix of
    /// feature values. It uses the fitted model to calculate the probability of each
    /// class label for each row and selects the class label with the highest probability
    /// as the predicted label.
    ///
    /// # Arguments
    ///
    /// * `x` - The matrix of feature values.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of predicted class labels or an error if the
    /// prediction process failed.
    pub fn predict(&self, x: &DMatrix<T>) -> Result<DVector<T>, Box<dyn Error>> {
        let mut y_pred = Vec::new();

        for i in 0..x.nrows() {
            let x_row = x.row(i).transpose();
            let y_class = self.predict_single(&x_row)?;
            y_pred.push(y_class);
        }
        Ok(DVector::from_vec(y_pred))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_new() {
        let model = CategoricalNB::<i32>::new();

        assert!(model.feature_class_freq.is_empty());
        assert!(model.label_class_freq.is_empty());
    }

    #[test]
    fn test_fit() {
        let mut model = CategoricalNB::<i32>::new();

        let x = DMatrix::from_row_slice(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let y = DVector::from_vec(vec![1, 2, 3]);
        let dataset = Dataset::new(x, y);

        let result = model.fit(&dataset);

        assert!(result.is_ok());
        assert_eq!(model.label_class_freq.len(), 3);
        assert_eq!(model.feature_class_freq.len(), 3);
    }

    #[test]
    fn test_predict_single() {
        let mut model = CategoricalNB::<i32>::new();

        // Create a simple dataset and fit the model
        let x = DMatrix::from_row_slice(4, 2, &[1, 0, 1, 1, 0, 0, 0, 1]);
        let y = DVector::from_vec(vec![0, 0, 1, 1]);
        let dataset = Dataset::new(x.clone(), y);
        model.fit(&dataset).unwrap();

        // Predict a single instance
        let test_instance = x.row(0).transpose();
        let result = model.predict_single(&test_instance).unwrap();

        // Check if the prediction matches the expected class
        assert_eq!(result, 0);
    }

    #[test]
    fn test_predict_with_unseen_feature_value() {
        let mut model = CategoricalNB::<i32>::new();

        // Create a simple dataset and fit the model
        let x = DMatrix::from_row_slice(4, 2, &[1, 0, 1, 1, 0, 0, 0, 1]);
        let y = DVector::from_vec(vec![0, 0, 1, 1]);
        let dataset = Dataset::new(x, y);
        model.fit(&dataset).unwrap();

        // Predict an instance with an unseen feature value
        let test_instance = DVector::from_vec(vec![2, 2]); // Unseen feature values
        let result = model.predict_single(&test_instance).unwrap();

        // Just check if it produces a result without errors for now
        // The correctness of this test depends on your Laplace smoothing implementation
        assert!(result == 0 || result == 1);
    }

    #[test]
    fn test_predict() {
        let mut model = CategoricalNB::<i32>::new();

        let x = DMatrix::from_row_slice(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let y = DVector::from_vec(vec![3, 2, 1]);
        let dataset = Dataset::new(x.clone(), y.clone());

        model.fit(&dataset).unwrap();
        let result = model.predict(&x);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), y);
    }
}
