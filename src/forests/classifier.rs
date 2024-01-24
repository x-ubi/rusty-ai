use crate::data::dataset::{Dataset, Number, WholeNumber};
use crate::metrics::confusion::ClassificationMetrics;
use crate::trees::classifier::DecisionTreeClassifier;
use crate::trees::params::TreeClassifierParams;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;

use super::params::ForestParams;

#[derive(Clone, Debug)]
pub struct RandomForestClassifier<XT: Number, YT: WholeNumber> {
    forest_params: ForestParams<DecisionTreeClassifier<XT, YT>>,
    tree_params: TreeClassifierParams,
}

impl<XT: Number, YT: WholeNumber> ClassificationMetrics<YT> for RandomForestClassifier<XT, YT> {}

impl<XT: Number, YT: WholeNumber> Default for RandomForestClassifier<XT, YT> {
    fn default() -> Self {
        Self::new()
    }
}

/// This module contains the implementation of the `RandomForestClassifier` struct.
///
/// The `RandomForestClassifier` is a machine learning algorithm that combines multiple decision trees to make predictions.
/// It is used for classification tasks where the input features are of type `XT` and the target labels are of type `YT`.
///
/// # Example
///
/// ```rust
/// use rusty_ai::forests::classifier::RandomForestClassifier;
/// use rusty_ai::data::dataset::Dataset;
/// use nalgebra::{DMatrix, DVector};
///
/// // Create a mock dataset
/// let x = DMatrix::from_row_slice(
///     6,
///     2,
///     &[1.0, 2.0, 1.1, 2.1, 1.2, 2.2, 3.0, 4.0, 3.1, 4.1, 3.2, 4.2],
/// );
/// let y = DVector::from_vec(vec![0, 0, 0, 1, 1, 1]);
/// let dataset = Dataset::new(x, y);
///
/// // Create a random forest classifier with default parameters
/// let mut forest = RandomForestClassifier::<f64, u8>::default();
///
/// // Fit the classifier to the dataset
/// forest.fit(&dataset, Some(42)).unwrap();
///
/// // Make predictions on new features
/// let features = DMatrix::from_row_slice(
///     2,
///     2,
///     &[
///         1.0, 2.0, // Should be classified as class 0
///         3.0, 4.0, // Should be classified as class 1
///     ],
/// );
/// let predictions = forest.predict(&features).unwrap();
/// println!("Predictions: {:?}", predictions);
/// ```

impl<XT: Number, YT: WholeNumber> RandomForestClassifier<XT, YT> {
    /// Creates a new instance of the Random Forest Classifier.
    ///
    /// This function initializes the classifier with empty frequency maps and an empty
    /// vector to store the count of unique feature values.
    ///
    /// # Returns
    ///
    /// A new instance of the Random Forest Classifier.
    pub fn new() -> Self {
        Self {
            forest_params: ForestParams::new(),
            tree_params: TreeClassifierParams::new(),
        }
    }

    /// Creates a new instance of the Random Forest Classifier with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `num_trees` - The number of trees in the forest. If not specified, defaults to 3.
    /// * `min_samples_split` - The minimum number of samples required to split an internal node. If not specified, defaults to 2.
    /// * `max_depth` - The maximum depth of the decision trees. If not specified, defaults to None.
    /// * `criterion` - The function to measure the quality of a split. If not specified, defaults to "gini".
    /// * `sample_size` - The size of the random subsets of the dataset to train each tree. If not specified, defaults to None.
    ///
    /// # Returns
    ///
    /// A `Result` containing the Random Forest Classifier instance or an error.
    pub fn with_params(
        num_trees: Option<usize>,
        min_samples_split: Option<u16>,
        max_depth: Option<u16>,
        criterion: Option<String>,
        sample_size: Option<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut forest = Self::new();

        forest.set_num_trees(num_trees.unwrap_or(3))?;
        forest.set_sample_size(sample_size)?;
        forest.set_min_samples_split(min_samples_split.unwrap_or(2))?;
        forest.set_max_depth(max_depth)?;
        forest.set_criterion(criterion.unwrap_or("gini".to_string()))?;
        Ok(forest)
    }

    /// Sets the decision trees of the random forest.
    ///
    /// # Arguments
    ///
    /// * `trees` - A vector of DecisionTreeClassifier instances.
    pub fn set_trees(&mut self, trees: Vec<DecisionTreeClassifier<XT, YT>>) {
        self.forest_params.set_trees(trees);
    }

    /// Sets the number of trees in the random forest.
    ///
    /// # Arguments
    ///
    /// * `num_trees` - The number of trees.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or an error.
    pub fn set_num_trees(&mut self, num_trees: usize) -> Result<(), Box<dyn Error>> {
        self.forest_params.set_num_trees(num_trees)
    }

    /// Sets the sample size for each tree in the random forest.
    ///
    /// # Arguments
    ///
    /// * `sample_size` - The sample size.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or an error.
    pub fn set_sample_size(&mut self, sample_size: Option<usize>) -> Result<(), Box<dyn Error>> {
        self.forest_params.set_sample_size(sample_size)
    }

    /// Sets the minimum number of samples required to split an internal node in each decision tree.
    ///
    /// # Arguments
    ///
    /// * `min_samples_split` - The minimum number of samples.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or an error.
    pub fn set_min_samples_split(&mut self, min_samples_split: u16) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_min_samples_split(min_samples_split)
    }

    /// Sets the maximum depth of each decision tree in the random forest.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - The maximum depth.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or an error.
    pub fn set_max_depth(&mut self, max_depth: Option<u16>) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_max_depth(max_depth)
    }

    /// Sets the criterion function to measure the quality of a split in each decision tree.
    ///
    /// # Arguments
    ///
    /// * `criterion` - The criterion function.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or an error.
    pub fn set_criterion(&mut self, criterion: String) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_criterion(criterion)
    }

    /// Returns a reference to the decision trees in the random forest.
    pub fn trees(&self) -> &Vec<DecisionTreeClassifier<XT, YT>> {
        self.forest_params.trees()
    }

    /// Returns the number of trees in the random forest.
    pub fn num_trees(&self) -> usize {
        self.forest_params.num_trees()
    }

    /// Returns the sample size for each tree in the random forest.
    pub fn sample_size(&self) -> Option<usize> {
        self.forest_params.sample_size()
    }

    /// Returns the minimum number of samples required to split an internal node in each decision tree.
    pub fn min_samples_split(&self) -> u16 {
        self.tree_params.min_samples_split()
    }

    /// Returns the maximum depth of each decision tree in the random forest.
    pub fn max_depth(&self) -> Option<u16> {
        self.tree_params.max_depth()
    }

    /// Returns a reference to the criterion function used to measure the quality of a split in each decision tree.
    pub fn criterion(&self) -> &String {
        &self.tree_params.criterion
    }

    /// Fits the random forest to the given dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to fit the random forest to.
    /// * `seed` - The seed for the random number generator used to generate random subsets of the dataset. If not specified, a random seed will be used.
    ///
    /// # Returns
    ///
    /// A `Result` indicating whether the fitting process was successful or an error occurred.
    pub fn fit(
        &mut self,
        dataset: &Dataset<XT, YT>,
        seed: Option<u64>,
    ) -> Result<String, Box<dyn Error>> {
        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            _ => StdRng::from_entropy(),
        };

        let seeds = (0..self.num_trees())
            .map(|_| rng.gen::<u64>())
            .collect::<Vec<_>>();

        match self.sample_size() {
            Some(sample_size) if sample_size > dataset.nrows() => {
                return Err(format!(
                    "The set sample size is greater than the dataset size. {} > {}",
                    sample_size,
                    dataset.nrows()
                )
                .into());
            }
            None => self.set_sample_size(Some(dataset.nrows() / self.num_trees()))?,
            _ => {}
        }

        let trees: Result<Vec<_>, String> = seeds
            .into_par_iter()
            .map(|tree_seed| {
                let subset = dataset.samples(self.sample_size().unwrap(), Some(tree_seed));
                let mut tree = DecisionTreeClassifier::with_params(
                    Some(self.criterion().clone()),
                    Some(self.min_samples_split()),
                    self.max_depth(),
                )
                .map_err(|error| error.to_string())?;
                tree.fit(&subset).map_err(|error| error.to_string())?;
                Ok(tree)
            })
            .collect();
        self.set_trees(trees?);
        Ok("Finished building the trees".into())
    }

    /// Predicts the class labels for the given features using the random forest.
    ///
    /// # Arguments
    ///
    /// * `features` - The features to predict the class labels for.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of predicted class labels or an error if the prediction
    /// process fails.
    pub fn predict(&self, features: &DMatrix<XT>) -> Result<DVector<YT>, Box<dyn Error>> {
        let mut predictions = DVector::from_element(features.nrows(), YT::from_u8(0).unwrap());

        for i in 0..features.nrows() {
            let mut class_counts = HashMap::new();
            for tree in self.trees() {
                let prediction = tree
                    .predict(&DMatrix::from_row_slice(
                        1,
                        features.ncols(),
                        features.row(i).transpose().as_slice(),
                    ))
                    .map_err(|error| error.to_string())?;
                *class_counts.entry(prediction[0]).or_insert(0) += 1;
            }

            let chosen_class = class_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(class, _)| class)
                .ok_or(
                    "Prediction failure. No trees built or class counts are empty.".to_string(),
                )?;
            predictions[i] = chosen_class;
        }
        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_mock_dataset() -> Dataset<f64, u8> {
        let x = DMatrix::from_row_slice(
            6,
            2,
            &[1.0, 2.0, 1.1, 2.1, 1.2, 2.2, 3.0, 4.0, 3.1, 4.1, 3.2, 4.2],
        );
        let y = DVector::from_vec(vec![0, 0, 0, 1, 1, 1]);
        Dataset::new(x, y)
    }

    #[test]
    fn test_default() {
        let forest = RandomForestClassifier::<f64, u8>::default();
        assert_eq!(forest.num_trees(), 3); // Default number of trees
        assert_eq!(forest.min_samples_split(), 2); // Default min_samples_split
    }

    #[test]
    fn test_new() {
        let forest = RandomForestClassifier::<f64, u8>::new();
        assert_eq!(forest.num_trees(), 3); // Default number of trees
        assert_eq!(forest.min_samples_split(), 2); // Default min_samples_split
    }

    #[test]
    fn test_with_params() {
        let forest = RandomForestClassifier::<f64, u8>::with_params(
            Some(10),                    // num_trees
            Some(4),                     // min_samples_split
            Some(5),                     // max_depth
            Some("entropy".to_string()), // criterion
            Some(100),                   // sample_size
        )
        .unwrap();
        assert_eq!(forest.num_trees(), 10);
        assert_eq!(forest.min_samples_split(), 4);
        assert_eq!(forest.max_depth(), Some(5));
        assert_eq!(forest.criterion(), "entropy");
        assert_eq!(forest.sample_size(), Some(100));
    }

    #[test]
    fn test_too_low_sample_size() {
        let forest = RandomForestClassifier::<f64, u8>::new().set_sample_size(Some(0));
        assert!(forest.is_err());
        assert_eq!(
            forest.unwrap_err().to_string(),
            "The sample size must be greater than 0."
        );
    }

    #[test]
    fn test_too_low_num_trees() {
        let forest = RandomForestClassifier::<f64, u8>::new().set_num_trees(1);
        assert!(forest.is_err());
        assert_eq!(
            forest.unwrap_err().to_string(),
            "The number of trees must be greater than 1."
        );
    }

    #[test]
    fn test_fit() {
        let mut forest = RandomForestClassifier::<f64, u8>::new();
        let dataset = create_mock_dataset();
        let fit_result = forest.fit(&dataset, Some(42)); // Using a fixed seed for reproducibility
        assert!(fit_result.is_ok());
        assert_eq!(forest.trees().len(), 3); // Should have 3 trees after fitting
    }

    #[test]
    fn test_fit_too_many_samples() {
        let mut forest = RandomForestClassifier::<f64, u8>::new();
        let _ = forest.set_sample_size(Some(1000));
        let dataset = create_mock_dataset();
        let fit_result = forest.fit(&dataset, Some(42)); // Using a fixed seed for reproducibility

        assert!(fit_result.is_err());
        assert_eq!(
            fit_result.unwrap_err().to_string(),
            "The set sample size is greater than the dataset size. 1000 > 6"
        );
    }

    #[test]
    fn test_predict() {
        let mut forest = RandomForestClassifier::<f64, u8>::new();
        let _ = forest.set_sample_size(Some(3));
        let dataset = create_mock_dataset();
        forest.fit(&dataset, Some(42)).unwrap();

        let features = DMatrix::from_row_slice(
            2,
            2,
            &[
                1.0, 2.0, // Should be classified as class 0
                3.0, 4.0, // Should be classified as class 1
            ],
        );
        let predictions = forest.predict(&features).unwrap();
        assert_eq!(predictions, DVector::from_vec(vec![0, 1]));
    }
}
