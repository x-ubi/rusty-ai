use std::error::Error;

use nalgebra::{DMatrix, DVector};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::{
    data::dataset::{Dataset, RealNumber},
    metrics::errors::RegressionMetrics,
    trees::{params::TreeParams, regressor::DecisionTreeRegressor},
};

use super::params::ForestParams;

#[derive(Clone, Debug)]
pub struct RandomForestRegressor<T: RealNumber> {
    forest_params: ForestParams<DecisionTreeRegressor<T>>,
    tree_params: TreeParams,
}

impl<T: RealNumber> Default for RandomForestRegressor<T> {
    /// Creates a new `RandomForestRegressor` with default parameters.
    ///
    /// # Returns
    ///
    /// A new instance of the `RandomForestRegressor`.
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealNumber> RegressionMetrics<T> for RandomForestRegressor<T> {}

impl<T: RealNumber> RandomForestRegressor<T> {
    /// Creates a new `RandomForestRegressor` with default parameters.
    ///
    /// # Returns
    ///
    /// A new instance of the `RandomForestRegressor`.
    pub fn new() -> Self {
        Self {
            forest_params: ForestParams::new(),
            tree_params: TreeParams::new(),
        }
    }

    /// Creates a new `RandomForestRegressor` with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `num_trees` - The number of trees in the random forest. If not specified, the default value is 3.
    /// * `min_samples_split` - The minimum number of samples required to split an internal node. If not specified, the default value is 2.
    /// * `max_depth` - The maximum depth of the decision trees. If not specified, there is no maximum depth.
    /// * `sample_size` - The size of the random subsets of the training data used to train each tree. If not specified, the default value is calculated as the total number of samples divided by the number of trees.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `RandomForestRegressor` if the parameters are valid, or a `Box<dyn Error>` if an error occurs.
    pub fn with_params(
        num_trees: Option<usize>,
        min_samples_split: Option<u16>,
        max_depth: Option<u16>,
        sample_size: Option<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut forest = Self::new();

        forest.set_num_trees(num_trees.unwrap_or(3))?;
        forest.set_sample_size(sample_size)?;
        forest.set_min_samples_split(min_samples_split.unwrap_or(2))?;
        forest.set_max_depth(max_depth)?;
        Ok(forest)
    }

    /// Sets the decision trees for the random forest regressor.
    ///
    /// # Arguments
    ///
    /// * `trees` - A vector of `DecisionTreeRegressor` instances.
    pub fn set_trees(&mut self, trees: Vec<DecisionTreeRegressor<T>>) {
        self.forest_params.set_trees(trees);
    }

    /// Sets the number of trees in the random forest regressor.
    ///
    /// # Arguments
    ///
    /// * `num_trees` - The number of trees.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, otherwise returns an error.
    pub fn set_num_trees(&mut self, num_trees: usize) -> Result<(), Box<dyn Error>> {
        self.forest_params.set_num_trees(num_trees)
    }

    /// Sets the sample size for each tree in the random forest regressor.
    ///
    /// # Arguments
    ///
    /// * `sample_size` - The sample size for each tree. Use `None` for full sample size.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, otherwise returns an error.
    pub fn set_sample_size(&mut self, sample_size: Option<usize>) -> Result<(), Box<dyn Error>> {
        self.forest_params.set_sample_size(sample_size)
    }

    /// Sets the minimum number of samples required to split an internal node in each decision tree.
    ///
    /// # Arguments
    ///
    /// * `min_samples_split` - The minimum number of samples required to split an internal node.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, otherwise returns an error.
    pub fn set_min_samples_split(&mut self, min_samples_split: u16) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_min_samples_split(min_samples_split)
    }

    /// Sets the maximum depth of each decision tree in the random forest regressor.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - The maximum depth of each decision tree. Use `None` for unlimited depth.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, otherwise returns an error.
    pub fn set_max_depth(&mut self, max_depth: Option<u16>) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_max_depth(max_depth)
    }

    /// Returns a reference to the decision trees in the random forest regressor.
    pub fn trees(&self) -> &Vec<DecisionTreeRegressor<T>> {
        self.forest_params.trees()
    }

    /// Returns the number of trees in the random forest regressor.
    pub fn num_trees(&self) -> usize {
        self.forest_params.num_trees()
    }

    /// Returns the sample size for each tree in the random forest regressor.
    pub fn sample_size(&self) -> Option<usize> {
        self.forest_params.sample_size()
    }

    /// Returns the minimum number of samples required to split an internal node in each decision tree.
    pub fn min_samples_split(&self) -> u16 {
        self.tree_params.min_samples_split()
    }

    /// Returns the maximum depth of each decision tree in the random forest regressor.
    pub fn max_depth(&self) -> Option<u16> {
        self.tree_params.max_depth()
    }

    /// Fits the random forest regressor to the given dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to fit the random forest regressor to.
    /// * `seed` - The seed for the random number generator. Use `None` for a random seed.
    ///
    /// # Returns
    ///
    /// Returns a string indicating the completion of the fitting process if successful, otherwise returns an error.
    pub fn fit(
        &mut self,
        dataset: &Dataset<T, T>,
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
            Some(sample_size) if sample_size > dataset.x.nrows() => {
                return Err("The set sample size is greater than the dataset size.".into())
            }
            None => self.set_sample_size(Some(dataset.x.nrows() / self.num_trees()))?,
            _ => {}
        }
        let trees: Result<Vec<_>, String> = seeds
            .into_par_iter()
            .map(|tree_seed| {
                let subset = dataset.samples(self.sample_size().unwrap(), Some(tree_seed));
                let mut tree = DecisionTreeRegressor::with_params(
                    Some(self.min_samples_split()),
                    self.max_depth(),
                )
                .map_err(|error| error.to_string())?;
                tree.fit(&subset).map_err(|error| error.to_string())?;
                Ok(tree)
            })
            .collect();
        self.set_trees(trees?);
        Ok("Finished building the trees.".into())
    }

    /// Predicts the target values for the given features using the random forest regressor.
    ///
    /// # Arguments
    ///
    /// * `features` - The features to predict the target values for.
    ///
    /// # Returns
    ///
    /// Returns a vector of predicted target values if successful, otherwise returns an error.
    pub fn predict(&self, features: &DMatrix<T>) -> Result<DVector<T>, Box<dyn Error>> {
        let mut predictions = DVector::from_element(features.nrows(), T::from_f64(0.0).unwrap());

        for i in 0..features.nrows() {
            let mut total_prediction = T::from_f64(0.0).unwrap();
            for tree in self.trees() {
                let prediction = tree.predict(&DMatrix::from_row_slice(
                    1,
                    features.ncols(),
                    features.row(i).transpose().as_slice(),
                ))?;
                total_prediction += prediction[0];
            }

            predictions[i] = total_prediction / T::from_usize(self.trees().len()).unwrap();
        }
        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    // Helper function to create a small mock dataset
    fn create_mock_dataset() -> Dataset<f64, f64> {
        let x = DMatrix::from_row_slice(
            6,
            2,
            &[1.0, 2.0, 1.1, 2.1, 1.2, 2.2, 3.0, 4.0, 3.1, 4.1, 3.2, 4.2],
        );
        let y = DVector::from_vec(vec![0.5, 0.5, 0.5, 1.5, 1.5, 1.5]);
        Dataset::new(x, y)
    }

    #[test]
    fn test_default() {
        let forest = RandomForestRegressor::<f64>::default();
        assert_eq!(forest.num_trees(), 3);
        assert_eq!(forest.min_samples_split(), 2);
    }

    #[test]
    fn test_with_params() {
        let forest =
            RandomForestRegressor::<f64>::with_params(Some(10), Some(4), Some(5), Some(100))
                .unwrap();
        assert_eq!(forest.num_trees(), 10);
        assert_eq!(forest.min_samples_split(), 4);
        assert_eq!(forest.max_depth(), Some(5));
        assert_eq!(forest.sample_size(), Some(100));
    }

    #[test]
    fn test_fit() {
        let mut forest = RandomForestRegressor::<f64>::new();
        let dataset = create_mock_dataset();
        let fit_result = forest.fit(&dataset, Some(42));
        assert!(fit_result.is_ok());
        assert_eq!(forest.trees().len(), 3);
    }

    #[test]
    fn test_predict() {
        let mut forest = RandomForestRegressor::<f64>::new();
        let dataset = create_mock_dataset();
        forest.fit(&dataset, Some(42)).unwrap();

        let features = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let predictions = forest.predict(&features).unwrap();
        assert_eq!(predictions.len(), 2);

        assert!(predictions[0] <= 1.5 && predictions[0] >= 0.5);
        assert!(predictions[1] <= 1.5 && predictions[1] >= 0.5);
    }
}
