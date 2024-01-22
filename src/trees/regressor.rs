//! Decision Tree Regressor
use super::{node::TreeNode, params::TreeParams};
use crate::{
    data::dataset::{Dataset, RealNumber},
    metrics::errors::RegressionMetrics,
};
use nalgebra::{DMatrix, DVector};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{error::Error, f64, marker::PhantomData};

pub struct SplitData<T: RealNumber> {
    pub feature_index: usize,
    pub threshold: T,
    pub left: Dataset<T, T>,
    pub right: Dataset<T, T>,
    information_gain: f64,
}

/// Decision Tree Regressor
#[derive(Clone, Debug)]
pub struct DecisionTreeRegressor<T: RealNumber> {
    root: Option<Box<TreeNode<T, T>>>,
    tree_params: TreeParams,

    _marker: PhantomData<T>,
}

impl<T: RealNumber> Default for DecisionTreeRegressor<T> {
            /// Creates a new instance of the decision tree regressor with default parameters.
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealNumber> RegressionMetrics<T> for DecisionTreeRegressor<T> {}

impl<T: RealNumber> DecisionTreeRegressor<T> {
        /// Creates a new instance of the decision tree regressor with default parameters.
    pub fn new() -> Self {
        Self {
            root: None,
            tree_params: TreeParams::new(),
            _marker: PhantomData,
        }
    }

    /// Creates a new instance of the decision tree regressor with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `min_samples_split` - The minimum number of samples required to split an internal node.
    /// * `max_depth` - The maximum depth of the tree.
    ///
    /// # Returns
    ///
    /// A new instance of the decision tree regressor with the specified parameters.
    ///
    /// # Errors
    ///
    /// This method will return an error if the minimum number of samples to split is less than 2 or if the maximum depth is less than 1.
    pub fn with_params(
        min_samples_split: Option<u16>,
        max_depth: Option<u16>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut tree = Self::new();

        tree.set_min_samples_split(min_samples_split.unwrap_or(2))?;
        tree.set_max_depth(max_depth)?;
        Ok(tree)
    }

        /// Sets the minimum number of samples required to split an internal node.
    ///
    /// # Arguments
    ///
    /// * `min_samples_split` - The minimum number of samples required to split an internal node.
    ///
    /// # Errors
    ///
    /// This method will return an error if the minimum number of samples to split is less than 2.
    pub fn set_min_samples_split(&mut self, min_samples_split: u16) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_min_samples_split(min_samples_split)
    }

        /// Sets the maximum depth of the tree.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - The maximum depth of the tree.
    ///
    /// # Errors
    ///
    /// This method will return an error if the maximum depth is less than 1.
    pub fn set_max_depth(&mut self, max_depth: Option<u16>) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_max_depth(max_depth)
    }

        /// Returns the maximum depth of the tree.
    pub fn max_depth(&self) -> Option<u16> {
        self.tree_params.max_depth()
    }

        /// Returns the minimum number of samples required to split an internal node.
    pub fn min_samples_split(&self) -> u16 {
        self.tree_params.min_samples_split()
    }

        /// Builds the decision tree from a dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset containing features and labels.
    ///
    /// # Returns
    ///
    /// A string indicating that the tree was built successfully.
    ///
    /// # Errors
    ///
    /// This method will return an error if the tree couldn't be built.
    pub fn fit(&mut self, dataset: &Dataset<T, T>) -> Result<String, Box<dyn Error>> {
        self.root = Some(Box::new(self.build_tree(
            dataset,
            self.max_depth().map(|_| 0),
            self.variance(&dataset.y),
        )?));
        Ok("Finished building the tree.".into())
    }

    /// Predicts the labels for new data.
    ///
    /// # Arguments
    ///
    /// * `features` - The matrix of features for the new data.
    ///
    /// # Returns
    ///
    /// A vector containing the predicted target values for the new data.
    ///
    /// # Errors
    ///
    /// This method will return an error if the tree wasn't built yet.
    pub fn predict(&self, prediction_features: &DMatrix<T>) -> Result<DVector<T>, String> {
        if self.root.is_none() {
            return Err("Tree wasn't built yet.".to_string());
        }
        let predictions: Vec<_> = prediction_features
            .row_iter()
            .map(|row| Self::make_prediction(row.transpose(), self.root.as_ref().unwrap()))
            .collect();

        Ok(DVector::from_vec(predictions))
    }

    fn make_prediction(features: DVector<T>, node: &TreeNode<T, T>) -> T {
        if let Some(value) = &node.value {
            return *value;
        }
        match &features[node.feature_index.unwrap()] {
            x if x <= node.threshold.as_ref().unwrap() => {
                return Self::make_prediction(features, node.left.as_ref().unwrap())
            }
            _ => return Self::make_prediction(features, node.right.as_ref().unwrap()),
        }
    }

    fn build_tree(
        &mut self,
        dataset: &Dataset<T, T>,
        current_depth: Option<u16>,
        base_variance: f64,
    ) -> Result<TreeNode<T, T>, Box<dyn Error>> {
        let (x, y) = &dataset.into_parts();
        let (num_samples, num_features) = x.shape();

        let is_homogenous = self.variance(y) < 0.01 * base_variance;
        if num_samples >= self.min_samples_split().into()
            && current_depth <= self.max_depth()
            && !is_homogenous
        {
            let splits = (0..num_features)
                .into_par_iter()
                .map(|feature_idx| self.get_split(dataset, feature_idx))
                .collect::<Vec<_>>();

            let valid_splits = splits
                .into_iter()
                .filter_map(Result::ok)
                .collect::<Vec<_>>();

            if valid_splits.is_empty() {
                return Ok(TreeNode::new(Some(self.mean(y))));
            }

            let best_split = match valid_splits.into_iter().max_by(|split1, split2| {
                split1
                    .information_gain
                    .partial_cmp(&split2.information_gain)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                Some(split) => split,
                _ => {
                    return Err("No best split found.".into());
                }
            };
            let left_child = best_split.left;
            let right_child = best_split.right;
            if best_split.information_gain > 0.0 {
                let new_depth = current_depth.map(|depth| depth + 1);
                let left_node = self.build_tree(&left_child, new_depth, base_variance)?;
                let right_node = self.build_tree(&right_child, new_depth, base_variance)?;
                return Ok(TreeNode {
                    feature_index: Some(best_split.feature_index),
                    threshold: Some(best_split.threshold),
                    left: Some(Box::new(left_node)),
                    right: Some(Box::new(right_node)),
                    value: None,
                });
            }
        }

        let leaf_value = self.mean(y);
        Ok(TreeNode::new(Some(leaf_value)))
    }

    fn get_split(
        &self,
        dataset: &Dataset<T, T>,
        feature_index: usize,
    ) -> Result<SplitData<T>, String> {
        let mut best_split: Option<SplitData<T>> = None;
        let mut best_information_gain = f64::NEG_INFINITY;

        let mut unique_values: Vec<_> = dataset.x.column(feature_index).iter().cloned().collect();
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_values.dedup();

        for value in &unique_values {
            let (left_child, right_child) = dataset.split_on_threshold(feature_index, *value);

            if left_child.is_not_empty() && right_child.is_not_empty() {
                let current_information_gain =
                    self.calculate_variance_reduction(&dataset.y, &left_child.y, &right_child.y);

                if current_information_gain > best_information_gain {
                    best_split = Some(SplitData {
                        feature_index,
                        threshold: *value,
                        left: left_child,
                        right: right_child,
                        information_gain: current_information_gain,
                    });
                    best_information_gain = current_information_gain;
                }
            }
        }
        best_split.ok_or("No split found.".into())
    }

    fn calculate_variance_reduction(
        &self,
        parent_y: &DVector<T>,
        left_y: &DVector<T>,
        right_y: &DVector<T>,
    ) -> f64 {
        let variance = self.variance(parent_y);
        let left_variance = self.variance(left_y);
        let right_variance = self.variance(right_y);
        let num_samples = parent_y.len() as f64;
        variance
            - (left_variance * (left_y.len() as f64) / num_samples)
            - (right_variance * (right_y.len() as f64) / num_samples)
    }

    fn variance(&self, y: &DVector<T>) -> f64 {
        let mean = self.mean(y);
        let variance = y.iter().fold(T::from_f64(0.0).unwrap(), |acc, x| {
            acc + (*x - mean) * (*x - mean)
        });
        let variance_f64 = T::to_f64(&variance).unwrap();
        variance_f64 / y.len() as f64
    }

    fn mean(&self, y: &DVector<T>) -> T {
        let zero = T::from_f64(0.0).unwrap();
        let sum: T = y.iter().fold(zero, |acc, x| acc + *x);
        sum / T::from_usize(y.len()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_mean() {
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let regressor: DecisionTreeRegressor<f64> = DecisionTreeRegressor::new();
        let mean = regressor.mean(&y);
        assert_eq!(mean, 3.5);
    }

    #[test]
    fn test_variance() {
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let regressor: DecisionTreeRegressor<f64> = DecisionTreeRegressor::new();
        let variance = regressor.variance(&y);
        assert_eq!(variance, 2.0);
    }

    #[test]
    fn test_calculate_variance_reduction() {
        let parent_y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let left_y = DVector::from_vec(vec![1.0, 2.0]);
        let right_y = DVector::from_vec(vec![3.0, 4.0, 5.0]);
        let regressor: DecisionTreeRegressor<f64> = DecisionTreeRegressor::new();
        let variance_reduction =
            regressor.calculate_variance_reduction(&parent_y, &left_y, &right_y);
        assert!(variance_reduction > 0.0);
    }

    #[test]
    fn test_fit_and_predict() {
        let x = DMatrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = DVector::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
        let dataset = Dataset::new(x, y);
        let mut regressor: DecisionTreeRegressor<f64> = DecisionTreeRegressor::new();
        let _ = regressor.fit(&dataset);

        let test_x = DMatrix::from_vec(3, 1, vec![2.0, 3.0, 4.0]);
        let predictions = regressor.predict(&test_x).unwrap();

        assert_eq!(predictions.len(), 3);
        assert!(predictions.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_fit_and_predict_with_single_row() {
        let x = DMatrix::from_vec(1, 2, vec![1.0, 2.0]);
        let y = DVector::from_vec(vec![1.0]);
        let dataset = Dataset::new(x, y);
        let mut regressor: DecisionTreeRegressor<f64> = DecisionTreeRegressor::new();
        let _ = regressor.fit(&dataset);

        let test_x = DMatrix::from_vec(1, 2, vec![2.0, 3.0]);
        let predictions = regressor.predict(&test_x).unwrap();

        assert_eq!(predictions.len(), 1);
        assert!(predictions.iter().all(|&x| x >= 0.0));
    }
}
