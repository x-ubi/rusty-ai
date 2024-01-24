use super::node::TreeNode;
use super::params::TreeClassifierParams;
use crate::data::dataset::{Dataset, Number, WholeNumber};
use crate::metrics::confusion::ClassificationMetrics;
use nalgebra::{DMatrix, DVector};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::f64;
use std::marker::PhantomData;

struct SplitData<XT: Number, YT: WholeNumber> {
    pub feature_index: usize,
    pub threshold: XT,
    pub left: Dataset<XT, YT>,
    pub right: Dataset<XT, YT>,
    information_gain: f64,
}
/// Implementation of a decision tree classifier.
///
/// This struct represents a decision tree classifier, which is a supervised machine learning algorithm
/// used for classification tasks. It can be used to build a decision tree from a dataset and make
/// predictions on new data.
///
/// # Type Parameters
///
/// - `XT`: The type of the features in the dataset.
/// - `YT`: The type of the labels in the dataset.
///
/// # Examples
///
/// ```
/// use rusty_ai::trees::classifier::DecisionTreeClassifier;
/// use rusty_ai::data::dataset::Dataset;
/// use nalgebra::{DMatrix, DVector};
///
/// // Create a new decision tree classifier
/// let mut tree = DecisionTreeClassifier::<f64, u8>::new();
///
/// // Set the minimum number of samples required to split an internal node
/// tree.set_min_samples_split(5).unwrap();
///
/// // Set the maximum depth of the tree
/// tree.set_max_depth(Some(10)).unwrap();
///
///
///
/// let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let y = DVector::from_vec(vec![0, 1, 0]);
/// let dataset = Dataset::new(x, y);
/// tree.fit(&dataset).unwrap();
///
/// // Make predictions on new data points
/// let x_test = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
/// let predictions = tree.predict(&x_test);
/// assert!(predictions.is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct DecisionTreeClassifier<XT: Number, YT: WholeNumber> {
    root: Option<Box<TreeNode<XT, YT>>>,
    tree_params: TreeClassifierParams,

    _marker: PhantomData<XT>,
}

impl<XT: Number, YT: WholeNumber> ClassificationMetrics<YT> for DecisionTreeClassifier<XT, YT> {}

impl<XT: Number, YT: WholeNumber> Default for DecisionTreeClassifier<XT, YT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<XT: Number, YT: WholeNumber> DecisionTreeClassifier<XT, YT> {
    pub fn new() -> Self {
        Self {
            root: None,
            tree_params: TreeClassifierParams::new(),

            _marker: PhantomData,
        }
    }

    /// Creates a new instance of the decision tree classifier with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `criterion` - The criterion used for splitting nodes. Default is "gini".
    /// * `min_samples_split` - The minimum number of samples required to split an internal node. Default is 2.
    /// * `max_depth` - The maximum depth of the tree. Default is None (unlimited depth).
    ///
    /// # Returns
    ///
    /// A new instance of the decision tree classifier with the specified parameters.
    ///
    /// # Errors
    ///
    /// This method will return an error if the classifier is unknown, the minimum number of samples to split is less than 2, or if the maximum depth is less than 1.
    pub fn with_params(
        criterion: Option<String>,
        min_samples_split: Option<u16>,
        max_depth: Option<u16>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut tree = Self::new();
        tree.set_criterion(criterion.unwrap_or("gini".to_string()))?;
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

    /// Sets the criterion used for splitting nodes.
    ///
    /// # Arguments
    ///
    /// * `criterion` - The criterion used for splitting nodes.
    ///
    /// # Errors
    ///
    /// This method will return an error if the criterion is not supported.
    pub fn set_criterion(&mut self, criterion: String) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_criterion(criterion)
    }

    /// Returns the maximum depth of the tree.
    pub fn max_depth(&self) -> Option<u16> {
        self.tree_params.max_depth()
    }

    /// Returns the minimum number of samples required to split an internal node.
    pub fn min_samples_split(&self) -> u16 {
        self.tree_params.min_samples_split()
    }

    /// Returns the criterion used for splitting nodes.
    pub fn criterion(&self) -> &str {
        self.tree_params.criterion()
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
    pub fn fit(&mut self, dataset: &Dataset<XT, YT>) -> Result<String, Box<dyn Error>> {
        self.root = Some(Box::new(
            self.build_tree(dataset, self.max_depth().map(|_| 0))?,
        ));
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
    /// A vector containing the predicted labels for the new data.
    ///
    /// # Errors
    ///
    /// This method will return an error if the tree wasn't built yet.
    pub fn predict(&self, features: &DMatrix<XT>) -> Result<DVector<YT>, Box<dyn Error>> {
        if self.root.is_none() {
            return Err("Tree wasn't built yet.".into());
        }

        let predictions: Vec<_> = features
            .row_iter()
            .map(|row| Self::make_prediction(row.transpose(), self.root.as_ref().unwrap()))
            .collect();

        Ok(DVector::from_vec(predictions))
    }

    fn make_prediction(features: DVector<XT>, node: &TreeNode<XT, YT>) -> YT {
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
        dataset: &Dataset<XT, YT>,
        current_depth: Option<u16>,
    ) -> Result<TreeNode<XT, YT>, Box<dyn Error>> {
        let (x, y) = &dataset.into_parts();
        let (num_samples, num_features) = x.shape();
        let is_data_homogenous = y.iter().all(|&val| val == y[0]);

        if num_samples >= self.min_samples_split().into()
            && current_depth <= self.max_depth()
            && !is_data_homogenous
        {
            let splits = (0..num_features)
                .into_par_iter()
                .map(|feature_idx| {
                    self.get_split(dataset, feature_idx)
                        .map_err(|err| err.to_string())
                })
                .collect::<Vec<_>>();

            let valid_splits = splits
                .into_iter()
                .filter_map(Result::ok)
                .collect::<Vec<_>>();

            if valid_splits.is_empty() {
                return Ok(TreeNode::new(self.leaf_value(y.clone_owned())));
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
                let left_node = self.build_tree(&left_child, new_depth)?;
                let right_node = self.build_tree(&right_child, new_depth)?;
                return Ok(TreeNode {
                    feature_index: Some(best_split.feature_index),
                    threshold: Some(best_split.threshold),
                    left: Some(Box::new(left_node)),
                    right: Some(Box::new(right_node)),
                    value: None,
                });
            }
        }

        let leaf_value = self.leaf_value(y.clone_owned());
        Ok(TreeNode::new(leaf_value))
    }

    fn leaf_value(&self, y: DVector<YT>) -> Option<YT> {
        let mut class_counts = HashMap::new();
        for item in y.iter() {
            *class_counts.entry(item).or_insert(0) += 1;
        }
        class_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(val, _)| *val)
    }

    fn get_split(
        &self,
        dataset: &Dataset<XT, YT>,
        feature_index: usize,
    ) -> Result<SplitData<XT, YT>, String> {
        let mut best_split: Option<SplitData<XT, YT>> = None;
        let mut best_information_gain = f64::NEG_INFINITY;

        let mut unique_values: Vec<_> = dataset.x.column(feature_index).iter().cloned().collect();
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_values.dedup();

        for value in &unique_values {
            let (left_child, right_child) = dataset.split_on_threshold(feature_index, *value);

            if left_child.is_not_empty() && right_child.is_not_empty() {
                let current_information_gain =
                    self.calculate_information_gain(&dataset.y, &left_child.y, &right_child.y);

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

        best_split.ok_or(String::from("No split found."))
    }

    fn calculate_information_gain(
        &self,
        parent_y: &DVector<YT>,
        left_y: &DVector<YT>,
        right_y: &DVector<YT>,
    ) -> f64 {
        let weight_left = left_y.len() as f64 / parent_y.len() as f64;
        let weight_right = right_y.len() as f64 / parent_y.len() as f64;

        match self.criterion() {
            "gini" => {
                Self::gini_impurity(parent_y)
                    - weight_left * Self::gini_impurity(left_y)
                    - weight_right * Self::gini_impurity(right_y)
            }
            _ => {
                Self::entropy(parent_y)
                    - weight_left * Self::entropy(left_y)
                    - weight_right * Self::entropy(right_y)
            }
        }
    }

    fn gini_impurity(y: &DVector<YT>) -> f64 {
        let classes: HashSet<_> = y.iter().collect();
        let mut impurity = 0.0;
        for class in classes.into_iter() {
            let p_class = y.iter().filter(|&x| x == class).count() as f64 / y.len() as f64;
            impurity += p_class * p_class;
        }
        1.0 - impurity
    }

    fn entropy(y: &DVector<YT>) -> f64 {
        let classes: HashSet<_> = y.iter().collect();
        let mut entropy = 0.0;
        for class in classes.into_iter() {
            let p_class = y.iter().filter(|&x| x == class).count() as f64 / y.len() as f64;
            entropy += p_class * p_class.log2();
        }
        -entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_default() {
        let tree = DecisionTreeClassifier::<f64, u8>::default();
        assert_eq!(tree.min_samples_split(), 2); // Default min_samples_split
        assert_eq!(tree.max_depth(), None); // Default max_depth
        assert_eq!(tree.criterion(), "gini"); // Default criterion
    }

    #[test]
    fn test_too_low_min_samples() {
        let tree = DecisionTreeClassifier::<f64, u8>::new().set_min_samples_split(0);
        assert!(tree.is_err());
        assert_eq!(
            tree.unwrap_err().to_string(),
            "The minimum number of samples to split must be greater than 1."
        );
    }

    #[test]
    fn test_to_low_depth() {
        let tree = DecisionTreeClassifier::<f64, u8>::new().set_max_depth(Some(0));
        assert!(tree.is_err());
        assert_eq!(
            tree.unwrap_err().to_string(),
            "The maximum depth must be greater than 0."
        );
    }

    #[test]
    fn test_calculate_information_gain() {
        let classifier = DecisionTreeClassifier::<f64, u8>::new();
        let parent_y = DVector::from_vec(vec![1, 1, 0, 0]);
        let left_y = DVector::from_vec(vec![1, 1]);
        let right_y = DVector::from_vec(vec![0, 0]);

        let result = classifier.calculate_information_gain(&parent_y, &left_y, &right_y);
        assert_eq!(result, 0.5); // replace with your expected result
    }

    #[test]
    fn test_gini_impurity_homogeneous() {
        let y = DVector::from_vec(vec![1, 1, 1, 1]);
        assert_eq!(DecisionTreeClassifier::<f64, u32>::gini_impurity(&y), 0.0);
    }

    #[test]
    fn test_gini_impurity_mixed() {
        let y = DVector::from_vec(vec![1, 0, 1, 0]);
        assert!((DecisionTreeClassifier::<f64, u32>::gini_impurity(&y) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gini_impurity_multiple_classes() {
        let y = DVector::from_vec(vec![1, 2, 1, 2, 3]);
        let expected_impurity =
            1.0 - (2.0 / 5.0) * (2.0 / 5.0) - (2.0 / 5.0) * (2.0 / 5.0) - (1.0 / 5.0) * (1.0 / 5.0);
        assert!(
            (DecisionTreeClassifier::<f64, u32>::gini_impurity(&y) - expected_impurity).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_entropy() {
        let y = DVector::from_vec(vec![1, 1, 0, 0]);
        assert_eq!(DecisionTreeClassifier::<f64, u32>::entropy(&y), 1.0);
    }

    #[test]
    fn test_entropy_homogeneous() {
        let y = DVector::from_vec(vec![1, 1, 1, 1]);
        assert_eq!(DecisionTreeClassifier::<f64, u32>::entropy(&y), 0.0);
    }

    #[test]
    fn test_information_gain_gini() {
        let classifier = DecisionTreeClassifier::<f64, u32>::new();
        let parent_y = DVector::from_vec(vec![1, 1, 1, 0, 0, 1]);
        let left_y = DVector::from_vec(vec![1, 1]);
        let right_y = DVector::from_vec(vec![1, 0, 0, 1]);

        let parent_impurity = DecisionTreeClassifier::<f64, u32>::gini_impurity(&parent_y);
        let left_impurity = DecisionTreeClassifier::<f64, u32>::gini_impurity(&left_y);
        let right_impurity = DecisionTreeClassifier::<f64, u32>::gini_impurity(&right_y);

        let weight_left = left_y.len() as f64 / parent_y.len() as f64;
        let weight_right = right_y.len() as f64 / parent_y.len() as f64;
        let expected_gain =
            parent_impurity - (weight_left * left_impurity + weight_right * right_impurity);

        let result = classifier.calculate_information_gain(&parent_y, &left_y, &right_y);
        assert!((result - expected_gain).abs() < f64::EPSILON);
    }

    #[test]
    fn test_information_gain_entropy() {
        let mut classifier = DecisionTreeClassifier::<f64, u32>::new();
        classifier.set_criterion("entropy".to_string()).unwrap();
        let parent_y = DVector::from_vec(vec![1, 1, 1, 0, 0, 1]);
        let left_y = DVector::from_vec(vec![1, 1]);
        let right_y = DVector::from_vec(vec![1, 0, 0, 1]);

        let parent_impurity = DecisionTreeClassifier::<f64, u32>::entropy(&parent_y);
        let left_impurity = DecisionTreeClassifier::<f64, u32>::entropy(&left_y);
        let right_impurity = DecisionTreeClassifier::<f64, u32>::entropy(&right_y);

        let weight_left = left_y.len() as f64 / parent_y.len() as f64;
        let weight_right = right_y.len() as f64 / parent_y.len() as f64;
        let expected_gain =
            parent_impurity - (weight_left * left_impurity + weight_right * right_impurity);

        let result = classifier.calculate_information_gain(&parent_y, &left_y, &right_y);

        assert!((result - expected_gain).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tree_building() {
        let mut classifier = DecisionTreeClassifier::<f64, u32>::new();

        // Assuming a simple dataset with two features
        let x = DMatrix::from_row_slice(
            4,
            2,
            &[
                1.0, 2.0, // Sample 1
                1.1, 2.1, // Sample 2
                2.0, 3.0, // Sample 3
                2.1, 3.1, // Sample 4
            ],
        );
        let y = DVector::from_vec(vec![0, 0, 1, 1]); // Target values
        let dataset = Dataset::new(x, y);

        let _ = classifier.fit(&dataset);

        // Check if the root of the tree is correctly set
        assert!(classifier.root.is_some());

        // Further checks would depend on your tree structure and the expected outcome after fitting the dataset
    }

    #[test]
    fn test_empty_predict() {
        let classifier = DecisionTreeClassifier::<f64, u32>::new();
        let features = DMatrix::from_row_slice(0, 0, &[]);
        let result = classifier.predict(&features);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "Tree wasn't built yet.");
    }
}
