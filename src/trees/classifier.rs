//! Decision Tree Classifier
use super::base::TreeNode;
use crate::dataset::{Dataset, FeatureValue, TargetValue};
use nalgebra::{DMatrix, DVector};
use std::collections::{HashMap, HashSet};
use std::f64;
use std::hash::Hash;
use std::marker::PhantomData;

struct SplitData<XT: FeatureValue, YT: TargetValue + Eq + Hash> {
    pub feature_index: usize,
    pub threshold: XT,
    pub left: Dataset<XT, YT>,
    pub right: Dataset<XT, YT>,
    information_gain: f64,
}

pub struct DecisionTreeClassifier<XT: FeatureValue, YT: TargetValue + Eq + Hash> {
    pub root: Option<Box<TreeNode<XT, YT>>>,
    pub min_samples_split: u16,
    pub max_depth: Option<u16>,
    criterion: String,

    _marker: PhantomData<XT>,
}

impl<XT: FeatureValue, YT: TargetValue + Eq + Hash> DecisionTreeClassifier<XT, YT> {
    pub fn new() -> Self {
        Self {
            root: None,
            min_samples_split: 2,
            max_depth: None,
            criterion: "gini".to_string(),

            _marker: PhantomData,
        }
    }

    pub fn with_params(
        criterion: Option<String>,
        min_samples_split: Option<u16>,
        max_depth: Option<u16>,
    ) -> Self {
        Self {
            root: None,
            min_samples_split: min_samples_split.unwrap_or(2),
            max_depth,
            criterion: criterion.unwrap_or("gini".to_string()),

            _marker: PhantomData,
        }
    }

    pub fn fit(&mut self, dataset: Dataset<XT, YT>) {
        self.root = Some(Box::new(
            self.build_tree(dataset, self.max_depth.map(|_| 0)),
        ));
    }

    pub fn predict(&self, prediction_features: &DMatrix<XT>) -> DVector<YT> {
        let predictions: Vec<_> = prediction_features
            .row_iter()
            .map(|row| self.make_prediction(row.transpose(), self.root.as_ref().unwrap()))
            .collect();

        DVector::from_vec(predictions)
    }

    fn make_prediction(&self, features: DVector<XT>, node: &TreeNode<XT, YT>) -> YT {
        if let Some(value) = &node.value {
            return *value;
        }
        match &features[node.feature_index.unwrap()] {
            x if x <= node.threshold.as_ref().unwrap() => {
                return self.make_prediction(features, node.left.as_ref().unwrap())
            }
            _ => return self.make_prediction(features, node.right.as_ref().unwrap()),
        }
    }

    fn build_tree(
        &mut self,
        dataset: Dataset<XT, YT>,
        current_depth: Option<u16>,
    ) -> TreeNode<XT, YT> {
        let (x, y) = &dataset.into_parts();
        let (num_samples, num_features) = x.shape();
        if num_samples >= self.min_samples_split.into() && current_depth <= self.max_depth {
            let best_split = self.get_best_split(&dataset, num_features).unwrap();
            let left_child = best_split.left;
            let right_child = best_split.right;
            if best_split.information_gain > 0.0 {
                let new_depth = current_depth.map(|depth| depth + 1);
                let left_node = self.build_tree(left_child, new_depth);
                let right_node = self.build_tree(right_child, new_depth);
                return TreeNode {
                    feature_index: Some(best_split.feature_index),
                    threshold: Some(best_split.threshold),
                    left: Some(Box::new(left_node)),
                    right: Some(Box::new(right_node)),
                    value: None,
                };
            }
        }

        let leaf_value = self.leaf_value(y.clone_owned());
        TreeNode::new(leaf_value)
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

    fn get_best_split(
        &self,
        dataset: &Dataset<XT, YT>,
        num_features: usize,
    ) -> Option<SplitData<XT, YT>> {
        let mut best_split: Option<SplitData<XT, YT>> = None;
        let mut best_information_gain = f64::NEG_INFINITY;

        for feature_index in 0..num_features {
            let mut unique_values: Vec<_> =
                dataset.x.column(feature_index).iter().cloned().collect();
            unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            unique_values.dedup();

            for value in &unique_values {
                let (left_child, right_child) = dataset.split(feature_index, *value);

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
        }
        best_split
    }

    fn calculate_information_gain(
        &self,
        parent_y: &DVector<YT>,
        left_y: &DVector<YT>,
        right_y: &DVector<YT>,
    ) -> f64 {
        let weight_left = left_y.len() as f64 / parent_y.len() as f64;
        let weight_right = right_y.len() as f64 / parent_y.len() as f64;

        if self.criterion == "gini" {
            return self.gini_impurity(parent_y)
                - weight_left * self.gini_impurity(left_y)
                - weight_right * self.gini_impurity(right_y);
        }
        0.0
    }

    fn gini_impurity(&self, y: &DVector<YT>) -> f64 {
        let classes: HashSet<_> = y.iter().collect();
        let mut impurity = 0.0;
        for class in classes.into_iter() {
            let p_class = y.iter().filter(|&x| x == class).count() as f64 / y.len() as f64;
            impurity += p_class * p_class;
        }
        1.0 - impurity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

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
        let classifier = DecisionTreeClassifier::<f64, u32>::new();
        let y = DVector::from_vec(vec![1, 1, 1, 1]); // Homogeneous set
        assert_eq!(classifier.gini_impurity(&y), 0.0);
    }

    #[test]
    fn test_gini_impurity_mixed() {
        let classifier = DecisionTreeClassifier::<f64, u32>::new();
        let y = DVector::from_vec(vec![1, 0, 1, 0]); // Evenly split set
        assert!((classifier.gini_impurity(&y) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gini_impurity_multiple_classes() {
        let classifier = DecisionTreeClassifier::<f64, u32>::new();
        let y = DVector::from_vec(vec![1, 2, 1, 2, 3]); // Three classes
        let expected_impurity =
            1.0 - (2.0 / 5.0) * (2.0 / 5.0) - (2.0 / 5.0) * (2.0 / 5.0) - (1.0 / 5.0) * (1.0 / 5.0);
        assert!((classifier.gini_impurity(&y) - expected_impurity).abs() < f64::EPSILON);
    }

    #[test]
    fn test_information_gain() {
        let classifier = DecisionTreeClassifier::<f64, u32>::new();
        let parent_y = DVector::from_vec(vec![1, 1, 1, 0, 0, 1]);
        let left_y = DVector::from_vec(vec![1, 1]);
        let right_y = DVector::from_vec(vec![1, 0, 0, 1]);

        let parent_impurity = classifier.gini_impurity(&parent_y);
        let left_impurity = classifier.gini_impurity(&left_y);
        let right_impurity = classifier.gini_impurity(&right_y);

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

        classifier.fit(dataset);

        // Check if the root of the tree is correctly set
        assert!(classifier.root.is_some());

        // Further checks would depend on your tree structure and the expected outcome after fitting the dataset
    }
}
