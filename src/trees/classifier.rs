//! Decision Tree Classifier
use crate::trees::base::{DecisionTreeBase, TreeNode, SplitData};
use crate::dataset::{Dataset, FeatureValue, TargetValue};
use nalgebra::{DMatrix,DVector};
use std::collections::HashSet;
use std::f64::NEG_INFINITY;

pub struct DecisionTreeClassifier<XT: FeatureValue, YT: TargetValue> {

    base: DecisionTreeBase<XT, YT>,
    criterion: String,
}

impl<XT: FeatureValue, YT: TargetValue> DecisionTreeClassifier<XT, YT> {
    pub fn new(
        criterion: Option<String>,
        min_samples_split: Option<u16>,
        max_depth: Option<u16>,
    ) -> Self {
        Self {
            base: DecisionTreeBase::new(min_samples_split, max_depth),
            criterion: criterion.unwrap_or("gini".to_string()),
        }
    }

    pub fn fit(&mut self, dataset: Dataset<XT, YT>) {
        self.base.root = Some(Box::new(
            self.build_tree(dataset, self.base.max_depth.map(|_| 0)),
        ));
    }

    pub fn predict(&self, features:DMatrix<XT>) -> DVector<YT> {
        self.base.predict(features)
    }

    fn build_tree(
        &mut self,
        dataset: Dataset<XT, YT>,
        current_depth: Option<u16>,
    ) -> TreeNode<XT, YT> {
        let (x, y) = &dataset.into_parts();
        let (num_samples, num_features) = x.shape();
        if num_samples >= self.base.min_samples_split.into() && current_depth <= self.base.max_depth {
            let best_split = self.get_best_split(&dataset, num_features).unwrap();
            let left_child = best_split.left;
            let right_child = best_split.right;
            if best_split.information_gain > 0.0 {
                let new_depth = match current_depth {
                    Some(depth) => Some(depth + 1),
                    _ => None,
                };
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

        let leaf_value = self.base.leaf_value(y.clone_owned());
        TreeNode::new(leaf_value)
    }

    fn get_best_split(
        &self,
        dataset: &Dataset<XT, YT>,
        num_features: usize,
    ) -> Option<SplitData<XT, YT>> {
        let mut best_split: Option<SplitData<XT, YT>> = None;
        let mut best_information_gain = NEG_INFINITY;

        for feature_index in 0..num_features {
            let mut unique_values: Vec<_> =
                dataset.x.column(feature_index).iter().cloned().collect();
            unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            unique_values.dedup();

            for value in &unique_values {
                let (left_child, right_child) = dataset.split(feature_index, value.clone());

                if left_child.is_not_empty() && right_child.is_not_empty() {
                    let current_information_gain = self.calculate_information_gain(
                        dataset.y.clone(),
                        left_child.y.clone(),
                        right_child.y.clone(),
                    );

                    if current_information_gain > best_information_gain {
                        best_split = Some(SplitData {
                            feature_index: feature_index,
                            threshold: value.clone(),
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
        parent_values: DVector<YT>,
        left_values: DVector<YT>,
        right_values: DVector<YT>,
    ) -> f64 {
        let weight_left = left_values.len() as f64 / parent_values.len() as f64;
        let weight_right = right_values.len() as f64 / parent_values.len() as f64;

        if self.criterion == "gini" {
            return self.gini_index(parent_values)
                - weight_left * self.gini_index(left_values)
                - weight_right * self.gini_index(right_values);
        }
        0.0
    }

    fn gini_index(&self, y: DVector<YT>) -> f64 {
        let classes: HashSet<_> = y.iter().collect();
        let mut gini_index = 0.0;
        for class in classes.into_iter() {
            let p_class = y.iter().filter(|&x| x == class).count() as f64 / y.len() as f64;
            gini_index += p_class * p_class;
        }
        gini_index
    }
}
