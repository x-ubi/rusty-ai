// Base Decision Tree
use crate::dataset::{Dataset, FeatureValue, TargetValue};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::marker::PhantomData;

pub struct SplitData<XT: FeatureValue, YT: TargetValue> {
    pub feature_index: usize,
    pub threshold: XT,
    pub left: Dataset<XT, YT>,
    pub right: Dataset<XT, YT>,
    pub information_gain: f64,
}

pub struct TreeNode<XT: FeatureValue, YT: TargetValue> {
    pub feature_index: Option<usize>,
    pub threshold: Option<XT>,
    pub left: Option<Box<TreeNode<XT, YT>>>,
    pub right: Option<Box<TreeNode<XT, YT>>>,
    pub value: Option<YT>,
}

impl<XT: FeatureValue, YT: TargetValue> TreeNode<XT, YT> {
    pub fn new(value: Option<YT>) -> Self {
        Self {
            feature_index: None,
            threshold: None,
            left: None,
            right: None,

            value: value,
        }
    }
}

pub struct DecisionTreeBase<XT: FeatureValue, YT: TargetValue> {
    pub root: Option<Box<TreeNode<XT, YT>>>,
    pub min_samples_split: u16,
    pub max_depth: Option<u16>,

    _marker: PhantomData<XT>,
}
impl<XT: FeatureValue, YT: TargetValue> DecisionTreeBase<XT, YT> {
    pub fn new(
        min_samples_split: Option<u16>,
        max_depth: Option<u16>,
    ) -> Self {
        Self {
            root: None,
            min_samples_split: min_samples_split.unwrap_or(2),
            max_depth: max_depth,
            _marker: PhantomData,
        }
    }

    // pub fn fit(&mut self, dataset: Dataset<XT, YT>) {
    //     self.root = Some(Box::new(
    //         self.build_tree(dataset, self.max_depth.map(|_| 0)),
    //     ));
    // }

    pub fn make_prediction(&self, features: DVector<XT>, node: &TreeNode<XT, YT>) -> YT {
        if let Some(value) = &node.value {
            return value.clone();
        }
        match &features[node.feature_index.unwrap()] {
            x if x <= node.threshold.as_ref().unwrap() => {
                return self.make_prediction(features, node.left.as_ref().unwrap())
            }
            _ => return self.make_prediction(features, node.right.as_ref().unwrap()),
        }
    }

    pub fn predict(&self, prediction_features: DMatrix<XT>) -> DVector<YT> {
        let predictions: Vec<_> = prediction_features
            .row_iter()
            .map(|row| self.make_prediction(row.transpose(), self.root.as_ref().unwrap()))
            .collect();

        DVector::from_vec(predictions)
    }

    pub fn leaf_value(&self, y: DVector<YT>) -> Option<YT> {
        let mut class_counts = HashMap::new();
        for item in y.iter() {
            *class_counts.entry(item).or_insert(0) += 1;
        }
        class_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(val, _)| val.clone())
    }

    // pub fn build_tree(
    //     &mut self,
    //     dataset: Dataset<XT, YT>,
    //     current_depth: Option<u16>,
    // ) -> TreeNode<XT, YT> {
    //     let (x, y) = &dataset.into_parts();
    //     let (num_samples, num_features) = x.shape();
    //     if num_samples >= self.min_samples_split.into() && current_depth <= self.max_depth {
    //         let best_split = self.get_best_split(&dataset, num_features).unwrap();
    //         let left_child = best_split.left;
    //         let right_child = best_split.right;
    //         if best_split.information_gain > 0.0 {
    //             let new_depth = match current_depth {
    //                 Some(depth) => Some(depth + 1),
    //                 _ => None,
    //             };
    //             let left_node = self.build_tree(left_child, new_depth);
    //             let right_node = self.build_tree(right_child, new_depth);
    //             return TreeNode {
    //                 feature_index: Some(best_split.feature_index),
    //                 threshold: Some(best_split.threshold),
    //                 left: Some(Box::new(left_node)),
    //                 right: Some(Box::new(right_node)),
    //                 value: None,
    //             };
    //         }
    //     }

    //     let leaf_value = self.leaf_value(y.clone_owned());
    //     TreeNode::new(leaf_value)
    // }

    // fn get_best_split(
    //     &self,
    //     dataset: &Dataset<XT, YT>,
    //     num_features: usize,
    // ) -> Option<SplitData<XT, YT>> {
    //     let mut best_split: Option<SplitData<XT, YT>> = None;
    //     let mut best_information_gain = NEG_INFINITY;

    //     for feature_index in 0..num_features {
    //         let mut unique_values: Vec<_> =
    //             dataset.x.column(feature_index).iter().cloned().collect();
    //         unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    //         unique_values.dedup();

    //         for value in &unique_values {
    //             let (left_child, right_child) = dataset.split(feature_index, value.clone());

    //             if left_child.is_not_empty() && right_child.is_not_empty() {
    //                 let current_information_gain = self.calculate_information_gain(
    //                     dataset.y.clone(),
    //                     left_child.y.clone(),
    //                     right_child.y.clone(),
    //                 );

    //                 if current_information_gain > best_information_gain {
    //                     best_split = Some(SplitData {
    //                         feature_index: feature_index,
    //                         threshold: value.clone(),
    //                         left: left_child,
    //                         right: right_child,
    //                         information_gain: current_information_gain,
    //                     });
    //                     best_information_gain = current_information_gain;
    //                 }
    //             }
    //         }
    //     }
    //     best_split
    // }
}
