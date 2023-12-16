// Base Decision Tree
use crate::dataset::{Dataset, FeatureValue, TargetValue};
use nalgebra::{DMatrix, DVector};
use std::marker::PhantomData;

pub struct SplitDataBase<XT: FeatureValue, YT: TargetValue> {
    pub feature_index: usize,
    pub threshold: XT,
    pub left: Dataset<XT, YT>,
    pub right: Dataset<XT, YT>,
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
    pub fn new(min_samples_split: Option<u16>, max_depth: Option<u16>) -> Self {
        Self {
            root: None,
            min_samples_split: min_samples_split.unwrap_or(2),
            max_depth: max_depth,
            _marker: PhantomData,
        }
    }

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

    pub fn predict(&self, prediction_features: &DMatrix<XT>) -> DVector<YT> {
        let predictions: Vec<_> = prediction_features
            .row_iter()
            .map(|row| self.make_prediction(row.transpose(), self.root.as_ref().unwrap()))
            .collect();

        DVector::from_vec(predictions)
    }
}
