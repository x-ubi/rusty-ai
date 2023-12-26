// Base Decision Tree
use crate::dataset::{FeatureValue, TargetValue};

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

            value,
        }
    }
}
