// Decision Tree
use nalgebra::{DMatrix, DVector};
use std::cmp::{Eq, PartialOrd};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

pub trait NodeValue: Debug + Clone {}
impl<T> NodeValue for T where T: Debug + Clone {}

pub trait FeatureValue: NodeValue + PartialOrd {}
impl<T> FeatureValue for T where T: NodeValue + PartialOrd {}

pub trait TargetValue: NodeValue + Eq + Hash {}
impl<T> TargetValue for T where T: NodeValue + Eq + Hash {}

pub struct Dataset<XT: FeatureValue, YT: TargetValue> {
    x: DMatrix<XT>,
    y: DVector<YT>,
}

impl<XT: FeatureValue, YT: TargetValue> Dataset<XT, YT> {
    pub fn new(x: DMatrix<XT>, y: DVector<YT>) -> Self {
        Self { x: x, y: y }
    }

    pub fn into_parts(&self) -> (&DMatrix<XT>, &DVector<YT>) {
        (&self.x, &self.y)
    }
}

pub struct TreeNode<YT: TargetValue> {
    feature_index: Option<usize>,
    left: Option<Box<TreeNode<YT>>>,
    right: Option<Box<TreeNode<YT>>>,
    information_gain: Option<f64>,
    value: Option<YT>,
}

impl<YT: TargetValue> TreeNode<YT> {
    pub fn new(value: Option<YT>) -> Self {
        Self {
            feature_index: None,
            left: None,
            right: None,
            information_gain: None,
            value: value,
        }
    }

    pub fn len(&self) -> usize {
        todo!()
    }
}

pub struct DecisionTreeClassifier<XT: FeatureValue, YT: TargetValue> {
    root: Option<Box<TreeNode<YT>>>,

    criterion: String,
    min_samples_split: u16,
    max_depth: Option<u16>,

    _marker: PhantomData<XT>,
}

impl<XT: FeatureValue, YT: TargetValue> DecisionTreeClassifier<XT, YT> {
    pub fn new(
        criterion: Option<String>,
        min_samples_split: Option<u16>,
        max_depth: Option<u16>,
    ) -> Self {
        Self {
            root: None,
            criterion: criterion.unwrap_or("gini".to_string()),
            min_samples_split: min_samples_split.unwrap_or(2),
            max_depth: max_depth,
            _marker: PhantomData,
        }
    }

    pub fn build_tree(&mut self, dataset: Dataset<XT, YT>, current_depth: Option<u16>) -> () {
        let (x, y) = dataset.into_parts();
        let (num_samples, num_features) = x.shape();
        if (num_samples >= self.min_samples_split.into() && current_depth <= self.max_depth) {
            let best_split = self.get_best_split();
        }
    }

    pub fn get_best_split(&self) -> DVector<u16> {
        todo!()
    }

    pub fn split(&self) -> (Dataset<XT, YT>, Dataset<XT, YT>) {
        todo!()
    }

    fn calculate_information_gain(&self, parent_node: TreeNode<YT>, left_child: TreeNode<YT>, right_child: TreeNode<YT>) -> f64 {
        let weight_left = left_child.len() as f64 / parent_node.len() as f64;
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

    fn leaf_value(&self, y: DVector<YT>) -> Option<YT> {
        let mut class_counts = HashMap::new();
        for item in y.iter() {
            *class_counts.entry(item).or_insert(0) += 1;
        }
        class_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(val, _)| val.clone())
    }
}

pub fn main() {
    println!("Hello, world!");
}
