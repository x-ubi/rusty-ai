// Decision Tree
use std::cmp::PartialEq;
use std::fmt::Debug;
use std::marker::PhantomData;
use nalgebra::{DMatrix, DVector};

pub trait Value: Debug + PartialEq + Clone {}
impl<T> Value for T where T: Debug + PartialEq + Clone {}

pub struct TreeNode<T: Value> {
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
    information_gain: Option<f64>,
    value: Option<T>,
}

impl<T: Value> TreeNode<T> {
    pub fn new(value: Option<T>) -> Self {
        Self {
            left: None,
            right: None,
            information_gain: None,
            value: value
        }
    }
}

pub struct DecisionTreeClassifier<XT: Value, YT:Value> {
    root: Option<Box<TreeNode<YT>>>,

    criterion: String,
    min_samples_split: u16,
    max_depth: Option<u16>,

    _marker: PhantomData<XT>,

}

impl<XT: Value, YT:Value> DecisionTreeClassifier<XT, YT> {
    pub fn new(criterion:Option<String>, min_samples_split: Option<u16>, max_depth: Option<u16>) -> Self {
        Self {
            root: None,
            criterion: criterion.unwrap_or("gini".to_string()),
            min_samples_split: min_samples_split.unwrap_or(2),
            max_depth: max_depth,
            _marker: PhantomData,
        }
    }

    pub fn build_tree(&mut self, X: DMatrix<XT>, y: DVector<YT>, current_depth:Option<u16>) -> () {
        let (num_samples, num_features) = X.shape();
        if (num_samples >= self.min_samples_split.into() && current_depth<= self.max_depth) {
            let best_split = self.get_best_split();

        }
    }

    pub fn get_best_split(&self) -> DVector<u16> { DVector::from_element(1, 1) }

}



pub fn main()
{
    println!("Hello, world!");
}