// Decision Tree
use std::fmt::Debug;
use std::cmp::PartialEq;
use nalgebra::{DMatrix, DVector};

pub trait Value: Debug + PartialEq + Clone {}
impl<T> Value for T where T: Debug + PartialEq + Clone {}

pub struct TreeNode<T: Value> {
    criterion: String,
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
    information_gain: f64,
    value: T,
}

impl<T: Value> TreeNode<T> {
    pub fn new(criterion: Option<String>, value: T) -> Self {
        Self {
            criterion: criterion.unwrap_or("gini".to_string()),
            left: Option::None,
            right: Option::None,
            information_gain: 0.0,
            value: value
        }
    }
}

pub struct DecisionTreeClassifier<T: Value> {
    root: Option<Box<TreeNode<T>>>,

    min_samples_split: u16,
    max_depth: Option<u16>,

}

impl<T: Value> DecisionTreeClassifier<T> {
    pub fn new(min_samples_split: Option<u16>, max_depth: Option<u16>) -> Self {
        Self {
            root: None,
            min_samples_split: min_samples_split.unwrap_or(2),
            max_depth: max_depth
        }
    }

    pub fn fit<XType:Value, YType:Value>(X: Vec<XType>, y: Vec<YType>) {

    }
}



pub fn main()
{
    println!("Hello, world!");
}