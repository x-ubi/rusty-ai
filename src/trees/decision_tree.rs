// Decision Tree

use rand;

#[derive(Debug)]
pub struct TreeHyperparameters {
    dimension: usize,
    min_sizes_split: usize,
    max_depth: Option<u16>,
    rng_seed: Option<u64>
}

impl TreeHyperparameters {
    pub fn new(dimension: usize) -> TreeHyperparameters {
        TreeHyperparameters {
            dimension: dimension,
            min_sizes_split: 2,
            max_depth: Option::None,
            rng_seed: Option::None
        }
    }
}

pub struct TreeNode<T> {
    criterion: String,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    information_gain: f64,
    value: T,
}

impl<T> TreeNode<T> {
    pub fn new(criterion: Option<String>, value: T) -> TreeNode<T> {
        TreeNode {
            criterion: criterion.unwrap_or("gini".to_string())
            left: Option::None,
            right: Option::None,
            information_gain: 0.0,
            value: value
        }
    }
}

pub struct DecisionTreeClassifier {
    root: Option<Box<TreeNode>>

}

pub fn main()
{
    println!("Hello, world!");
}