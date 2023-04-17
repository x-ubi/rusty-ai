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