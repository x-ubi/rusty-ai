use std::error::Error;

#[derive(Clone, Debug)]
pub struct ForestParams<T> {
    trees: Vec<T>,
    num_trees: usize,
    sample_size: Option<usize>,
}

impl<T> ForestParams<T> {
    pub fn new() -> Self {
        Self {
            trees: Vec::with_capacity(3),
            num_trees: 3,
            sample_size: None,
        }
    }

    pub fn set_trees(&mut self, trees: Vec<T>) {
        self.trees = trees;
    }

    pub fn set_num_trees(&mut self, num_trees: usize) -> Result<(), Box<dyn Error>> {
        if num_trees < 2 {
            return Err("The number of trees must be greater than 1.".into());
        }
        self.num_trees = num_trees;
        self.trees = Vec::with_capacity(num_trees);
        Ok(())
    }

    pub fn set_sample_size(&mut self, sample_size: Option<usize>) -> Result<(), Box<dyn Error>> {
        if sample_size.is_some_and(|size| size < 1) {
            return Err("The sample size must be greater than 0.".into());
        }
        self.sample_size = sample_size;
        Ok(())
    }

    pub fn trees(&self) -> &Vec<T> {
        &self.trees
    }

    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    pub fn sample_size(&self) -> Option<usize> {
        self.sample_size
    }
}
