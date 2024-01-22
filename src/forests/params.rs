use std::error::Error;

/// Struct representing the parameters for a forest.
#[derive(Clone, Debug)]
pub struct ForestParams<T> {
    trees: Vec<T>,
    num_trees: usize,
    sample_size: Option<usize>,
}

impl<T> ForestParams<T> {
    /// Creates a new instance of Forest Params.
    ///
    /// This function initializes the params with default values:
    /// * `trees` - An empty vector of size 3,
    /// * `num_trees` - 3,
    /// * `sample_size` - None.
    ///
    /// # Returns
    ///
    /// A new instance of the Random Forest Classifier.
    pub fn new() -> Self {
        Self {
            trees: Vec::with_capacity(3),
            num_trees: 3,
            sample_size: None,
        }
    }

    /// Sets the trees for the forest.
    ///
    /// # Arguments
    ///
    /// * `trees` - The trees to set.
    pub fn set_trees(&mut self, trees: Vec<T>) {
        self.trees = trees;
    }

    /// Sets the number of trees in the forest.
    ///
    /// # Arguments
    ///
    /// * `num_trees` - The number of trees.
    ///
    /// # Returns
    ///
    /// Returns an error if the number of trees is less than 2.
    pub fn set_num_trees(&mut self, num_trees: usize) -> Result<(), Box<dyn Error>> {
        if num_trees < 2 {
            return Err("The number of trees must be greater than 1.".into());
        }
        self.num_trees = num_trees;
        self.trees = Vec::with_capacity(num_trees);
        Ok(())
    }

    /// Sets the sample size for the forest.
    ///
    /// # Arguments
    ///
    /// * `sample_size` - The sample size.
    ///
    /// # Returns
    ///
    /// Returns an error if the sample size is less than 1.
    pub fn set_sample_size(&mut self, sample_size: Option<usize>) -> Result<(), Box<dyn Error>> {
        if sample_size.is_some_and(|size| size < 1) {
            return Err("The sample size must be greater than 0.".into());
        }
        self.sample_size = sample_size;
        Ok(())
    }

    /// Returns a reference to the trees in the forest.
    pub fn trees(&self) -> &Vec<T> {
        &self.trees
    }

    /// Returns the number of trees in the forest.
    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    /// Returns the sample size for the forest.
    pub fn sample_size(&self) -> Option<usize> {
        self.sample_size
    }
}
