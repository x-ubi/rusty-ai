use std::error::Error;

/// Struct representing the parameters for a decision tree.
#[derive(Clone, Debug)]
pub struct TreeParams {
    pub min_samples_split: u16,
    pub max_depth: Option<u16>,
}

impl Default for TreeParams {
    /// Creates a new instance of `TreeParams` with default values.
    fn default() -> Self {
        Self::new()
    }
}

impl TreeParams {
    /// Creates a new instance of `TreeParams` with default values.
    pub fn new() -> Self {
        Self {
            min_samples_split: 2,
            max_depth: None,
        }
    }

    /// Sets the minimum number of samples required to split a node.
    ///
    /// # Arguments
    ///
    /// * `min_samples_split` - The minimum number of samples to split.
    ///
    /// # Errors
    ///
    /// Returns an error if `min_samples_split` is less than 2.
    pub fn set_min_samples_split(&mut self, min_samples_split: u16) -> Result<(), Box<dyn Error>> {
        if min_samples_split < 2 {
            return Err("The minimum number of samples to split must be greater than 1.".into());
        }
        self.min_samples_split = min_samples_split;
        Ok(())
    }

    /// Sets the maximum depth of the decision tree.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - The maximum depth of the tree.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_depth` is less than 1.
    pub fn set_max_depth(&mut self, max_depth: Option<u16>) -> Result<(), Box<dyn Error>> {
        if max_depth.is_some_and(|depth| depth < 1) {
            return Err("The maximum depth must be greater than 0.".into());
        }
        self.max_depth = max_depth;
        Ok(())
    }

    /// Returns the minimum number of samples required to split a node.
    pub fn min_samples_split(&self) -> u16 {
        self.min_samples_split
    }

    /// Returns the maximum depth of the decision tree.
    pub fn max_depth(&self) -> Option<u16> {
        self.max_depth
    }
}

/// Struct representing the parameters for a decision tree classifier.
#[derive(Clone, Debug)]
pub struct TreeClassifierParams {
    pub base_params: TreeParams,
    pub criterion: String,
}

impl Default for TreeClassifierParams {
    /// Creates a new instance of `TreeClassifierParams` with default values.
    fn default() -> Self {
        Self::new()
    }
}

impl TreeClassifierParams {
    /// Creates a new instance of `TreeClassifierParams` with default values.
    pub fn new() -> Self {
        Self {
            base_params: TreeParams::new(),
            criterion: "gini".to_string(),
        }
    }

    /// Sets the minimum number of samples required to split a node.
    ///
    /// # Arguments
    ///
    /// * `min_samples_split` - The minimum number of samples to split.
    ///
    /// # Errors
    ///
    /// Returns an error if `min_samples_split` is less than 2.
    pub fn set_min_samples_split(&mut self, min_samples_split: u16) -> Result<(), Box<dyn Error>> {
        self.base_params.set_min_samples_split(min_samples_split)
    }

    /// Sets the maximum depth of the decision tree.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - The maximum depth of the tree.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_depth` is less than 1.
    pub fn set_max_depth(&mut self, max_depth: Option<u16>) -> Result<(), Box<dyn Error>> {
        self.base_params.set_max_depth(max_depth)
    }

    /// Sets the criterion used for splitting nodes in the decision tree.
    ///
    /// # Arguments
    ///
    /// * `criterion` - The criterion for splitting nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if `criterion` is not "gini" or "entropy".
    pub fn set_criterion(&mut self, criterion: String) -> Result<(), Box<dyn Error>> {
        if !["gini", "entropy"].contains(&criterion.as_str()) {
            return Err("The criterion must be either 'gini' or 'entropy'.".into());
        }
        self.criterion = criterion;
        Ok(())
    }

    /// Returns the minimum number of samples required to split a node.
    pub fn min_samples_split(&self) -> u16 {
        self.base_params.min_samples_split
    }

    /// Returns the maximum depth of the decision tree.
    pub fn max_depth(&self) -> Option<u16> {
        self.base_params.max_depth
    }

    /// Returns the criterion used for splitting nodes in the decision tree.
    pub fn criterion(&self) -> &str {
        &self.criterion
    }
}
