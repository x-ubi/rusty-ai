use std::error::Error;

#[derive(Clone, Debug)]
pub struct TreeParams {
    pub min_samples_split: u16,
    pub max_depth: Option<u16>,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeParams {
    pub fn new() -> Self {
        Self {
            min_samples_split: 2,
            max_depth: None,
        }
    }

    pub fn set_min_samples_split(&mut self, min_samples_split: u16) -> Result<(), Box<dyn Error>> {
        if min_samples_split < 2 {
            return Err("The minimum number of samples to split must be greater than 1.".into());
        }
        self.min_samples_split = min_samples_split;
        Ok(())
    }

    pub fn set_max_depth(&mut self, max_depth: Option<u16>) -> Result<(), Box<dyn Error>> {
        if max_depth.is_some_and(|depth| depth < 1) {
            return Err("The maximum depth must be greater than 0.".into());
        }
        self.max_depth = max_depth;
        Ok(())
    }

    pub fn min_samples_split(&self) -> u16 {
        self.min_samples_split
    }

    pub fn max_depth(&self) -> Option<u16> {
        self.max_depth
    }
}

#[derive(Clone, Debug)]
pub struct TreeClassifierParams {
    pub base_params: TreeParams,
    pub criterion: String,
}

impl Default for TreeClassifierParams {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeClassifierParams {
    pub fn new() -> Self {
        Self {
            base_params: TreeParams::new(),
            criterion: "gini".to_string(),
        }
    }

    pub fn set_min_samples_split(&mut self, min_samples_split: u16) -> Result<(), Box<dyn Error>> {
        self.base_params.set_min_samples_split(min_samples_split)
    }

    pub fn set_max_depth(&mut self, max_depth: Option<u16>) -> Result<(), Box<dyn Error>> {
        self.base_params.set_max_depth(max_depth)
    }

    pub fn set_criterion(&mut self, criterion: String) -> Result<(), Box<dyn Error>> {
        if !["gini", "entropy"].contains(&criterion.as_str()) {
            return Err("The criterion must be either 'gini' or 'entropy'.".into());
        }
        self.criterion = criterion;
        Ok(())
    }

    pub fn min_samples_split(&self) -> u16 {
        self.base_params.min_samples_split
    }

    pub fn max_depth(&self) -> Option<u16> {
        self.base_params.max_depth
    }

    pub fn criterion(&self) -> &str {
        &self.criterion
    }
}
