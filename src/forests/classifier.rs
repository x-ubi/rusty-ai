use crate::data::dataset::{Dataset, Number, WholeNumber};
use crate::trees::classifier::DecisionTreeClassifier;
use crate::trees::params::TreeClassifierParams;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;

use super::params::ForestParams;

#[derive(Clone, Debug)]
pub struct RandomForestClassifier<XT: Number, YT: WholeNumber> {
    forest_params: ForestParams<DecisionTreeClassifier<XT, YT>>,
    tree_params: TreeClassifierParams,
}

impl<XT: Number, YT: WholeNumber> Default for RandomForestClassifier<XT, YT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<XT: Number, YT: WholeNumber> RandomForestClassifier<XT, YT> {
    pub fn new() -> Self {
        Self {
            forest_params: ForestParams::new(),
            tree_params: TreeClassifierParams::new(),
        }
    }

    pub fn with_params(
        num_trees: Option<usize>,
        min_samples_split: Option<u16>,
        max_depth: Option<u16>,
        criterion: Option<String>,
        sample_size: Option<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut forest = Self::new();

        forest.set_num_trees(num_trees.unwrap_or(3))?;
        forest.set_sample_size(sample_size)?;
        forest.set_min_samples_split(min_samples_split.unwrap_or(2))?;
        forest.set_max_depth(max_depth)?;
        forest.set_criterion(criterion.unwrap_or("gini".to_string()))?;
        Ok(forest)
    }

    pub fn set_trees(&mut self, trees: Vec<DecisionTreeClassifier<XT, YT>>) {
        self.forest_params.set_trees(trees);
    }

    pub fn set_num_trees(&mut self, num_trees: usize) -> Result<(), Box<dyn Error>> {
        self.forest_params.set_num_trees(num_trees)
    }

    pub fn set_sample_size(&mut self, sample_size: Option<usize>) -> Result<(), Box<dyn Error>> {
        self.forest_params.set_sample_size(sample_size)
    }

    pub fn set_min_samples_split(&mut self, min_samples_split: u16) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_min_samples_split(min_samples_split)
    }

    pub fn set_max_depth(&mut self, max_depth: Option<u16>) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_max_depth(max_depth)
    }

    pub fn set_criterion(&mut self, criterion: String) -> Result<(), Box<dyn Error>> {
        self.tree_params.set_criterion(criterion)
    }

    pub fn trees(&self) -> &Vec<DecisionTreeClassifier<XT, YT>> {
        self.forest_params.trees()
    }

    pub fn num_trees(&self) -> usize {
        self.forest_params.num_trees()
    }

    pub fn sample_size(&self) -> Option<usize> {
        self.forest_params.sample_size()
    }

    pub fn min_samples_split(&self) -> u16 {
        self.tree_params.min_samples_split()
    }

    pub fn max_depth(&self) -> Option<u16> {
        self.tree_params.max_depth()
    }

    pub fn criterion(&self) -> &String {
        &self.tree_params.criterion
    }

    pub fn fit(
        &mut self,
        dataset: &Dataset<XT, YT>,
        seed: Option<u64>,
    ) -> Result<String, Box<dyn Error>> {
        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            _ => StdRng::from_entropy(),
        };

        let seeds = (0..self.num_trees())
            .map(|_| rng.gen::<u64>())
            .collect::<Vec<_>>();

        match self.sample_size() {
            Some(sample_size) if sample_size > dataset.nrows() => {
                return Err(format!(
                    "The set sample size is greater than the dataset size. {} > {}",
                    sample_size,
                    dataset.nrows()
                )
                .into());
            }
            None => self.set_sample_size(Some(dataset.nrows() / self.num_trees()))?,
            _ => {}
        }

        let trees: Result<Vec<_>, String> = seeds
            .into_par_iter()
            .map(|tree_seed| {
                let subset = dataset.samples(self.sample_size().unwrap(), Some(tree_seed));
                let mut tree = DecisionTreeClassifier::with_params(
                    Some(self.criterion().clone()),
                    Some(self.min_samples_split()),
                    self.max_depth(),
                )
                .map_err(|error| error.to_string())?;
                tree.fit(&subset).map_err(|error| error.to_string())?;
                Ok(tree)
            })
            .collect();
        self.set_trees(trees?);
        Ok("Finished building the trees".into())
    }

    pub fn predict(&self, features: &DMatrix<XT>) -> Result<DVector<YT>, Box<dyn Error>> {
        let mut predictions = DVector::from_element(features.nrows(), YT::from_u8(0).unwrap());

        for i in 0..features.nrows() {
            let mut class_counts = HashMap::new();
            for tree in self.trees() {
                let prediction = tree
                    .predict(&DMatrix::from_row_slice(
                        1,
                        features.ncols(),
                        features.row(i).transpose().as_slice(),
                    ))
                    .map_err(|error| error.to_string())?;
                *class_counts.entry(prediction[0]).or_insert(0) += 1;
            }

            let chosen_class = class_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(class, _)| class)
                .ok_or(
                    "Prediction failure. No trees built or class counts are empty.".to_string(),
                )?;
            predictions[i] = chosen_class;
        }
        Ok(predictions)
    }
}
