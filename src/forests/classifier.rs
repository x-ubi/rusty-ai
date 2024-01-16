use crate::data::dataset::{Dataset, Number, WholeNumber};
use crate::trees::classifier::DecisionTreeClassifier;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;

pub struct RandomForestClassifier<XT: Number, YT: WholeNumber> {
    trees: Vec<DecisionTreeClassifier<XT, YT>>,
    num_trees: usize,
    min_samples_split: u16,
    max_depth: Option<u16>,
    criterion: String,
    sample_size: Option<usize>,
}

impl<XT: Number, YT: WholeNumber> Default for RandomForestClassifier<XT, YT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<XT: Number, YT: WholeNumber> RandomForestClassifier<XT, YT> {
    pub fn new() -> Self {
        Self {
            trees: Vec::with_capacity(3),
            num_trees: 3,
            min_samples_split: 2,
            max_depth: None,
            sample_size: None,
            criterion: "gini".to_string(),
        }
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

        let seeds = (0..self.num_trees)
            .map(|_| rng.gen::<u64>())
            .collect::<Vec<_>>();

        match self.sample_size {
            // @TODO: Remove this after adding with_params()
            Some(sample_size) if sample_size > dataset.x.nrows() => {
                return Err("The sample size is greater than the dataset size.".into())
            }
            None => self.sample_size = Some(dataset.x.nrows() / self.num_trees),
            _ => {}
        }

        let trees: Result<Vec<_>, String> = seeds
            .into_par_iter()
            .map(|tree_seed| {
                let subset = dataset.samples(self.sample_size.unwrap(), Some(tree_seed));
                let mut tree = DecisionTreeClassifier::with_params(
                    Some(self.criterion.clone()),
                    Some(self.min_samples_split),
                    self.max_depth,
                )
                .map_err(|error| error.to_string())?;
                tree.fit(&subset).map_err(|error| error.to_string())?;
                Ok(tree)
            })
            .collect();
        self.trees = trees?;
        Ok("Finished building the trees".into())
    }

    pub fn predict(&self, features: &DMatrix<XT>) -> Result<DVector<YT>, Box<dyn Error>> {
        let mut predictions = DVector::from_element(features.nrows(), YT::from_u8(0).unwrap());

        for i in 0..features.nrows() {
            let mut class_counts = HashMap::new();
            for tree in &self.trees {
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
