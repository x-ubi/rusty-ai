use std::error::Error;

use nalgebra::{DMatrix, DVector};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::{
    data::dataset::{Dataset, RealNumber},
    metrics::errors::RegressionMetrics,
    trees::regressor::DecisionTreeRegressor,
};

pub struct RandomForestRegressor<T: RealNumber> {
    trees: Vec<DecisionTreeRegressor<T>>,
    num_trees: usize,
    min_samples_split: u16,
    max_depth: Option<u16>,
    sample_size: Option<usize>,
}

impl<T: RealNumber> Default for RandomForestRegressor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealNumber> RegressionMetrics<T> for RandomForestRegressor<T> {}

impl<T: RealNumber> RandomForestRegressor<T> {
    pub fn new() -> Self {
        Self {
            trees: Vec::with_capacity(2),
            num_trees: 2,
            min_samples_split: 2,
            max_depth: None,
            sample_size: None,
        }
    }

    pub fn fit(
        &mut self,
        dataset: &Dataset<T, T>,
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
                let mut tree = DecisionTreeRegressor::with_params(
                    Some(self.min_samples_split),
                    self.max_depth,
                );
                tree.fit(&subset).map_err(|error| error.to_string())?;
                Ok(tree)
            })
            .collect();
        self.trees = trees?;
        Ok("Finished building the trees.".into())
    }

    pub fn predict(&self, features: &DMatrix<T>) -> Result<DVector<T>, Box<dyn Error>> {
        let mut predictions = DVector::from_element(features.nrows(), T::from_f64(0.0).unwrap());

        for i in 0..features.nrows() {
            let mut total_prediction = T::from_f64(0.0).unwrap();
            for tree in &self.trees {
                let prediction = tree.predict(&DMatrix::from_row_slice(
                    1,
                    features.ncols(),
                    features.row(i).transpose().as_slice(),
                ))?;
                total_prediction = total_prediction + prediction[0];
            }

            predictions[i] = total_prediction / T::from_usize(self.trees.len()).unwrap();
        }
        Ok(predictions)
    }
}
