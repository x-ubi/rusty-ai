use std::collections::HashMap;

use crate::dataset::{Dataset, Number, WholeNumber};
use crate::trees::classifier::DecisionTreeClassifier;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

pub struct RandomForestClassifier<XT: Number + Send + Sync, YT: WholeNumber + Send + Sync> {
    trees: Vec<DecisionTreeClassifier<XT, YT>>,
    num_trees: usize,
    min_samples_split: u16,
    max_depth: Option<u16>,
    criterion: String,
    sample_size: usize,
}

impl<XT: Number + Send + Sync, YT: WholeNumber + Send + Sync> RandomForestClassifier<XT, YT> {
    pub fn new() -> Self {
        Self {
            trees: Vec::with_capacity(3),
            num_trees: 3,
            min_samples_split: 2,
            max_depth: None,
            sample_size: 1000,
            criterion: "gini".to_string(),
        }
    }
    pub fn fit(&mut self, dataset: Dataset<XT, YT>, seed: Option<u64>) {
        self.trees = (0..self.num_trees)
            .into_par_iter()
            .map(|_| {
                let subset = dataset.samples(self.sample_size, seed);
                let mut tree = DecisionTreeClassifier::with_params(
                    Some(self.criterion.clone()),
                    Some(self.min_samples_split),
                    self.max_depth,
                );
                tree.fit(&subset);
                tree
            })
            .collect::<Vec<_>>();
    }

    pub fn predict(&self, features: &DMatrix<XT>) -> DVector<YT> {
        let mut predictions = DVector::from_element(features.nrows(), YT::from_u8(0).unwrap());

        for i in 0..features.nrows() {
            let mut class_counts = HashMap::new();
            for tree in &self.trees {
                let prediction = tree.predict(&DMatrix::from_row_slice(
                    1,
                    features.ncols(),
                    &features.row(i).transpose().as_slice(),
                ));
                *class_counts.entry(prediction[0]).or_insert(0) += 1;
            }

            let chosen_class = class_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(class, _)| class)
                .unwrap();
            predictions[i] = chosen_class;
        }
        predictions
    }
}
