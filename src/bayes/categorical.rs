use crate::dataset::{Dataset, WholeNumber};
use nalgebra::DVector;
use std::collections::{HashMap, HashSet};

pub struct CategoricalNB<T: WholeNumber> {
    feature_class_freq: HashMap<T, DVector<HashMap<T, f64>>>,
    label_class_freq: HashMap<T, f64>,
}

impl<T: WholeNumber> CategoricalNB<T> {
    pub fn new() -> Self {
        Self {
            feature_class_freq: HashMap::new(),
            label_class_freq: HashMap::new(),
        }
    }

    pub fn fit(&mut self, dataset: &Dataset<T, T>) {
        let (x, y) = dataset.into_parts();
        let y_classes = y.iter().cloned().collect::<HashSet<_>>();

        for y_class in y_classes {
            let class_mask = y.map(|label| label == y_class);
            let class_indices = class_mask
                .iter()
                .enumerate()
                .filter(|&(_, &value)| value)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();

            let x_y_class = x.select_rows(class_indices.as_slice());

            let mut all_features_freq = DVector::from_element(x.ncols(), HashMap::new());
            for (idx, feature) in x_y_class.column_iter().enumerate() {
                let feature_count = feature.iter().fold(HashMap::new(), |mut acc, &val| {
                    *acc.entry(val).or_insert(0) += 1;
                    acc
                });
                let feature_freq = feature_count
                    .into_iter()
                    .map(|(class, count)| (class, count as f64 / x.ncols() as f64))
                    .collect();
                all_features_freq[idx] = feature_freq;
            }

            let label_class_freq = class_indices.len() as f64 / y.nrows() as f64;

            self.label_class_freq.insert(y_class, label_class_freq);
            self.feature_class_freq.insert(y_class, all_features_freq);
        }
    }
}
