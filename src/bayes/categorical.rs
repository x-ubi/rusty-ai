use crate::data::dataset::{Dataset, WholeNumber};
use nalgebra::{DMatrix, DVector};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
};

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

    pub fn feature_class_freq(&self) -> &HashMap<T, DVector<HashMap<T, f64>>> {
        &self.feature_class_freq
    }

    pub fn label_class_freq(&self) -> &HashMap<T, f64> {
        &self.label_class_freq
    }

    pub fn fit(&mut self, dataset: &Dataset<T, T>) -> Result<String, Box<dyn Error>> {
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

        Ok("Finished fitting".into())
    }

    fn predict_single(&self, x: &DVector<T>) -> Result<T, Box<dyn Error>> {
        let mut max_prob = f64::NEG_INFINITY;
        let mut max_class = T::from_i8(0).unwrap();

        for (y_class, label_freq) in &self.label_class_freq {
            let mut prob = label_freq.ln();
            for (idx, feature) in x.iter().enumerate() {
                prob += self
                    .feature_class_freq
                    .get(y_class)
                    .ok_or(format!("Class {:?} wasn't obtained.", y_class))?[idx]
                    .get(feature)
                    .ok_or(format!(
                        "Class {:?} frequency of feature {:?} wasn't obtained.",
                        feature, idx
                    ))?
                    .ln();
            }

            if prob > max_prob {
                max_prob = prob;
                max_class = *y_class;
            }
        }

        Ok(max_class)
    }

    pub fn predict(&self, x: &DMatrix<T>) -> Result<DVector<T>, Box<dyn Error>> {
        let mut y_pred = Vec::new();

        for i in 0..x.nrows() {
            let x_row = x.row(i).transpose();
            let y_class = self.predict_single(&x_row)?;
            y_pred.push(y_class);
        }
        Ok(DVector::from_vec(y_pred))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_new() {
        let model = CategoricalNB::<i32>::new();

        assert!(model.feature_class_freq.is_empty());
        assert!(model.label_class_freq.is_empty());
    }

    #[test]
    fn test_fit() {
        let mut model = CategoricalNB::<i32>::new();

        let x = DMatrix::from_row_slice(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let y = DVector::from_vec(vec![1, 2, 3]);
        let dataset = Dataset::new(x, y);

        let result = model.fit(&dataset);

        assert!(result.is_ok());
        assert_eq!(model.label_class_freq.len(), 3);
        assert_eq!(model.feature_class_freq.len(), 3);
    }

    // #[test]
    // fn test_predict() {
    //     let mut model = CategoricalNB::<i32>::new();

    //     let x = DMatrix::from_row_slice(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    //     let y = DVector::from_vec(vec![1, 2, 3]);
    //     let dataset = Dataset::new(x.clone(), y.clone());

    //     model.fit(&dataset).unwrap();

    //     let result = model.predict(&x);
    //     println!("{:?}", result);
    //     //assert!(result.is_ok());
    //     //assert_eq!(result.unwrap(), y);
    // }
}
