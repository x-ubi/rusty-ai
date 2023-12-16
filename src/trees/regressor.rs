//! Decision Tree Regressor
use super::base::{DecisionTreeBase, SplitDataBase, TreeNode};
use crate::dataset::{Dataset, FeatureValue, TargetValue};
use nalgebra::{DMatrix, DVector};
use num_traits::ToPrimitive;
use std::f64::NEG_INFINITY;

pub struct SplitData<XT: FeatureValue, YT: TargetValue> {
    base: SplitDataBase<XT, YT>,
    information_gain: f64,
}

pub struct DecisionTreeRegressor<XT: FeatureValue, YT: TargetValue> {
    base: DecisionTreeBase<XT, YT>,
}

impl<XT: FeatureValue, YT: TargetValue> DecisionTreeRegressor<XT, YT> {
    pub fn new(min_samples_split: Option<u16>, max_depth: Option<u16>) -> Self {
        Self {
            base: DecisionTreeBase::new(min_samples_split, max_depth),
        }
    }

    pub fn fit(&mut self, dataset: &Dataset<XT, YT>) {
        self.base.root = Some(Box::new(
            self.build_tree(dataset, self.base.max_depth.map(|_| 0)),
        ));
    }

    pub fn predict(&self, features: &DMatrix<XT>) -> DVector<YT> {
        self.base.predict(features)
    }

    fn build_tree(
        &mut self,
        dataset: &Dataset<XT, YT>,
        current_depth: Option<u16>,
    ) -> TreeNode<XT, YT> {
        let (x, y) = &dataset.into_parts();
        let (num_samples, num_features) = x.shape();
        if num_samples >= self.base.min_samples_split.into() && current_depth <= self.base.max_depth
        {
            let best_split = self.get_best_split(&dataset, num_features).unwrap();
            let left_child = best_split.base.left;
            let right_child = best_split.base.right;
            if best_split.information_gain > 0.0 {
                let new_depth = match current_depth {
                    Some(depth) => Some(depth + 1),
                    _ => None,
                };
                let left_node = self.build_tree(&left_child, new_depth);
                let right_node = self.build_tree(&right_child, new_depth);
                return TreeNode {
                    feature_index: Some(best_split.base.feature_index),
                    threshold: Some(best_split.base.threshold),
                    left: Some(Box::new(left_node)),
                    right: Some(Box::new(right_node)),
                    value: None,
                };
            }
        }

        let leaf_value = self.mean(&y);
        TreeNode::new(Some(leaf_value))
    }

    fn get_best_split(
        &self,
        dataset: &Dataset<XT, YT>,
        num_features: usize,
    ) -> Option<SplitData<XT, YT>> {
        let mut best_split: Option<SplitData<XT, YT>> = None;
        let mut best_information_gain = NEG_INFINITY;

        for feature_index in 0..num_features {
            let mut unique_values: Vec<_> =
                dataset.x.column(feature_index).iter().cloned().collect();
            unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            unique_values.dedup();

            for value in &unique_values {
                let (left_child, right_child) = dataset.split(feature_index, value.clone());

                if left_child.is_not_empty() && right_child.is_not_empty() {
                    let current_information_gain = self.calculate_variance_reduction(
                        dataset.y.clone(),
                        left_child.y.clone(),
                        right_child.y.clone(),
                    );

                    if current_information_gain > best_information_gain {
                        best_split = Some(SplitData {
                            base: SplitDataBase {
                                feature_index: feature_index,
                                threshold: value.clone(),
                                left: left_child,
                                right: right_child,
                            },
                            information_gain: current_information_gain,
                        });
                        best_information_gain = current_information_gain;
                    }
                }
            }
        }
        best_split
    }

    fn calculate_variance_reduction(
        &self,
        parent_y: DVector<YT>,
        left_y: DVector<YT>,
        right_y: DVector<YT>,
    ) -> f64 {
        let variance = self.variance(&parent_y);
        let left_variance = self.variance(&left_y);
        let right_variance = self.variance(&right_y);
        let num_samples = parent_y.len() as f64;
        variance
            - (left_variance * (left_y.len() as f64) / num_samples)
            - (right_variance * (right_y.len() as f64) / num_samples)
    }

    fn variance(&self, y: &DVector<YT>) -> f64 {
        let mean = self.mean(y);
        let variance = y.iter().fold(YT::from_f64(0.0).unwrap(), |acc, x| {
            acc + (*x - mean) * (*x - mean)
        });
        let variance_f64 = ToPrimitive::to_f64(&variance).unwrap();
        variance_f64 / y.len() as f64
    }

    fn mean(&self, y: &DVector<YT>) -> YT {
        let zero = YT::from_f64(0.0).unwrap();
        let sum: YT = y.iter().fold(zero, |acc, x| acc + *x);
        sum / YT::from_usize(y.len()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_mean() {
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let regressor: DecisionTreeRegressor<f64, f64> = DecisionTreeRegressor::new(None, None);
        let mean = regressor.mean(&y);
        assert_eq!(mean, 3.5);
    }

    #[test]
    fn test_variance() {
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let regressor: DecisionTreeRegressor<f64, f64> = DecisionTreeRegressor::new(None, None);
        let variance = regressor.variance(&y);
        assert_eq!(variance, 2.0);
    }

    #[test]
    fn test_calculate_variance_reduction() {
        let parent_y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let left_y = DVector::from_vec(vec![1.0, 2.0]);
        let right_y = DVector::from_vec(vec![3.0, 4.0, 5.0]);
        let regressor: DecisionTreeRegressor<f64, f64> = DecisionTreeRegressor::new(None, None);
        let variance_reduction = regressor.calculate_variance_reduction(parent_y, left_y, right_y);
        assert!(variance_reduction > 0.0);
    }

    #[test]
    fn test_fit_and_predict() {
        let x = DMatrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = DVector::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
        let dataset = Dataset::new(x, y);
        let mut regressor: DecisionTreeRegressor<f64, f64> = DecisionTreeRegressor::new(None, None);
        regressor.fit(&dataset);

        let test_x = DMatrix::from_vec(3, 1, vec![2.0, 3.0, 4.0]);
        let predictions = regressor.predict(&test_x);

        assert_eq!(predictions.len(), 3);
        assert!(predictions.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_fit_and_predict_with_multiple_rows() {
        let x = DMatrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = DVector::from_vec(vec![1.0, 4.0, 9.0]);
        let dataset = Dataset::new(x, y);
        let mut regressor: DecisionTreeRegressor<f64, f64> = DecisionTreeRegressor::new(None, None);
        regressor.fit(&dataset);

        let test_x = DMatrix::from_vec(3, 2, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        let predictions = regressor.predict(&test_x);

        assert_eq!(predictions.len(), 3);
        assert!(predictions.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_fit_and_predict_with_single_row() {
        let x = DMatrix::from_vec(1, 2, vec![1.0, 2.0]);
        let y = DVector::from_vec(vec![1.0]);
        let dataset = Dataset::new(x, y);
        let mut regressor: DecisionTreeRegressor<f64, f64> = DecisionTreeRegressor::new(None, None);
        regressor.fit(&dataset);

        let test_x = DMatrix::from_vec(1, 2, vec![2.0, 3.0]);
        let predictions = regressor.predict(&test_x);

        assert_eq!(predictions.len(), 1);
        assert!(predictions.iter().all(|&x| x >= 0.0));
    }
}
