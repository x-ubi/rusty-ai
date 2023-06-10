use ndarray::Array2;
use ndarray_stats::QuantileExt;
use ndarray_stats::SummaryStatisticsExt;
use std::collections::HashMap;

struct GaussianNB {
    class_prior: HashMap<f32, f32>,
    class_count: HashMap<f32, f32>,
    theta: HashMap<f32, Vec<f32>>,
    sigma: HashMap<f32, Vec<f32>>,
}

impl GaussianNB {
    pub fn new() -> GaussianNB {
        GaussianNB {
            class_prior: HashMap::new(),
            class_count: HashMap::new(),
            theta: HashMap::new(),
            sigma: HashMap::new(),
        }
    }

    pub fn fit(&mut self, x: &Array2<f32>, y: &Array2<f32>) {
        let n_samples = x.nrows();
        let classes = y.unique();
        for class in classes.iter() {
            let mask = y.mapv(|yi| if yi == *class { true } else { false });
            let xi = x.select(Axis(0), mask);
            self.theta
                .insert(*class, xi.mean_axis(Axis(0)).unwrap().to_vec());
            self.sigma
                .insert(*class, xi.var_axis(Axis(0), 0.0).unwrap().to_vec());
            self.class_count
                .insert(*class, mask.iter().filter(|&&x| x).count() as f32);
        }
        for (class, count) in self.class_count.iter() {
            self.class_prior.insert(*class, *count / n_samples as f32);
        }
    }

    pub fn predict(&self, x: &Array2<f32>) -> Vec<f32> {
        let mut y_pred: Vec<f32> = Vec::new();
        for xi in x.outer_iter() {
            let mut max_log_prob: f32 = f32::MIN;
            let mut max_class = 0.0;
            for class in self.class_prior.keys() {
                let mut log_prob = self.class_prior.get(class).unwrap().ln();
                for feature in 0..xi.len() {
                    let mean = self.theta.get(class).unwrap()[feature];
                    let variance = self.sigma.get(class).unwrap()[feature];
                    log_prob += (-0.5 * ((xi[feature] - mean).powi(2) / variance).exp()
                        - 0.5 * variance.ln()
                        - 0.5 * std::f32::consts::LN_2_PI);
                }
                if log_prob > max_log_prob {
                    max_log_prob = log_prob;
                    max_class = *class;
                }
            }
            y_pred.push(max_class);
        }
        y_pred
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_fit() {
        let mut clf = GaussianNB::new();
        let x = arr2(&[[1.0, 2.0], [2.0, 1.0], [1.0, 0.0], [0.0, 1.0]]);
        let y = arr2(&[[0.0], [0.0], [1.0], [1.0]]);
        clf.fit(&x, &y);

        assert_relative_eq!(clf.class_prior[&0.0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(clf.class_prior[&1.0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_predict() {
        let mut clf = GaussianNB::new();
        let x_train = arr2(&[[1.0, 2.0], [2.0, 1.0], [1.0, 0.0], [0.0, 1.0]]);
        let y_train = arr2(&[[0.0], [0.0], [1.0], [1.0]]);
        clf.fit(&x_train, &y_train);

        let x_test = arr2(&[[1.5, 1.5], [0.5, 0.5]]);
        let y_pred = clf.predict(&x_test);

        assert_eq!(y_pred, vec![0.0, 1.0]);
    }
}
