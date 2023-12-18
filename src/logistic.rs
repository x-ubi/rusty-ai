use crate::dataset::{Dataset, RealNumber};
use nalgebra::{DMatrix, DVector};

pub struct LogisticRegression<T: RealNumber> {
    weights: DVector<T>,
}

impl<T: RealNumber> LogisticRegression<T> {
    pub fn new(
        dimension: Option<usize>,
        weights: Option<DVector<T>>,
    ) -> Result<Self, &'static str> {
        match (dimension, &weights) {
            (None, None) => Err("Please input the dimension or starting weights."),

            (Some(dim), Some(w)) if dim != w.len() => {
                Err("The dimension isn't equal the amount of weights.")
            }
            _ => Ok(Self {
                weights: weights.unwrap_or_else(|| {
                    DVector::<T>::from_element(dimension.unwrap(), T::from_f64(0.0).unwrap())
                }),
            }),
        }
    }

    pub fn fit(&mut self, dataset: Dataset<T, T>, lr: T, mut max_steps: usize, epsilon: Option<T>) {
        let (x, y) = dataset.into_parts();
        while max_steps > 0 {
            let weights_prev = self.weights.clone();
            let grad = self.gradient(x, y);
            self.weights = DVector::from_iterator(
                self.weights.len(),
                weights_prev
                    .iter()
                    .zip(grad.iter())
                    .map(|(&w, &g)| w - lr * g),
            );
            if self
                .weights
                .iter()
                .zip(weights_prev.iter())
                .map(|(&w, &w_prev)| (w - w_prev) * (w - w_prev))
                .fold(T::from_f64(0.0).unwrap(), |acc, x| acc + x)
                < epsilon.unwrap_or_else(|| T::from_f64(1e-6).unwrap())
            {
                return;
            }
            max_steps -= 1;
        }
    }

    fn gradient(&self, x: &DMatrix<T>, y: &DVector<T>) -> DVector<T> {
        let y_pred = self.h(x);

        let x_with_bias = x.clone().insert_column(0, T::from_f64(1.0).unwrap());

        x_with_bias
            .column_iter()
            .zip(y.iter())
            .zip(y_pred.iter())
            .map(|((x_i, &y_i), &y_pred_i)| x_i * (y_i - y_pred_i))
            .fold(DVector::zeros(self.weights.len()), |acc, v| acc + v)
    }

    fn cross_entropy(&self, x: &DMatrix<T>, y: &DVector<T>) -> T {
        let y_pred: DVector<T> = self.h(x);
        let one = T::from_f64(1.0).unwrap();

        y.iter()
            .zip(y_pred.iter())
            .map(|(&y_i, &y_pred_i)| -y_i * y_pred_i.ln() + (one - y_i) * (one - y_pred_i).ln())
            .fold(T::from_f64(0.0).unwrap(), |acc, x| acc + x)
    }

    fn h(&self, features: &DMatrix<T>) -> DVector<T> {
        let features_with_bias = features.clone().insert_column(0, T::from_f64(1.0).unwrap());
        let z = features_with_bias * &self.weights;

        z.map(|val| Self::sigmoid(val))
    }

    fn sigmoid(z: T) -> T {
        let one = T::from_f64(1.0).unwrap();

        one / (one + (-z).exp())
    }
}
