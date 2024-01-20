use crate::{
    data::dataset::{Dataset, RealNumber},
    metrics::errors::RegressionMetrics,
};
use nalgebra::{DMatrix, DVector};
use std::error::Error;

pub struct LinearRegression<T: RealNumber> {
    weights: DVector<T>,
}

impl<T: RealNumber> RegressionMetrics<T> for LinearRegression<T> {}

impl<T: RealNumber> LinearRegression<T> {
    pub fn new(
        dimension: Option<usize>,
        weights: Option<DVector<T>>,
    ) -> Result<Self, Box<dyn Error>> {
        match (dimension, &weights) {
            (None, None) => Err("Please input the dimension or starting weights.".into()),

            (Some(dim), Some(w)) if dim != w.len() - 1 => {
                Err("The dimension isn't equal the amount of weights.".into())
            }
            _ => Ok(Self {
                weights: weights.unwrap_or_else(|| {
                    DVector::<T>::from_element(dimension.unwrap() + 1, T::from_f64(1.0).unwrap())
                }),
            }),
        }
    }

    pub fn weights(&self) -> &DVector<T> {
        &self.weights
    }

    pub fn predict(&self, x_pred: &DMatrix<T>) -> Result<DVector<T>, Box<dyn Error>> {
        let x_pred_with_bias = x_pred.clone().insert_column(0, T::from_f64(1.0).unwrap());
        Ok(self.h(&x_pred_with_bias))
    }

    pub fn fit(
        &mut self,
        dataset: &Dataset<T, T>,
        lr: T,
        mut max_steps: usize,
        epsilon: Option<T>,
        progress: Option<usize>,
    ) -> Result<String, Box<dyn Error>> {
        if progress.is_some_and(|steps| steps == 0) {
            return Err(
                "The number of steps for progress visualization must be greater than 0.".into(),
            );
        }

        let (x, y) = dataset.into_parts();

        let epsilon = epsilon.unwrap_or_else(|| T::from_f64(1e-6).unwrap());
        let initial_max_steps = max_steps.clone();
        let x_with_bias = x.clone().insert_column(0, T::from_f64(1.0).unwrap());
        while max_steps > 0 {
            let weights_prev = self.weights.clone();

            let gradient = self.gradient(&x_with_bias, y);

            if gradient.iter().any(|&g| g.is_nan()) {
                return Err("Gradient turned to NaN during training.".into());
            }

            self.weights -= gradient * lr;

            if progress.is_some_and(|steps| max_steps % steps == 0) {
                println!("Step: {}", initial_max_steps - max_steps);
                println!("Weights: {:?}", self.weights);
                println!("MSE: {:?}", self.mse_training(&x_with_bias, y));
            }

            let delta = self
                .weights
                .iter()
                .zip(weights_prev.iter())
                .map(|(&w, &w_prev)| (w - w_prev) * (w - w_prev))
                .fold(T::from_f64(0.0).unwrap(), |acc, x| acc + x);

            if delta < epsilon {
                return Ok(format!(
                    "Finished training in {} steps.",
                    initial_max_steps - max_steps,
                ));
            }
            max_steps -= 1;
        }
        Ok("Reached maximum steps without converging.".into())
    }

    fn gradient(&self, x: &DMatrix<T>, y: &DVector<T>) -> DVector<T> {
        let y_pred = self.h(x);

        let errors = y_pred - y;

        x.transpose() * errors / T::from_usize(y.len()).unwrap()
    }

    fn h(&self, x: &DMatrix<T>) -> DVector<T> {
        x * &self.weights
    }

    pub fn mse_training(&self, x: &DMatrix<T>, y: &DVector<T>) -> T {
        let m = T::from_usize(y.len()).unwrap();
        let y_pred = self.h(x);
        let errors = y_pred - y;
        let errors_sq = errors.component_mul(&errors);

        errors_sq.sum() / m
    }
}
