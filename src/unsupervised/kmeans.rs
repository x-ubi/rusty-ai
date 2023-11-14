//! K-Means Clustering
use nalgebra::{ComplexField, DMatrix, DVector, SimdRealField};
use num::Num;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Uniform};
use std::fmt::Debug;

pub trait DataValue:
    Debug + Clone + PartialOrd + Num + SimdRealField + ComplexField + SampleUniform + 'static
{
}
impl<T> DataValue for T where
    T: Debug + Clone + PartialOrd + Num + SimdRealField + ComplexField + SampleUniform + 'static
{
}

pub struct KMeans<XT: DataValue> {
    data: DMatrix<XT>,
    centroids: DMatrix<XT>,
    num_clusters: usize,
}

impl<XT: DataValue> KMeans<XT> {
    pub fn new(data: &DMatrix<XT>, num_clusters: Option<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let num_clusters = num_clusters.unwrap_or(2);
        let mut centroids = DMatrix::zeros(num_clusters, data.ncols());

        // should i make this more rust-like? With comprehensions
        for dimension in 0..data.ncols() {
            let column = data.column(dimension);
            let range = Uniform::new_inclusive(column.min(), column.max());
            for i in 0..num_clusters {
                centroids[(i, dimension)] = range.sample(&mut rng);
            }
        }

        Self {
            data: data.clone(),
            centroids: centroids,
            num_clusters: num_clusters,
        }
    }

    fn distance(x: &DVector<XT>, y: &DVector<XT>) -> XT {
        (x - y).norm_squared().sqrt()
    }
}
