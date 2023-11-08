//! K-Means Clustering
use nalgebra::DMatrix;
use std::fmt::Debug;

pub trait DataValue: Debug + Clone + PartialOrd {}
impl<T> DataValue for T where T: Debug + Clone + PartialOrd {}

pub struct KMeans<XT: DataValue> {
    data: DMatrix<XT>,
    num_clusters: usize,
}

impl<XT: DataValue> KMeans<XT> {
    pub fn new(data: DMatrix<XT>, num_clusters: Option<usize>) -> Self {
        Self {
            data,
            num_clusters: num_clusters.unwrap_or(2),
        }
    }


}
