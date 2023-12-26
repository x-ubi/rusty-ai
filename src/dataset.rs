use nalgebra::{DMatrix, DVector};
use num_traits::{FromPrimitive, Num, ToPrimitive};
use std::cmp::PartialOrd;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

pub trait DataValue:
    Debug
    + Clone
    + Copy
    + Num
    + FromPrimitive
    + ToPrimitive
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + 'static
{
}

impl<T> DataValue for T where
    T: Debug
        + Clone
        + Copy
        + Num
        + FromPrimitive
        + ToPrimitive
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + 'static
{
}

pub trait FeatureValue: DataValue + PartialOrd {}
impl<T> FeatureValue for T where T: DataValue + PartialOrd {}

pub trait TargetValue: DataValue {}
impl<T> TargetValue for T where T: DataValue {}

pub struct Dataset<XT: FeatureValue, YT: TargetValue> {
    pub x: DMatrix<XT>,
    pub y: DVector<YT>,
}

impl<XT: FeatureValue, YT: TargetValue> Debug for Dataset<XT, YT> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dataset")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}

impl<XT: FeatureValue, YT: TargetValue> Dataset<XT, YT> {
    pub fn new(x: DMatrix<XT>, y: DVector<YT>) -> Self {
        Self { x, y }
    }

    pub fn into_parts(&self) -> (&DMatrix<XT>, &DVector<YT>) {
        (&self.x, &self.y)
    }

    pub fn is_not_empty(&self) -> bool {
        !(self.x.is_empty() || self.y.is_empty())
    }

    pub fn split(&self, feature_index: usize, threshold: XT) -> (Dataset<XT, YT>, Dataset<XT, YT>) {
        let (left_indices, right_indices): (Vec<_>, Vec<_>) = self
            .x
            .row_iter()
            .enumerate()
            .partition(|(_, row)| row[feature_index] <= threshold);

        let left_x: Vec<_> = left_indices
            .iter()
            .map(|&(index, _)| self.x.row(index).clone_owned())
            .collect();
        let left_y: Vec<_> = left_indices
            .iter()
            .map(|&(index, _)| self.y.row(index))
            .collect();

        let right_x: Vec<_> = right_indices
            .iter()
            .map(|&(index, _)| self.x.row(index))
            .collect();
        let right_y: Vec<_> = right_indices
            .iter()
            .map(|&(index, _)| self.y.row(index))
            .collect();

        (
            Dataset::new(DMatrix::from_rows(&left_x), DVector::from_rows(&left_y)),
            Dataset::new(DMatrix::from_rows(&right_x), DVector::from_rows(&right_y)),
        )
    }
}
