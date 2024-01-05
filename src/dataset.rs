use nalgebra::{DMatrix, DVector};
use num_traits::{Float, FromPrimitive, Num, ToPrimitive};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};
use std::cmp::PartialOrd;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
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

pub trait Number: DataValue + PartialOrd {}
impl<T> Number for T where T: DataValue + PartialOrd {}

pub trait WholeNumber: Number + Eq + Hash {}
impl<T> WholeNumber for T where T: Number + Eq + Hash {}

pub trait RealNumber: Number + Float {}
impl<T> RealNumber for T where T: Number + Float {}

pub trait TargetValue: DataValue {}
impl<T> TargetValue for T where T: DataValue {}

pub struct Dataset<XT: Number, YT: TargetValue> {
    pub x: DMatrix<XT>,
    pub y: DVector<YT>,
}

impl<XT: Number, YT: TargetValue> Debug for Dataset<XT, YT> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Dataset {{\n    x: [\n")?;

        for i in 0..self.x.nrows() {
            write!(f, "        [")?;
            for j in 0..self.x.ncols() {
                write!(f, "{:?}, ", self.x[(i, j)])?;
            }
            writeln!(f, "]")?;
        }

        write!(f, "    ],\n    y: [")?;
        for i in 0..self.y.len() {
            write!(f, "{:?}, ", self.y[i])?;
        }
        write!(f, "]\n}}")
    }
}

impl<XT: Number, YT: TargetValue> Dataset<XT, YT> {
    pub fn new(x: DMatrix<XT>, y: DVector<YT>) -> Self {
        Self { x, y }
    }

    pub fn into_parts(&self) -> (&DMatrix<XT>, &DVector<YT>) {
        (&self.x, &self.y)
    }

    pub fn is_not_empty(&self) -> bool {
        !(self.x.is_empty() || self.y.is_empty())
    }

    pub fn standardize(&mut self)
    where
        XT: RealNumber,
    {
        let (nrows, ncols) = self.x.shape();

        let means = self
            .x
            .column_iter()
            .map(|col| col.sum() / XT::from_usize(col.len()).unwrap())
            .collect::<Vec<_>>();
        let std_devs = self
            .x
            .column_iter()
            .zip(means.iter())
            .map(|(col, mean)| {
                let mut sum = XT::from_f64(0.0).unwrap();
                for val in col.iter() {
                    sum += (*val - *mean) * (*val - *mean);
                }
                (sum / XT::from_usize(nrows).unwrap()).sqrt()
            })
            .collect::<Vec<_>>();
        let standardized_cols = self
            .x
            .column_iter()
            .zip(means.iter())
            .zip(std_devs.iter())
            .map(|((col, &mean), &std_dev)| col.map(|val| (val - mean) / std_dev))
            .collect::<Vec<_>>();
        self.x = DMatrix::from_columns(&standardized_cols);
    }

    pub fn train_test_split(
        &self,
        train_size: f64,
        seed: Option<u64>,
    ) -> Result<(Self, Self), Box<dyn Error>> {
        if !(0.0..=1.0).contains(&train_size) {
            return Err("Train size should be between 0.0 and 1.0".into());
        }
        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let mut indices = (0..self.x.nrows()).collect::<Vec<_>>();
        indices.shuffle(&mut rng);
        let train_size = (self.x.nrows() as f64 * train_size).floor() as usize;
        let train_indices = &indices[..train_size];
        let test_indices = &indices[train_size..];

        let train_x = train_indices
            .iter()
            .map(|&index| self.x.row(index))
            .collect::<Vec<_>>();
        let train_y = train_indices
            .iter()
            .map(|&index| self.y[index])
            .collect::<Vec<_>>();

        let test_x = test_indices
            .iter()
            .map(|&index| self.x.row(index))
            .collect::<Vec<_>>();
        let test_y = test_indices
            .iter()
            .map(|&index| self.y[index])
            .collect::<Vec<_>>();

        let train_dataset = Self::new(DMatrix::from_rows(&train_x), DVector::from_vec(train_y));
        let test_dataset = Self::new(DMatrix::from_rows(&test_x), DVector::from_vec(test_y));

        Ok((train_dataset, test_dataset))
    }

    pub fn split_on_threshold(&self, feature_index: usize, threshold: XT) -> (Self, Self) {
        let (left_indices, right_indices): (Vec<_>, Vec<_>) = self
            .x
            .row_iter()
            .enumerate()
            .partition(|(_, row)| row[feature_index] <= threshold);

        let left_x: Vec<_> = left_indices
            .iter()
            .map(|&(index, _)| self.x.row(index))
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

        let left_dataset = if left_x.is_empty() {
            Self::new(DMatrix::zeros(0, self.x.ncols()), DVector::zeros(0))
        } else {
            Self::new(DMatrix::from_rows(&left_x), DVector::from_rows(&left_y))
        };

        let right_dataset = if right_x.is_empty() {
            Self::new(DMatrix::zeros(0, self.x.ncols()), DVector::zeros(0))
        } else {
            Self::new(DMatrix::from_rows(&right_x), DVector::from_rows(&right_y))
        };

        (left_dataset, right_dataset)
    }

    pub fn samples(&self, sample_size: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let nrows = self.x.nrows();
        let sample_indices = (0..sample_size)
            .map(|_| rng.gen_range(0..nrows))
            .collect::<Vec<_>>();

        let sample_x = sample_indices
            .iter()
            .map(|&index| self.x.row(index))
            .collect::<Vec<_>>();
        let sample_y = sample_indices
            .iter()
            .map(|&index| self.y[index])
            .collect::<Vec<_>>();

        Self::new(DMatrix::from_rows(&sample_x), DVector::from_vec(sample_y))
    }
}
