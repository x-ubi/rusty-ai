//! # Rusty-ai
//!
//! `rusty-ai` provides implementations of various classification and regression algorithms using Rust.
//! It also contains some utility functions for data manipulation and metrics.
//!
//! ## Getting Started
//!
//! To use `rusty-ai`, add the following to your `Cargo.toml` file:
//!
//! ```toml
//! [dependencies]
//! rusty-ai = "*"
//! ```
//!
//! ## Example Usage
//!
//! As a quick example, here's how you can use `rusty-ai` to train a gaussian naive bayes classifier on an example dataset:
//!
//! ```rust
//!
//! use rusty_ai::bayes::gaussian::*;
//! use rusty_ai::data::dataset::*;
//! use nalgebra::{DMatrix, DVector};
//!
//! let x = DMatrix::from_row_slice(4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//! let y = DVector::from_vec(vec![0, 0, 1, 1]);
//!
//! let dataset = Dataset::new(x, y);
//!
//! let mut model = GaussianNB::new();
//!
//! model.fit(&dataset).unwrap();
//!
//! let test_x = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
//!
//! let predictions = model.predict(&test_x).unwrap();
//! ```

/// Naive Bayes Classifiers
pub mod bayes;
/// Dataset and data manipulation utilities
pub mod data;
/// Random Forests
pub mod forests;
/// Functions for evaluating model performance
pub mod metrics;
/// Regression analysis algorithms
pub mod regression;
/// Decision trees
pub mod trees;
