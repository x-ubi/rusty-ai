use std::{collections::HashSet, error::Error};

use nalgebra::{DMatrix, DVector};

use crate::data::dataset::WholeNumber;

type ConfusionMatrix = DMatrix<usize>;

pub trait ClassificationMetrics<T: WholeNumber> {
    /// Computes the confusion matrix based on the true labels and predicted labels.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true labels.
    /// * `y_pred` - The predicted labels.
    ///
    /// # Returns
    ///
    /// The confusion matrix as a `Result` containing a `ConfusionMatrix` or an error message.
    fn confusion_matrix(
        &self,
        y_true: &DVector<T>,
        y_pred: &DVector<T>,
    ) -> Result<ConfusionMatrix, Box<dyn Error>> {
        if y_true.len() != y_pred.len() {
            return Err("Predictions and labels are of different sizes.".into());
        }

        let mut classes_set = HashSet::<T>::new();
        classes_set.extend(y_true);
        classes_set.extend(y_pred);

        let mut classes = Vec::from_iter(classes_set.iter().cloned());
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut matrix = DMatrix::zeros(classes_set.len(), classes_set.len());

        for (y_t, y_p) in y_true.iter().zip(y_pred.iter()) {
            let matrix_row = classes.iter().position(|&c| c == *y_t).unwrap();
            let matrix_col = classes.iter().position(|&c| c == *y_p).unwrap();
            matrix[(matrix_row, matrix_col)] += 1;
        }

        Ok(matrix)
    }

    /// Computes the accuracy based on the true labels and predicted labels.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true labels.
    /// * `y_pred` - The predicted labels.
    ///
    /// # Returns
    ///
    /// The accuracy as a `Result` containing a `f64` value or an error message.
    fn accuracy(&self, y_true: &DVector<T>, y_pred: &DVector<T>) -> Result<f64, Box<dyn Error>> {
        let matrix = self.confusion_matrix(y_true, y_pred)?;

        let mut correct = 0;

        matrix.diagonal().iter().for_each(|e| correct += e);

        Ok(correct as f64 / y_true.len() as f64)
    }

    /// Computes the precision based on the true labels and predicted labels.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true labels.
    /// * `y_pred` - The predicted labels.
    ///
    /// # Returns
    ///
    /// The precision as a `Result` containing a `f64` value or an error message.
    fn precision(&self, y_true: &DVector<T>, y_pred: &DVector<T>) -> Result<f64, Box<dyn Error>> {
        let matrix = self.confusion_matrix(y_true, y_pred)?;

        let num_classes = matrix.nrows();

        if num_classes == 2 {
            let tp = matrix[(1, 1)];
            let fp = matrix[(0, 1)];

            if tp + fp > 0 {
                return Ok(tp as f64 / (tp + fp) as f64);
            }
        }

        let mut precision_total = 0.0;
        for class in 0..num_classes {
            let tp = matrix[(class, class)];
            let fp = matrix.column(class).sum() - tp;

            if tp + fp > 0 {
                let precision = tp as f64 / (tp + fp) as f64;
                precision_total += precision;
            }
        }

        Ok(precision_total / num_classes as f64)
    }

    /// Computes the recall based on the true labels and predicted labels.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true labels.
    /// * `y_pred` - The predicted labels.
    ///
    /// # Returns
    ///
    /// The recall as a `Result` containing a `f64` value or an error message.
    fn recall(&self, y_true: &DVector<T>, y_pred: &DVector<T>) -> Result<f64, Box<dyn Error>> {
        let matrix = self.confusion_matrix(y_true, y_pred)?;

        let num_classes = matrix.nrows();

        if num_classes == 2 {
            let tp = matrix[(1, 1)];
            let fn_ = matrix[(1, 0)];

            if tp + fn_ > 0 {
                return Ok(tp as f64 / (tp + fn_) as f64);
            }
        }

        let mut recall_total = 0.0;

        for class in 0..num_classes {
            let tp = matrix[(class, class)];
            let fn_ = matrix.row(class).sum() - tp;

            if tp + fn_ > 0 {
                let recall = tp as f64 / (tp + fn_) as f64;
                recall_total += recall;
            }
        }

        Ok(recall_total / num_classes as f64)
    }

    /// Computes the F1 score based on the true labels and predicted labels.
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true labels.
    /// * `y_pred` - The predicted labels.
    ///
    /// # Returns
    ///
    /// The F1 score as a `Result` containing a `f64` value or an error message.
    fn f1_score(&self, y_true: &DVector<T>, y_pred: &DVector<T>) -> Result<f64, Box<dyn Error>> {
        let precision = self.precision(y_true, y_pred)?;
        let recall = self.recall(y_true, y_pred)?;

        match (precision + recall).abs() < std::f64::EPSILON {
            true => Err("Precision and recall are both 0, F1 score undefined.".into()),
            false => Ok(2.0 * (precision * recall) / (precision + recall)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    struct MockClassifier;

    impl ClassificationMetrics<u8> for MockClassifier {}

    #[test]
    fn test_confusion_matrix() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 0, 1, 0, 1]);
        let y_pred = DVector::from_vec(vec![1, 1, 0, 0, 1]);

        let result = classifier.confusion_matrix(&y_true, &y_pred).unwrap();

        let expected = DMatrix::from_vec(2, 2, vec![1, 1, 1, 2]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_confusion_matrix_unequal() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 0, 1, 0, 1, 0]);
        let y_pred = DVector::from_vec(vec![1, 1, 0, 0, 1]);

        let result = classifier.confusion_matrix(&y_true, &y_pred);

        assert!(result.is_err());
    }

    #[test]
    fn test_confusion_matrix_multiclass() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![0, 1, 2, 1, 0, 2]);
        let y_pred = DVector::from_vec(vec![0, 2, 1, 1, 0, 2]);

        let result = classifier.confusion_matrix(&y_true, &y_pred).unwrap();
        let expected = DMatrix::from_vec(3, 3, vec![2, 0, 0, 0, 1, 1, 0, 1, 1]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_accuracy() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 0, 1, 0, 1]);
        let y_pred = DVector::from_vec(vec![1, 1, 0, 0, 1]);

        let result = classifier.accuracy(&y_true, &y_pred).unwrap();

        let expected = 0.6;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_accuracy_perfect_classification() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 0, 1, 0, 1]);
        let y_pred = DVector::from_vec(vec![1, 0, 1, 0, 1]);

        let result = classifier.accuracy(&y_true, &y_pred).unwrap();
        let expected = 1.0;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_precision() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 0, 1, 0, 1]);
        let y_pred = DVector::from_vec(vec![1, 1, 0, 0, 1]);

        let conf = classifier.confusion_matrix(&y_true, &y_pred).unwrap();
        println!("conf: {}", conf);
        let result = classifier.precision(&y_true, &y_pred).unwrap();

        let expected = 2.0 / 3.0;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_precision_no_positive_predictions() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 1, 1, 1, 1]);
        let y_pred = DVector::from_vec(vec![0, 0, 0, 0, 0]);

        let result = classifier.precision(&y_true, &y_pred).unwrap();

        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_precision_multiclass() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![0, 1, 2, 1, 0, 2]);
        let y_pred = DVector::from_vec(vec![0, 2, 1, 1, 0, 2]);

        let result = classifier.precision(&y_true, &y_pred).unwrap();
        let expected = (2.0 / 2.0 + 1.0 / 2.0 + 1.0 / 2.0) / 3.0;

        assert!((result - expected).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_recall() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 0, 1, 0, 1]);
        let y_pred = DVector::from_vec(vec![1, 1, 0, 0, 1]);

        let result = classifier.recall(&y_true, &y_pred).unwrap();

        let expected = 2.0 / 3.0;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_recall_no_true_positives() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 1, 1, 1, 1]);
        let y_pred = DVector::from_vec(vec![0, 0, 0, 0, 0]);

        let result = classifier.recall(&y_true, &y_pred).unwrap();
        let expected = 0.0;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_recall_multiclass() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![0, 1, 2, 1, 0, 2]);
        let y_pred = DVector::from_vec(vec![0, 2, 1, 1, 0, 2]);

        let result = classifier.recall(&y_true, &y_pred).unwrap();
        let expected = (2.0 / 2.0 + 1.0 / 2.0 + 1.0 / 2.0) / 3.0;

        assert!((result - expected).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_f1_score() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 0, 1, 0, 1]);
        let y_pred = DVector::from_vec(vec![1, 1, 0, 0, 1]);

        let result = classifier.f1_score(&y_true, &y_pred).unwrap();

        let expected = 2.0 / 3.0;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_f1_score_perfect_classification() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 0, 1, 0, 1]);
        let y_pred = DVector::from_vec(vec![1, 0, 1, 0, 1]);

        let result = classifier.f1_score(&y_true, &y_pred).unwrap();
        let expected = 1.0;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_f1_score_error() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![1, 1, 1, 1, 1]);
        let y_pred = DVector::from_vec(vec![0, 0, 0, 0, 0]);

        let result = classifier.f1_score(&y_true, &y_pred);

        assert!(result.is_err());
    }

    #[test]
    fn test_f1_score_multiclass() {
        let classifier = MockClassifier;

        let y_true = DVector::from_vec(vec![0, 1, 2, 1, 0, 2]);
        let y_pred = DVector::from_vec(vec![0, 2, 1, 1, 0, 2]);

        let result = classifier.f1_score(&y_true, &y_pred).unwrap();
        let precision = classifier.precision(&y_true, &y_pred).unwrap();
        let recall = classifier.recall(&y_true, &y_pred).unwrap();
        let expected = 2.0 * (precision * recall) / (precision + recall); // Harmonic mean of precision and recall

        assert!((result - expected).abs() < std::f64::EPSILON);
    }
}
