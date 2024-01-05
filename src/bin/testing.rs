use csv::ReaderBuilder;
use nalgebra::{DMatrix, DVector};
use rusty_ai::dataset::Dataset;
use rusty_ai::forests::classifier::RandomForestClassifier;
use rusty_ai::regression::logistic::LogisticRegression;
use rusty_ai::trees::classifier::DecisionTreeClassifier;
use std::collections::HashMap;
use std::error::Error;

fn read_file_classification(
    file_path: &str,
    dimension: usize,
    header: bool,
) -> Result<Dataset<f64, u8>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(header)
        .from_path(file_path)?;
    let mut features = Vec::new();
    let mut labels = Vec::new();
    let mut label_map = HashMap::new();
    let mut label_count = 0;

    for result in reader.records() {
        let record = result?;
        let mut feature_row = Vec::new();

        for feature in record.iter().take(dimension) {
            feature_row.push(feature.parse::<f64>()?);
        }

        let label = record.get(dimension).ok_or("Missing label")?;
        let label_id = *label_map.entry(label.to_string()).or_insert_with(|| {
            let id = label_count;
            label_count += 1;
            id
        });

        features.push(feature_row);
        labels.push(label_id);
    }
    let feature_matrix =
        DMatrix::from_row_slice(features.len(), features[0].len(), &features.concat());
    let label_vector = DVector::from_vec(labels);

    Ok(Dataset::new(feature_matrix, label_vector))
}

fn test_tree_classifier(
    train_dataset: &Dataset<f64, u8>,
    test_dataset: &Dataset<f64, u8>,
) -> Result<(), String> {
    let mut classifier = DecisionTreeClassifier::with_params(None, None, Some(3));
    classifier.fit(&train_dataset)?;
    let predictions = classifier.predict(&test_dataset.x)?;
    let mut correct = 0;
    for (prediction, actual) in predictions.iter().zip(test_dataset.y.iter()) {
        if prediction == actual {
            correct += 1;
        }
    }
    println!(
        "Accuracy: {}%",
        (correct as f64 / test_dataset.y.len() as f64) * 100.0
    );
    Ok(())
}

fn test_random_forest_classifier(
    train_dataset: &Dataset<f64, u8>,
    test_dataset: &Dataset<f64, u8>,
) -> Result<(), String> {
    let mut classifier = RandomForestClassifier::new();
    classifier.fit(train_dataset, None)?;
    let predictions = classifier.predict(&test_dataset.x)?;
    let mut correct = 0;
    for (prediction, actual) in predictions.iter().zip(test_dataset.y.iter()) {
        if prediction == actual {
            correct += 1;
        }
    }
    println!(
        "Accuracy: {}%",
        (correct as f64 / test_dataset.y.len() as f64) * 100.0
    );
    Ok(())
}

fn test_logistic_regression(
    train_dataset: &Dataset<f64, u8>,
    test_dataset: &Dataset<f64, u8>,
) -> Result<String, Box<dyn Error>> {
    let mut classifier = LogisticRegression::new(Some(30), None)?;
    println!(
        "{}",
        classifier.fit(train_dataset, 0.00001, 1000, Some(1e-8), Some(50))?
    );
    let predictions = classifier.predict(&test_dataset.x);
    let mut correct = 0;
    for (prediction, actual) in predictions.iter().zip(test_dataset.y.iter()) {
        if prediction == actual {
            correct += 1;
        }
    }
    let accuracy = (correct as f64 / test_dataset.y.len() as f64) * 100.0;
    Ok(format!("Accuracy: {}%", accuracy))
}

fn main() {
    let mut dataset = match read_file_classification("datasets/cancer_clean.csv", 30, true) {
        Ok(dataset) => {
            println!("Loaded dataset");
            dataset
        }
        Err(err) => panic!("{}", err),
    };
    dataset.standardize();

    let (train_dataset, test_dataset) = match dataset.train_test_split(0.75, None) {
        Ok(datasets) => datasets,
        Err(err) => panic!("{}", err),
    };
    println!("{:?}", test_tree_classifier(&train_dataset, &test_dataset));
}
