use crate::dataset::{Dataset, FeatureValue, TargetValue};

pub struct LogisticRegression<XT: FeatureValue, YT: TargetValue> {
    pub dataset: Dataset<XT, YT>,

}