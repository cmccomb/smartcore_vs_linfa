use linfa::prelude::*;
use linfa_logistic::{FittedLogisticRegression, LogisticRegression as LinfaLogisticRegression};
use linfa_trees::{DecisionTree as LinfaDecisionTree, DecisionTree, SplitQuality};
use ndarray::{Array1, Array2};
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::{
    dataset::{generator::make_blobs, Dataset as SCDataset},
    linalg::naive::dense_matrix::DenseMatrix,
    linear::logistic_regression::LogisticRegression as SCLogisticRegression,
    tree::decision_tree_classifier::{
        DecisionTreeClassifier, DecisionTreeClassifierParameters, SplitCriterion,
    },
};

use super::TestSize;

pub fn xy_classification(size: &TestSize) -> (Array2<f64>, Array1<usize>) {
    match size {
        TestSize::Small => dataset_to_classification_array(make_blobs(100, 10, 2)),
        TestSize::Medium => dataset_to_classification_array(make_blobs(1000, 50, 2)),
        TestSize::Large => dataset_to_classification_array(make_blobs(10000, 100, 2)),
    }
}

pub fn dataset_to_classification_array(
    dataset: SCDataset<f32, f32>,
) -> (Array2<f64>, Array1<usize>) {
    (
        Array2::from_shape_vec(
            (dataset.num_samples, dataset.num_features),
            dataset.data.iter().map(|elem| *elem as f64).collect(),
        )
            .unwrap()
            .to_owned(),
        Array1::from_vec(dataset.target.iter().map(|elem| *elem as usize).collect()).to_owned(),
    )
}

pub fn get_smartcore_classification_data(size: &TestSize) -> (DenseMatrix<f64>, Vec<f64>) {
    let (x, y) = xy_classification(size);
    (
        DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap()),
        y.to_vec().iter().map(|&elem| elem as f64).collect(),
    )
}

pub fn get_linfa_classification_data(size: &TestSize) -> Dataset<f64, usize> {
    let (x, y) = xy_classification(size);
    Dataset::new(x, y)
}

/// Logistic regression smartcore
/// ```
/// use smartcore_vs_linfa::{get_smartcore_classification_data, smartcore_logistic_regression, TestSize};
/// let (x, y) = get_smartcore_classification_data(&TestSize::Small);
/// smartcore_logistic_regression(&x, &y);
/// ```
pub fn smartcore_logistic_regression(
    x: &DenseMatrix<f64>,
    y: &Vec<f64>,
)  {
    SCLogisticRegression::fit(x, y, Default::default());
}

/// linfa logistic regression
/// ```
/// use smartcore_vs_linfa::{get_linfa_classification_data, linfa_logistic_regression, TestSize};
/// linfa_logistic_regression(&get_linfa_classification_data(&TestSize::Small));
/// ```
pub fn linfa_logistic_regression(
    dataset: &Dataset<f64, usize>,
) {
    LinfaLogisticRegression::default()
        .gradient_tolerance(1e-8)
        .max_iterations(1000)
        .fit(dataset);
}

/// Decision tree smartcore
/// ```
/// use smartcore_vs_linfa::{get_smartcore_classification_data, smartcore_decision_tree_classifier, TestSize};
/// let (x, y) = get_smartcore_classification_data(&TestSize::Small);
/// smartcore_decision_tree_classifier(&x, &y);
/// ```
pub fn smartcore_decision_tree_classifier(
    x: &DenseMatrix<f64>,
    y: &Vec<f64>,
) {
    DecisionTreeClassifier::fit(
        x,
        y,
        DecisionTreeClassifierParameters::default()
            .with_criterion(SplitCriterion::Gini)
            .with_max_depth(100)
            .with_min_samples_leaf(1)
            .with_min_samples_split(1),
    );
}

/// decision tree linfa
/// ```
/// use smartcore_vs_linfa::{get_linfa_classification_data, linfa_decision_tree_classifier, TestSize};
/// linfa_decision_tree_classifier(&get_linfa_classification_data(&TestSize::Small));
/// ```
pub fn linfa_decision_tree_classifier(dataset: &Dataset<f64, usize>) {
    LinfaDecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(dataset);
}
