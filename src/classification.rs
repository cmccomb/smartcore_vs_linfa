use linfa::prelude::*;
use linfa_logistic::LogisticRegression as LinfaLogisticRegression;
use linfa_trees::{DecisionTree as LinfaDecisionTree, SplitQuality};
use ndarray::{array, Array1, Array2};
use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    linear::logistic_regression::LogisticRegression as SCLogisticRegression,
    tree::decision_tree_classifier::{
        DecisionTreeClassifier, DecisionTreeClassifierParameters, SplitCriterion,
    },
};

use super::TestSize;

pub fn x_classification(_size: &TestSize) -> Array2<f64> {
    array![
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3],
        [5.0, 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1],
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [6.5, 2.8, 4.6, 1.5],
        [5.7, 2.8, 4.5, 1.3],
        [6.3, 3.3, 4.7, 1.6],
        [4.9, 2.4, 3.3, 1.0],
        [6.6, 2.9, 4.6, 1.3],
        [5.2, 2.7, 3.9, 1.4],
    ]
    .to_owned()
}

pub fn y_classification(_size: &TestSize) -> Array1<usize> {
    array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]
}

pub fn get_smartcore_classification_data(size: &TestSize) -> (DenseMatrix<f64>, Vec<f64>) {
    let x = x_classification(size).to_owned();
    (
        DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap()),
        y_classification(size)
            .to_vec()
            .iter()
            .map(|&elem| elem as f64)
            .collect(),
    )
}

pub fn get_linfa_classification_data(size: &TestSize) -> Dataset<f64, usize> {
    Dataset::new(x_classification(size).to_owned(), y_classification(size))
}

/// Logistic regression smartcore
/// ```
/// use smartcore_vs_linfa::{get_smartcore_classification_data, smartcore_logistic_regression, TestSize};
/// let (x, y) = get_smartcore_classification_data(&TestSize::Small);
/// smartcore_logistic_regression(&x, &y);
/// ```
pub fn smartcore_logistic_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    let _model = SCLogisticRegression::fit(x, y, Default::default()).unwrap();
}

/// linfa logistic regression
/// ```
/// use smartcore_vs_linfa::{get_linfa_classification_data, linfa_logistic_regression, TestSize};
/// linfa_logistic_regression(&get_linfa_classification_data(&TestSize::Small));
/// ```
pub fn linfa_logistic_regression(dataset: &Dataset<f64, usize>) {
    let _model = LinfaLogisticRegression::default()
        .gradient_tolerance(1e-8)
        .max_iterations(1000)
        .fit(dataset)
        .unwrap();
}

/// Decision tree smartcore
/// ```
/// use smartcore_vs_linfa::{get_smartcore_classification_data, smartcore_decision_tree_classifier, TestSize};
/// let (x, y) = get_smartcore_classification_data(&TestSize::Small);
/// smartcore_decision_tree_classifier(&x, &(y.iter().map(|&elem| elem as f64).collect()));
/// ```
pub fn smartcore_decision_tree_classifier(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    let _model = DecisionTreeClassifier::fit(
        x,
        y,
        DecisionTreeClassifierParameters::default()
            .with_criterion(SplitCriterion::Gini)
            .with_max_depth(100)
            .with_min_samples_leaf(1)
            .with_min_samples_split(1),
    )
    .unwrap();
}

/// decision tree linfa
/// ```
/// use smartcore_vs_linfa::{get_linfa_classification_data, linfa_decision_tree_classifier, TestSize};
/// linfa_decision_tree_classifier(&get_linfa_classification_data(&TestSize::Small));
/// ```
pub fn linfa_decision_tree_classifier(dataset: &Dataset<f64, usize>) {
    let _model = LinfaDecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(dataset)
        .unwrap();
}
