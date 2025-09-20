use linfa::prelude::*;
use linfa_bayes::GaussianNb as LinfaGaussianNb;
use linfa_logistic::LogisticRegression as LinfaLogisticRegression;
use linfa_svm::Svm as LinfaSvm;
use linfa_trees::{DecisionTree as LinfaDecisionTree, SplitQuality};
use ndarray::{Array1, Array2, Ix1};
use once_cell::sync::Lazy;
use smartcore::{
    dataset::{generator::make_blobs, Dataset as SCDataset},
    linalg::basic::matrix::DenseMatrix,
    linear::logistic_regression::LogisticRegression as SCLogisticRegression,
    naive_bayes::gaussian::{GaussianNB as SCGaussianNB, GaussianNBParameters},
    svm::{
        svc::{SVCParameters, SVC as SCSVC},
        Kernels,
    },
    tree::decision_tree_classifier::{
        DecisionTreeClassifier, DecisionTreeClassifierParameters, SplitCriterion,
    },
};
use std::collections::HashMap;

use super::TestSize;
use crate::array2_to_dense_matrix;

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

type ClassificationData = (Array2<f64>, Array1<usize>);
type ClassificationCache = HashMap<TestSize, ClassificationData>;

static CLASSIFICATION_DATA_CACHE: Lazy<ClassificationCache> = Lazy::new(|| {
    HashMap::from([
        (
            TestSize::Small,
            dataset_to_classification_array(make_blobs(100, 10, 2)),
        ),
        (
            TestSize::Medium,
            dataset_to_classification_array(make_blobs(1000, 50, 2)),
        ),
        (
            TestSize::Large,
            dataset_to_classification_array(make_blobs(10000, 100, 2)),
        ),
    ])
});

fn classification_cache(size: &TestSize) -> &'static ClassificationData {
    CLASSIFICATION_DATA_CACHE
        .get(size)
        .expect("classification data cache is populated for all test sizes")
}

pub fn xy_classification(size: &TestSize) -> (Array2<f64>, Array1<usize>) {
    let (x, y) = classification_cache(size);
    (x.clone(), y.clone())
}

pub fn get_smartcore_classification_data(size: &TestSize) -> (DenseMatrix<f64>, Vec<u32>) {
    let (x, y) = xy_classification(size);
    let dense = array2_to_dense_matrix(&x).expect("valid dense matrix conversion");
    (dense, y.iter().map(|&elem| elem as u32).collect())
}

pub fn get_linfa_classification_data(size: &TestSize) -> Dataset<f64, usize, Ix1> {
    let (x, y) = xy_classification(size);
    Dataset::new(x, y)
}

pub fn get_linfa_classification_data_as_bool(size: &TestSize) -> Dataset<f64, bool, Ix1> {
    let (x, y) = xy_classification(size);
    let ybool = y.mapv(|elem| elem == 1);
    Dataset::new(x, ybool)
}

/// Logistic regression smartcore
/// ```
/// use smartcore_vs_linfa::{get_smartcore_classification_data, smartcore_logistic_regression, TestSize};
/// let (x, y) = get_smartcore_classification_data(&TestSize::Small);
/// smartcore_logistic_regression(&x, &y);
/// ```
pub fn smartcore_logistic_regression(x: &DenseMatrix<f64>, y: &Vec<u32>) {
    SCLogisticRegression::fit(x, y, Default::default()).unwrap();
}

/// linfa logistic regression
/// ```
/// use smartcore_vs_linfa::{get_linfa_classification_data, linfa_logistic_regression, TestSize};
/// linfa_logistic_regression(&get_linfa_classification_data(&TestSize::Small));
/// ```
pub fn linfa_logistic_regression(dataset: &Dataset<f64, usize, Ix1>) {
    LinfaLogisticRegression::default()
        .gradient_tolerance(1e-8)
        .max_iterations(1000)
        .fit(dataset)
        .unwrap();
}

/// Decision tree smartcore
/// ```
/// use smartcore_vs_linfa::{get_smartcore_classification_data, smartcore_decision_tree_classifier, TestSize};
/// let (x, y) = get_smartcore_classification_data(&TestSize::Small);
/// smartcore_decision_tree_classifier(&x, &y);
/// ```
pub fn smartcore_decision_tree_classifier(x: &DenseMatrix<f64>, y: &Vec<u32>) {
    let params = DecisionTreeClassifierParameters::default()
        .with_criterion(SplitCriterion::Gini)
        .with_max_depth(100)
        .with_min_samples_leaf(1)
        .with_min_samples_split(1);
    DecisionTreeClassifier::fit(x, y, params).unwrap();
}

/// decision tree linfa
/// ```
/// use smartcore_vs_linfa::{get_linfa_classification_data, linfa_decision_tree_classifier, TestSize};
/// linfa_decision_tree_classifier(&get_linfa_classification_data(&TestSize::Small));
/// ```
pub fn linfa_decision_tree_classifier(dataset: &Dataset<f64, usize, Ix1>) {
    LinfaDecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(dataset)
        .unwrap();
}

/// gaussian naive bayes smartcore
/// ```
/// use smartcore_vs_linfa::{get_smartcore_classification_data, smartcore_gnb_classifier, TestSize};
/// let (x, y) = get_smartcore_classification_data(&TestSize::Small);
/// smartcore_gnb_classifier(&x, &y);
/// ```
pub fn smartcore_gnb_classifier(x: &DenseMatrix<f64>, y: &Vec<u32>) {
    SCGaussianNB::fit(x, y, GaussianNBParameters::default()).unwrap();
}

/// gaussian naive bayes linfa
/// ```
/// use smartcore_vs_linfa::{get_linfa_classification_data, linfa_gnb_classifier, TestSize};
/// linfa_gnb_classifier(&get_linfa_classification_data(&TestSize::Small));
/// ```
pub fn linfa_gnb_classifier(dataset: &Dataset<f64, usize, Ix1>) {
    LinfaGaussianNb::params().fit(dataset).unwrap();
}

/// svm smartcore
/// ```
/// use smartcore_vs_linfa::{get_smartcore_classification_data, smartcore_svm_classifier, TestSize};
/// let (x, y) = get_smartcore_classification_data(&TestSize::Small);
/// smartcore_svm_classifier(&x, &y);
/// ```
pub fn smartcore_svm_classifier(x: &DenseMatrix<f64>, y: &Vec<u32>) {
    let params = SVCParameters::default().with_kernel(Kernels::linear());
    SCSVC::fit(x, y, &params).unwrap();
}

/// svm linfa
/// ```
/// use smartcore_vs_linfa::{get_linfa_classification_data_as_bool, linfa_svm_classifier, TestSize};
/// linfa_svm_classifier(&get_linfa_classification_data_as_bool(&TestSize::Small));
/// ```
pub fn linfa_svm_classifier(dataset: &Dataset<f64, bool, Ix1>) {
    LinfaSvm::<_, bool>::params().fit(dataset).unwrap();
}
