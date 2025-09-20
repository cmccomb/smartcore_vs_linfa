use linfa::prelude::*;
use linfa_bayes::GaussianNb as LinfaGaussianNb;
use linfa_logistic::LogisticRegression as LinfaLogisticRegression;
use linfa_svm::Svm as LinfaSvm;
use linfa_trees::{DecisionTree as LinfaDecisionTree, SplitQuality};
use ndarray::{Array1, Array2, Ix1};
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use smartcore::{
    dataset::Dataset as SCDataset,
    linalg::basic::matrix::DenseMatrix,
    linear::logistic_regression::{
        LogisticRegression as SCLogisticRegression, LogisticRegressionParameters,
    },
    naive_bayes::gaussian::{GaussianNB as SCGaussianNB, GaussianNBParameters},
    svm::{
        svc::{SVCParameters, SVC as SCSVC},
        Kernels,
    },
    tree::decision_tree_classifier::{
        DecisionTreeClassifier, DecisionTreeClassifierParameters, SplitCriterion,
    },
};
use std::{collections::HashMap, sync::LazyLock};

use super::{seeded_rng, TestSize};
use crate::array2_to_dense_matrix;

/// Convert a `SmartCore` classification dataset into dense ndarray arrays.
///
/// # Panics
///
/// Panics if the underlying data vector cannot be reshaped into a
/// `num_samples` by `num_features` matrix. This should only occur when the
/// dataset metadata is inconsistent.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[must_use]
pub fn dataset_to_classification_array(
    dataset: &SCDataset<f32, f32>,
) -> (Array2<f64>, Array1<usize>) {
    (
        Array2::from_shape_vec(
            (dataset.num_samples, dataset.num_features),
            dataset.data.iter().map(|elem| f64::from(*elem)).collect(),
        )
        .unwrap(),
        Array1::from_vec(dataset.target.iter().map(|elem| *elem as usize).collect()),
    )
}

type ClassificationData = (Array2<f64>, Array1<usize>);
type ClassificationCache = HashMap<TestSize, ClassificationData>;

static CLASSIFICATION_DATA_CACHE: LazyLock<ClassificationCache> = LazyLock::new(|| {
    HashMap::from([
        (TestSize::Small, build_classification_data(TestSize::Small)),
        (
            TestSize::Medium,
            build_classification_data(TestSize::Medium),
        ),
        (TestSize::Large, build_classification_data(TestSize::Large)),
    ])
});

#[allow(clippy::cast_precision_loss)]
fn make_seeded_blobs(
    rng: &mut StdRng,
    num_samples: usize,
    num_features: usize,
    num_centers: usize,
) -> SCDataset<f32, f32> {
    let center_box = Uniform::from(-10.0..10.0);
    let cluster_std = 1.0;
    let mut centers: Vec<Vec<Normal<f32>>> = Vec::with_capacity(num_centers);

    for _ in 0..num_centers {
        centers.push(
            (0..num_features)
                .map(|_| Normal::new(center_box.sample(rng), cluster_std).unwrap())
                .collect(),
        );
    }

    let mut y: Vec<f32> = Vec::with_capacity(num_samples);
    let mut x: Vec<f32> = Vec::with_capacity(num_samples * num_features);

    for sample_index in 0..num_samples {
        let label = sample_index % num_centers;
        y.push(label as f32);
        for feature_index in 0..num_features {
            x.push(centers[label][feature_index].sample(rng));
        }
    }

    SCDataset {
        data: x,
        target: y,
        num_samples,
        num_features,
        feature_names: (0..num_features).map(|n| n.to_string()).collect(),
        target_names: vec!["label".to_string()],
        description: "Isotropic Gaussian blobs".to_string(),
    }
}

fn build_classification_data(size: TestSize) -> ClassificationData {
    let (num_samples, num_features, num_centers) = match size {
        TestSize::Small => (100, 10, 2),
        TestSize::Medium => (1000, 50, 2),
        TestSize::Large => (10000, 100, 2),
    };

    let mut rng = seeded_rng(size, "classification");
    let dataset = make_seeded_blobs(&mut rng, num_samples, num_features, num_centers);
    dataset_to_classification_array(&dataset)
}

fn classification_cache(size: TestSize) -> &'static ClassificationData {
    CLASSIFICATION_DATA_CACHE
        .get(&size)
        .expect("classification data cache is populated for all test sizes")
}

#[must_use]
pub fn xy_classification(size: &TestSize) -> (Array2<f64>, Array1<usize>) {
    let (x, y) = classification_cache(*size);
    (x.clone(), y.clone())
}

/// Retrieve `SmartCore`-compatible classification inputs.
///
/// # Panics
///
/// Panics if the cached ndarray cannot be converted into a dense matrix or if
/// a class label exceeds the bounds of `u32`.
#[must_use]
pub fn get_smartcore_classification_data(size: &TestSize) -> (DenseMatrix<f64>, Vec<u32>) {
    let (x, y) = xy_classification(size);
    let dense = array2_to_dense_matrix(&x).expect("valid dense matrix conversion");
    let classes: Vec<u32> = y
        .iter()
        .map(|&elem| u32::try_from(elem).expect("class label fits into u32"))
        .collect();
    (dense, classes)
}

#[must_use]
pub fn get_linfa_classification_data(size: &TestSize) -> Dataset<f64, usize, Ix1> {
    let (x, y) = xy_classification(size);
    Dataset::new(x, y)
}

#[must_use]
pub fn get_linfa_classification_data_as_bool(size: &TestSize) -> Dataset<f64, bool, Ix1> {
    let (x, y) = xy_classification(size);
    let ybool = y.mapv(|elem| elem == 1);
    Dataset::new(x, ybool)
}

/// Logistic regression smartcore
///
/// # Panics
///
/// Panics if the `SmartCore` optimizer fails to converge.
/// ```
/// use smartcore_vs_linfa::{get_smartcore_classification_data, smartcore_logistic_regression, TestSize};
/// let (x, y) = get_smartcore_classification_data(&TestSize::Small);
/// smartcore_logistic_regression(&x, &y);
/// ```
pub fn smartcore_logistic_regression(x: &DenseMatrix<f64>, y: &Vec<u32>) {
    SCLogisticRegression::fit(x, y, LogisticRegressionParameters::default()).unwrap();
}

/// linfa logistic regression
///
/// # Panics
///
/// Panics if the Linfa optimizer fails to converge.
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
///
/// # Panics
///
/// Panics if the tree learner encounters invalid parameters.
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
///
/// # Panics
///
/// Panics if the Linfa decision tree training fails.
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
///
/// # Panics
///
/// Panics if Gaussian Naive Bayes training fails in `SmartCore`.
/// ```
/// use smartcore_vs_linfa::{get_smartcore_classification_data, smartcore_gnb_classifier, TestSize};
/// let (x, y) = get_smartcore_classification_data(&TestSize::Small);
/// smartcore_gnb_classifier(&x, &y);
/// ```
pub fn smartcore_gnb_classifier(x: &DenseMatrix<f64>, y: &Vec<u32>) {
    SCGaussianNB::fit(x, y, GaussianNBParameters::default()).unwrap();
}

/// gaussian naive bayes linfa
///
/// # Panics
///
/// Panics if Gaussian Naive Bayes training fails in Linfa.
/// ```
/// use smartcore_vs_linfa::{get_linfa_classification_data, linfa_gnb_classifier, TestSize};
/// linfa_gnb_classifier(&get_linfa_classification_data(&TestSize::Small));
/// ```
pub fn linfa_gnb_classifier(dataset: &Dataset<f64, usize, Ix1>) {
    LinfaGaussianNb::params().fit(dataset).unwrap();
}

/// svm smartcore
///
/// # Panics
///
/// Panics if the `SmartCore` SVM optimizer fails to converge.
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
///
/// # Panics
///
/// Panics if the Linfa SVM optimizer fails to converge.
/// ```
/// use smartcore_vs_linfa::{get_linfa_classification_data_as_bool, linfa_svm_classifier, TestSize};
/// linfa_svm_classifier(&get_linfa_classification_data_as_bool(&TestSize::Small));
/// ```
pub fn linfa_svm_classifier(dataset: &Dataset<f64, bool, Ix1>) {
    LinfaSvm::<_, bool>::params().fit(dataset).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_data_is_deterministic() {
        let (x_a, y_a) = super::build_classification_data(TestSize::Small);
        let (x_b, y_b) = super::build_classification_data(TestSize::Small);

        assert_eq!(x_a, x_b);
        assert_eq!(y_a, y_b);
    }

    #[test]
    fn classification_data_changes_with_size() {
        let (small_x, _) = super::build_classification_data(TestSize::Small);
        let (medium_x, _) = super::build_classification_data(TestSize::Medium);

        assert_ne!(small_x, medium_x);
    }
}
