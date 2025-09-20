use linfa::prelude::*;
use linfa_elasticnet::ElasticNet as LinfaElasticNet;
use linfa_linear::LinearRegression as LinfaLinearRegression;
use linfa_svm::Svm as LinfaSvm;

use ndarray::{Array1, Array2, Ix1};
use smartcore::{
    dataset::Dataset as SCDataset,
    linalg::basic::matrix::DenseMatrix,
    linear::{
        elastic_net::{ElasticNet as SCElasticNet, ElasticNetParameters},
        linear_regression::{LinearRegression as SCLinearRegression, LinearRegressionParameters},
    },
    svm::{
        svr::{SVRParameters, SVR as SCSVR},
        Kernels,
    },
};
use std::{collections::HashMap, sync::LazyLock};

use super::{make_regression, TestSize};
use crate::array2_to_dense_matrix;

/// Convert a `SmartCore` regression dataset into dense ndarray arrays.
///
/// # Panics
///
/// Panics if the dataset metadata and flattened data vector disagree on the
/// expected shape.
#[must_use]
pub fn dataset_to_regression_array(dataset: &SCDataset<f32, f32>) -> (Array2<f64>, Array1<f64>) {
    (
        Array2::from_shape_vec(
            (dataset.num_samples, dataset.num_features),
            dataset.data.iter().map(|elem| f64::from(*elem)).collect(),
        )
        .unwrap(),
        Array1::from_vec(dataset.target.iter().map(|elem| f64::from(*elem)).collect()),
    )
}

type RegressionData = (Array2<f64>, Array1<f64>);
type RegressionCache = HashMap<TestSize, RegressionData>;

static REGRESSION_DATA_CACHE: LazyLock<RegressionCache> = LazyLock::new(|| {
    HashMap::from([
        (TestSize::Small, build_regression_data(TestSize::Small)),
        (TestSize::Medium, build_regression_data(TestSize::Medium)),
        (TestSize::Large, build_regression_data(TestSize::Large)),
    ])
});

fn build_regression_data(size: TestSize) -> RegressionData {
    let (num_samples, num_features) = match size {
        TestSize::Small => (100, 10),
        TestSize::Medium => (1000, 50),
        TestSize::Large => (10000, 100),
    };

    let dataset = make_regression(size, num_samples, num_features, 1.0);
    dataset_to_regression_array(&dataset)
}

fn regression_cache(size: TestSize) -> &'static RegressionData {
    REGRESSION_DATA_CACHE
        .get(&size)
        .expect("regression data cache is populated for all test sizes")
}

#[must_use]
pub fn xy_regression(size: &TestSize) -> (Array2<f64>, Array1<f64>) {
    let (x, y) = regression_cache(*size);
    (x.clone(), y.clone())
}

/// Retrieve `SmartCore`-compatible regression inputs.
///
/// # Panics
///
/// Panics if the cached ndarray cannot be converted into a dense matrix.
#[must_use]
pub fn get_smartcore_regression_data(size: &TestSize) -> (DenseMatrix<f64>, Vec<f64>) {
    let (x, y) = xy_regression(size);
    let dense = array2_to_dense_matrix(&x).expect("valid dense matrix conversion");
    (dense, y.to_vec())
}

#[must_use]
pub fn get_linfa_regression_data(size: &TestSize) -> Dataset<f64, f64, Ix1> {
    let (x, y) = xy_regression(size);
    Dataset::new(x, y)
}

/// Smartcore linear regression
///
/// # Panics
///
/// Panics if `SmartCore`'s linear regression fit fails.
/// ```
/// use smartcore_vs_linfa::{get_smartcore_regression_data, smartcore_linear_regression, TestSize};
/// let (x, y) = get_smartcore_regression_data(&TestSize::Small);
/// smartcore_linear_regression(&x, &y);
/// ```
pub fn smartcore_linear_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    SCLinearRegression::fit(x, y, LinearRegressionParameters::default()).unwrap();
}

/// Linfa linear regression
///
/// # Panics
///
/// Panics if Linfa's linear regression fit fails.
/// ```
/// use smartcore_vs_linfa::{get_linfa_regression_data, linfa_linear_regression, TestSize};
/// linfa_linear_regression(&get_linfa_regression_data(&TestSize::Small));
/// ```
pub fn linfa_linear_regression(dataset: &Dataset<f64, f64, Ix1>) {
    LinfaLinearRegression::new().fit(dataset).unwrap();
}

/// Smartcore linear regression
///
/// # Panics
///
/// Panics if `SmartCore`'s elastic net solver fails to converge.
/// ```
/// use smartcore_vs_linfa::{get_smartcore_regression_data, smartcore_elasticnet_regression, TestSize};
/// let (x, y) = get_smartcore_regression_data(&TestSize::Small);
/// smartcore_elasticnet_regression(&x, &y);
/// ```
pub fn smartcore_elasticnet_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    SCElasticNet::fit(
        x,
        y,
        ElasticNetParameters::default()
            .with_alpha(0.5)
            .with_l1_ratio(0.5)
            .with_max_iter(1000)
            .with_tol(1e-4),
    )
    .unwrap();
}

/// Linfa linear regression
///
/// # Panics
///
/// Panics if Linfa's elastic net solver fails to converge.
/// ```
/// use smartcore_vs_linfa::{get_linfa_regression_data, linfa_elasticnet_regression, TestSize};
/// linfa_elasticnet_regression(&get_linfa_regression_data(&TestSize::Small));
/// ```
pub fn linfa_elasticnet_regression(dataset: &Dataset<f64, f64, Ix1>) {
    LinfaElasticNet::params()
        .penalty(0.5)
        .l1_ratio(0.5)
        .max_iterations(1000)
        .tolerance(1e-4)
        .fit(dataset)
        .unwrap();
}

/// svm smartcore
///
/// # Panics
///
/// Panics if `SmartCore`'s SVM regression fit fails.
/// ```
/// use smartcore_vs_linfa::{get_smartcore_regression_data, smartcore_svm_regression, TestSize};
/// let (x, y) = get_smartcore_regression_data(&TestSize::Small);
/// smartcore_svm_regression(&x, &y);
/// ```
pub fn smartcore_svm_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    let params = SVRParameters::default().with_kernel(Kernels::linear());
    SCSVR::fit(x, y, &params).unwrap();
}

/// svm linfa
///
/// # Panics
///
/// Panics if Linfa's SVM regression fit fails.
/// ```
/// use smartcore_vs_linfa::{get_linfa_regression_data, linfa_svm_regression, TestSize};
/// linfa_svm_regression(&get_linfa_regression_data(&TestSize::Small));
/// ```
pub fn linfa_svm_regression(dataset: &Dataset<f64, f64, Ix1>) {
    LinfaSvm::<_, f64>::params().fit(dataset).unwrap();
}
