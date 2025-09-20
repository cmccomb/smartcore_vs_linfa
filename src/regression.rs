use linfa::prelude::*;
use linfa_elasticnet::ElasticNet as LinfaElasticNet;
use linfa_linear::LinearRegression as LinfaLinearRegression;
use linfa_svm::Svm as LinfaSvm;

use ndarray::{Array1, Array2, Ix1};
use once_cell::sync::Lazy;
use smartcore::{
    dataset::Dataset as SCDataset,
    linalg::basic::matrix::DenseMatrix,
    linear::{
        elastic_net::{ElasticNet as SCElasticNet, ElasticNetParameters},
        linear_regression::LinearRegression as SCLinearRegression,
    },
    svm::{
        svr::{SVRParameters, SVR as SCSVR},
        Kernels,
    },
};
use std::collections::HashMap;

use super::make_regression;
use super::TestSize;
use crate::array2_to_dense_matrix;

pub fn dataset_to_regression_array(dataset: SCDataset<f32, f32>) -> (Array2<f64>, Array1<f64>) {
    (
        Array2::from_shape_vec(
            (dataset.num_samples, dataset.num_features),
            dataset.data.iter().map(|elem| *elem as f64).collect(),
        )
        .unwrap()
        .to_owned(),
        Array1::from_vec(dataset.target.iter().map(|elem| *elem as f64).collect()).to_owned(),
    )
}

type RegressionData = (Array2<f64>, Array1<f64>);
type RegressionCache = HashMap<TestSize, RegressionData>;

static REGRESSION_DATA_CACHE: Lazy<RegressionCache> = Lazy::new(|| {
    HashMap::from([
        (
            TestSize::Small,
            dataset_to_regression_array(make_regression(100, 10, 1.0)),
        ),
        (
            TestSize::Medium,
            dataset_to_regression_array(make_regression(1000, 50, 1.0)),
        ),
        (
            TestSize::Large,
            dataset_to_regression_array(make_regression(10000, 100, 1.0)),
        ),
    ])
});

fn regression_cache(size: &TestSize) -> &'static RegressionData {
    REGRESSION_DATA_CACHE
        .get(size)
        .expect("regression data cache is populated for all test sizes")
}

pub fn xy_regression(size: &TestSize) -> (Array2<f64>, Array1<f64>) {
    let (x, y) = regression_cache(size);
    (x.clone(), y.clone())
}

pub fn get_smartcore_regression_data(size: &TestSize) -> (DenseMatrix<f64>, Vec<f64>) {
    let (x, y) = xy_regression(size);
    let dense = array2_to_dense_matrix(&x).expect("valid dense matrix conversion");
    (dense, y.to_vec())
}

pub fn get_linfa_regression_data(size: &TestSize) -> Dataset<f64, f64, Ix1> {
    let (x, y) = xy_regression(size);
    Dataset::new(x, y)
}

/// Smartcore linear regression
/// ```
/// use smartcore_vs_linfa::{get_smartcore_regression_data, smartcore_linear_regression, TestSize};
/// let (x, y) = get_smartcore_regression_data(&TestSize::Small);
/// smartcore_linear_regression(&x, &y);
/// ```
pub fn smartcore_linear_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    SCLinearRegression::fit(x, y, Default::default()).unwrap();
}

/// Linfa linear regression
/// ```
/// use smartcore_vs_linfa::{get_linfa_regression_data, linfa_linear_regression, TestSize};
/// linfa_linear_regression(&get_linfa_regression_data(&TestSize::Small));
/// ```
pub fn linfa_linear_regression(dataset: &Dataset<f64, f64, Ix1>) {
    LinfaLinearRegression::new().fit(dataset).unwrap();
}

/// Smartcore linear regression
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
/// ```
/// use smartcore_vs_linfa::{get_linfa_regression_data, linfa_svm_regression, TestSize};
/// linfa_svm_regression(&get_linfa_regression_data(&TestSize::Small));
/// ```
pub fn linfa_svm_regression(dataset: &Dataset<f64, f64, Ix1>) {
    LinfaSvm::<_, f64>::params().fit(dataset).unwrap();
}
