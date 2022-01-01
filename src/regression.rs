use linfa::prelude::*;
use linfa_elasticnet::{ElasticNet as LinfaElasticNet, ElasticNet};
use linfa_linear::{FittedLinearRegression, LinearRegression as LinfaLinearRegression};
use linfa_svm::Svm as LinfaSvm;

use ndarray::{Array1, Array2};
use smartcore::{
    dataset::Dataset as SCDataset,
    linalg::naive::dense_matrix::DenseMatrix,
    linear::{
        elastic_net::{ElasticNet as SCElasticNet, ElasticNetParameters},
        linear_regression::LinearRegression as SCLinearRegression,
    },
    svm::svr::{SVRParameters, SVR as SCSVR},
};

use super::make_regression;
use super::TestSize;

pub fn xy_regression(size: &TestSize) -> (Array2<f64>, Array1<f64>) {
    match size {
        TestSize::Small => dataset_to_regression_array(make_regression(100, 10, 1.0)),
        TestSize::Medium => dataset_to_regression_array(make_regression(1000, 50, 1.0)),
        TestSize::Large => dataset_to_regression_array(make_regression(10000, 100, 1.0)),
    }
}

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

pub fn get_smartcore_regression_data(size: &TestSize) -> (DenseMatrix<f64>, Vec<f64>) {
    let (x, y) = xy_regression(size);
    (
        DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap()),
        y.to_vec().iter().map(|&elem| elem as f64).collect(),
    )
}

pub fn get_linfa_regression_data(size: &TestSize) -> Dataset<f64, f64> {
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
    SCLinearRegression::fit(x, y, Default::default());
}

/// Linfa linear regression
/// ```
/// use smartcore_vs_linfa::{get_linfa_regression_data, linfa_linear_regression, TestSize};
/// linfa_linear_regression(&get_linfa_regression_data(&TestSize::Small));
/// ```
pub fn linfa_linear_regression(dataset: &Dataset<f64, f64>) {
    LinfaLinearRegression::new().fit(dataset);
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
    );
}

/// Linfa linear regression
/// ```
/// use smartcore_vs_linfa::{get_linfa_regression_data, linfa_elasticnet_regression, TestSize};
/// linfa_elasticnet_regression(&get_linfa_regression_data(&TestSize::Small));
/// ```
pub fn linfa_elasticnet_regression(dataset: &Dataset<f64, f64>) {
    LinfaElasticNet::params()
        .penalty(0.5)
        .l1_ratio(0.5)
        .max_iterations(1000)
        .tolerance(1e-4)
        .fit(dataset);
}

/// svm smartcore
/// ```
/// use smartcore_vs_linfa::{get_smartcore_regression_data, smartcore_svm_regression, TestSize};
/// let (x, y) = get_smartcore_regression_data(&TestSize::Small);
/// smartcore_svm_regression(&x, &y);
/// ```
pub fn smartcore_svm_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    SCSVR::fit(x, y, SVRParameters::default());
}

/// svm linfa
/// ```
/// use smartcore_vs_linfa::{get_linfa_regression_data, linfa_svm_regression, TestSize};
/// linfa_svm_regression(&get_linfa_regression_data(&TestSize::Small));
/// ```
pub fn linfa_svm_regression(dataset: &Dataset<f64, f64>) {
    LinfaSvm::<_, f64>::params().fit(dataset);
}
