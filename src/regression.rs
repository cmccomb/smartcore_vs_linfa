use linfa::prelude::*;
use linfa_elasticnet::ElasticNet as LinfaElasticNet;
use linfa_linear::LinearRegression as LinfaLinearRegression;
use ndarray::{array, Array1, Array2};
use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    linear::{
        elastic_net::{ElasticNet as SCElasticNet, ElasticNetParameters},
        linear_regression::LinearRegression as SCLinearRegression,
    },
};

use super::TestSize;

pub fn x_regression(_size: &TestSize) -> Array2<f64> {
    array![
        [234.289, 235.6, 159.0, 107.608, 1947., 60.323],
        [259.426, 232.5, 145.6, 108.632, 1948., 61.122],
        [258.054, 368.2, 161.6, 109.773, 1949., 60.171],
        [284.599, 335.1, 165.0, 110.929, 1950., 61.187],
        [328.975, 209.9, 309.9, 112.075, 1951., 63.221],
        [346.999, 193.2, 359.4, 113.270, 1952., 63.639],
        [365.385, 187.0, 354.7, 115.094, 1953., 64.989],
        [363.112, 357.8, 335.0, 116.219, 1954., 63.761],
        [397.469, 290.4, 304.8, 117.388, 1955., 66.019],
        [419.180, 282.2, 285.7, 118.734, 1956., 67.857],
        [442.769, 293.6, 279.8, 120.445, 1957., 68.169],
        [444.546, 468.1, 263.7, 121.950, 1958., 66.513],
        [482.704, 381.3, 255.2, 123.366, 1959., 68.655],
        [502.601, 393.1, 251.4, 125.368, 1960., 69.564],
        [518.173, 480.6, 257.2, 127.852, 1961., 69.331],
        [554.894, 400.7, 282.7, 130.081, 1962., 70.551],
    ]
    .to_owned()
}

pub fn y_regression(_size: &TestSize) -> Array1<f64> {
    array![
        83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2,
        115.7, 116.9,
    ]
}

pub fn get_smartcore_regression_data(size: &TestSize) -> (DenseMatrix<f64>, Vec<f64>) {
    let x = x_regression(size).to_owned();
    (
        DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap()),
        y_regression(&size).to_vec(),
    )
}

pub fn get_linfa_regression_data(size: &TestSize) -> Dataset<f64, f64> {
    Dataset::new(x_regression(size).to_owned(), y_regression(size))
}

/// Smartcore linear regression
/// ```
/// use smartcore_vs_linfa::{get_smartcore_regression_data, smartcore_linear_regression, TestSize};
/// let (x, y) = get_smartcore_regression_data(&TestSize::Small);
/// smartcore_linear_regression(&x, &y);
/// ```
pub fn smartcore_linear_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    let _model = SCLinearRegression::fit(x, y, Default::default()).unwrap();
}

/// Linfa linear regression
/// ```
/// use smartcore_vs_linfa::{get_linfa_regression_data, linfa_linear_regression, TestSize};
/// linfa_linear_regression(&get_linfa_regression_data(&TestSize::Small));
/// ```
pub fn linfa_linear_regression(dataset: &Dataset<f64, f64>) {
    let _model = LinfaLinearRegression::new().fit(dataset).unwrap();
}

/// Smartcore linear regression
/// ```
/// use smartcore_vs_linfa::{get_smartcore_regression_data, smartcore_elasticnet_regression, TestSize};
/// let (x, y) = get_smartcore_regression_data(&TestSize::Small);
/// smartcore_elasticnet_regression(&x, &y);
/// ```
pub fn smartcore_elasticnet_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    let _model = SCElasticNet::fit(
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
pub fn linfa_elasticnet_regression(dataset: &Dataset<f64, f64>) {
    let _model = LinfaElasticNet::params()
        .penalty(0.5)
        .l1_ratio(0.5)
        .max_iterations(1000)
        .tolerance(1e-4)
        .fit(dataset)
        .unwrap();
}
