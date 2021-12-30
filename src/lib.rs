use linfa::prelude::*;
use linfa_clustering::KMeans as LinfaKMeans;
use linfa_linear::LinearRegression as LinfaLinearRegression;
use linfa_logistic::LogisticRegression as LinfaLogisticRegression;
use ndarray::{array, Array1, Array2};
use smartcore::{
    cluster::kmeans::{KMeans as SCKMeans, KMeansParameters},
    linalg::naive::dense_matrix::DenseMatrix,
    linear::{
        linear_regression::LinearRegression as SCLinearRegression,
        logistic_regression::LogisticRegression as SCLogisticRegression,
    },
};

pub fn x_regression() -> Array2<f64> {
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

pub fn y_regression() -> Array1<f64> {
    array![
        83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2,
        115.7, 116.9,
    ]
}

pub fn get_smartcore_regression_data() -> (DenseMatrix<f64>, Vec<f64>) {
    let x = x_regression().to_owned();
    (
        DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap()),
        y_regression().to_vec(),
    )
}

pub fn smartcore_linear_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    let _model = SCLinearRegression::fit(x, y, Default::default()).unwrap();
}

pub fn get_linfa_regression_data() -> Dataset<f64, f64> {
    Dataset::new(x_regression().to_owned(), y_regression())
}

pub fn linfa_linear_regression(dataset: &Dataset<f64, f64>) {
    let _model = LinfaLinearRegression::new().fit(dataset).unwrap();
}

pub fn x_classification() -> Array2<f64> {
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

pub fn y_classification() -> Array1<u64> {
    array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]
}

pub fn get_smartcore_classification_data() -> (DenseMatrix<f64>, Vec<u64>) {
    let x = x_classification().to_owned();
    (
        DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap()),
        y_classification().to_vec(),
    )
}

pub fn get_linfa_classification_data() -> Dataset<f64, u64> {
    Dataset::new(x_classification().to_owned(), y_classification())
}

pub fn smartcore_logistic_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    let _lr = SCLogisticRegression::fit(x, y, Default::default()).unwrap();
}

pub fn linfa_logistic_regression(dataset: &Dataset<f64, u64>) {
    let _lin_reg = LinfaLogisticRegression::default()
        .gradient_tolerance(1e-8)
        .max_iterations(1000)
        .fit(dataset)
        .unwrap();
}

pub fn x_clustering() -> Array2<f64> {
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

pub fn get_smartcore_clustering_data() -> DenseMatrix<f64> {
    let x = x_classification().to_owned();
    DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap())
}

pub fn get_linfa_clustering_data() -> Dataset<f64, ()> {
    Dataset::from(x_clustering().to_owned())
}

pub fn linfa_kmeans(dataset: &Dataset<f64, ()>) {
    let _model = LinfaKMeans::params(2)
        .max_n_iterations(3)
        .n_runs(1)
        .fit(dataset)
        .unwrap();
}
pub fn smartcore_kmeans(x: &DenseMatrix<f64>) {
    let _kmeans = SCKMeans::fit(x, KMeansParameters::default().with_k(2).with_max_iter(3)).unwrap();
}
