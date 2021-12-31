use linfa::prelude::*;
use linfa_logistic::LogisticRegression as LinfaLogisticRegression;
use ndarray::{array, Array1, Array2};
use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    linear::logistic_regression::LogisticRegression as SCLogisticRegression,
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

pub fn y_classification(_size: &TestSize) -> Array1<u64> {
    array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]
}

pub fn get_smartcore_classification_data(size: &TestSize) -> (DenseMatrix<f64>, Vec<u64>) {
    let x = x_classification(size).to_owned();
    (
        DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap()),
        y_classification(size).to_vec(),
    )
}

pub fn get_linfa_classification_data(size: &TestSize) -> Dataset<f64, u64> {
    Dataset::new(x_classification(size).to_owned(), y_classification(size))
}

pub fn smartcore_logistic_regression(x: &DenseMatrix<f64>, y: &Vec<f64>) {
    let _model = SCLogisticRegression::fit(x, y, Default::default()).unwrap();
}

pub fn linfa_logistic_regression(dataset: &Dataset<f64, u64>) {
    let _model = LinfaLogisticRegression::default()
        .gradient_tolerance(1e-8)
        .max_iterations(1000)
        .fit(dataset)
        .unwrap();
}
