use linfa::prelude::*;
use linfa_clustering::{Dbscan as LinfaDbscan, KMeans as LinfaKMeans};
use ndarray::Array2;
use smartcore::{
    cluster::{
        dbscan::{DBSCANParameters, DBSCAN as SCDBSCAN},
        kmeans::{KMeans as SCKMeans, KMeansParameters},
    },
    dataset::{generator::make_blobs, Dataset as SCDataset},
    linalg::naive::dense_matrix::DenseMatrix,
    math::distance::euclidian::Euclidian,
};

use super::TestSize;

pub fn x_unsupervised(size: &TestSize) -> Array2<f64> {
    match size {
        TestSize::Small => dataset_to_unsupervised_array(make_blobs(100, 10, 5)),
        TestSize::Medium => dataset_to_unsupervised_array(make_blobs(1000, 50, 10)),
        TestSize::Large => dataset_to_unsupervised_array(make_blobs(10000, 100, 20)),
    }
}

pub fn dataset_to_unsupervised_array(dataset: SCDataset<f32, f32>) -> Array2<f64> {
    Array2::from_shape_vec(
        (dataset.num_samples, dataset.num_features),
        dataset.data.iter().map(|elem| *elem as f64).collect(),
    )
        .unwrap()
        .to_owned()
}

pub fn get_smartcore_unsupervised_data(size: &TestSize) -> DenseMatrix<f64> {
    let x = x_clustering(size).to_owned();
    DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap())
}

pub fn get_linfa_unsupervised_data(size: &TestSize) -> Dataset<f64, ()> {
    Dataset::from(x_clustering(size).to_owned())
}


/// Run linfa kmeans
/// ```
/// use smartcore_vs_linfa::{get_linfa_unsupervised_data, linfa_kmeans, TestSize};
/// linfa_kmeans(&get_linfa_unsupervised_data(&TestSize::Small));
/// ```
pub fn linfa_kmeans(dataset: &Dataset<f64, ()>) {
    let _model = LinfaKMeans::params(5)
        .max_n_iterations(10)
        .n_runs(1)
        .fit(dataset);
}

/// Run smartcore kmeans
/// ```
/// use smartcore_vs_linfa::{get_smartcore_unsupervised_data, smartcore_kmeans, TestSize};
/// smartcore_kmeans(&get_smartcore_unsupervised_data(&TestSize::Small));
/// ```
pub fn smartcore_kmeans(x: &DenseMatrix<f64>) {
    SCKMeans::fit(x, KMeansParameters::default().with_k(5).with_max_iter(10));
}

/// Run linfa DBSCAN
/// ```
/// use smartcore_vs_linfa::{get_linfa_unsupervised_data, linfa_dbscan, TestSize};
/// linfa_dbscan(&get_linfa_unsupervised_data(&TestSize::Small));
/// ```
pub fn linfa_dbscan(dataset: &Array2<f64>) {
    let _model = LinfaDbscan::params(5).transform(dataset);
}

/// Run smartcore dbscan
/// ```
/// use smartcore_vs_linfa::{get_smartcore_unsupervised_data, smartcore_dbscan, TestSize};
/// smartcore_dbscan(&get_smartcore_unsupervised_data(&TestSize::Small));
/// ```
pub fn smartcore_dbscan(x: &DenseMatrix<f64>)  {
    SCDBSCAN::fit(
        x,
        DBSCANParameters::default()
            .with_min_samples(5)
            .with_eps(1e-4),
    );
}
