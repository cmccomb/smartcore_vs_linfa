use linfa::prelude::*;
use linfa_clustering::{Dbscan as LinfaDbscan, KMeans as LinfaKMeans};
use ndarray::{Array2, Ix1};
use once_cell::sync::Lazy;
use smartcore::{
    cluster::{
        dbscan::{DBSCANParameters, DBSCAN as SCDBSCAN},
        kmeans::{KMeans as SCKMeans, KMeansParameters},
    },
    dataset::{generator::make_blobs, Dataset as SCDataset},
    linalg::basic::matrix::DenseMatrix,
    metrics::distance::euclidian::Euclidian,
};
use std::collections::HashMap;

use super::TestSize;
use crate::array2_to_dense_matrix;

pub fn dataset_to_unsupervised_array(dataset: SCDataset<f32, f32>) -> Array2<f64> {
    Array2::from_shape_vec(
        (dataset.num_samples, dataset.num_features),
        dataset.data.iter().map(|elem| *elem as f64).collect(),
    )
    .unwrap()
    .to_owned()
}

type UnsupervisedCache = HashMap<TestSize, Array2<f64>>;

static UNSUPERVISED_DATA_CACHE: Lazy<UnsupervisedCache> = Lazy::new(|| {
    HashMap::from([
        (
            TestSize::Small,
            dataset_to_unsupervised_array(make_blobs(100, 10, 5)),
        ),
        (
            TestSize::Medium,
            dataset_to_unsupervised_array(make_blobs(1000, 50, 10)),
        ),
        (
            TestSize::Large,
            dataset_to_unsupervised_array(make_blobs(10000, 100, 20)),
        ),
    ])
});

fn unsupervised_cache(size: &TestSize) -> &'static Array2<f64> {
    UNSUPERVISED_DATA_CACHE
        .get(size)
        .expect("unsupervised data cache is populated for all test sizes")
}

pub fn x_unsupervised(size: &TestSize) -> Array2<f64> {
    unsupervised_cache(size).clone()
}

pub fn get_smartcore_unsupervised_data(size: &TestSize) -> DenseMatrix<f64> {
    let x = x_unsupervised(size);
    array2_to_dense_matrix(&x).expect("valid dense matrix conversion")
}

pub fn get_linfa_unsupervised_data(size: &TestSize) -> Dataset<f64, (), Ix1> {
    Dataset::from(x_unsupervised(size))
}

/// Run linfa kmeans
/// ```
/// use smartcore_vs_linfa::{get_linfa_unsupervised_data, linfa_kmeans, TestSize};
/// linfa_kmeans(&get_linfa_unsupervised_data(&TestSize::Small));
/// ```
pub fn linfa_kmeans(dataset: &Dataset<f64, (), Ix1>) {
    LinfaKMeans::params(5)
        .max_n_iterations(10)
        .n_runs(1)
        .fit(dataset)
        .unwrap();
}

/// Run smartcore kmeans
/// ```
/// use smartcore_vs_linfa::{get_smartcore_unsupervised_data, smartcore_kmeans, TestSize};
/// smartcore_kmeans(&get_smartcore_unsupervised_data(&TestSize::Small));
/// ```
pub fn smartcore_kmeans(x: &DenseMatrix<f64>) {
    let params = KMeansParameters::default().with_k(5).with_max_iter(10);
    SCKMeans::<f64, u32, DenseMatrix<f64>, Vec<u32>>::fit(x, params).unwrap();
}

/// Run linfa DBSCAN
/// ```
/// use smartcore_vs_linfa::{linfa_dbscan, TestSize, x_unsupervised};
/// linfa_dbscan(&x_unsupervised(&TestSize::Small));
/// ```
pub fn linfa_dbscan(dataset: &Array2<f64>) {
    let _ = LinfaDbscan::params(5).transform(dataset);
}

/// Run smartcore dbscan
/// ```
/// use smartcore_vs_linfa::{get_smartcore_unsupervised_data, smartcore_dbscan, TestSize};
/// smartcore_dbscan(&get_smartcore_unsupervised_data(&TestSize::Small));
/// ```
pub fn smartcore_dbscan(x: &DenseMatrix<f64>) {
    let params = DBSCANParameters::default()
        .with_min_samples(5)
        .with_eps(1e-4);
    SCDBSCAN::<f64, i32, DenseMatrix<f64>, Vec<i32>, Euclidian<f64>>::fit(x, params).unwrap();
}
