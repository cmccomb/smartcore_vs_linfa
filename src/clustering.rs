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

// pub fn x_clustering_original(_size: &TestSize) -> Array2<f64> {
//     array![
//         [5.1, 3.5, 1.4, 0.2],
//         [4.9, 3.0, 1.4, 0.2],
//         [4.7, 3.2, 1.3, 0.2],
//         [4.6, 3.1, 1.5, 0.2],
//         [5.0, 3.6, 1.4, 0.2],
//         [5.4, 3.9, 1.7, 0.4],
//         [4.6, 3.4, 1.4, 0.3],
//         [5.0, 3.4, 1.5, 0.2],
//         [4.4, 2.9, 1.4, 0.2],
//         [4.9, 3.1, 1.5, 0.1],
//         [7.0, 3.2, 4.7, 1.4],
//         [6.4, 3.2, 4.5, 1.5],
//         [6.9, 3.1, 4.9, 1.5],
//         [5.5, 2.3, 4.0, 1.3],
//         [6.5, 2.8, 4.6, 1.5],
//         [5.7, 2.8, 4.5, 1.3],
//         [6.3, 3.3, 4.7, 1.6],
//         [4.9, 2.4, 3.3, 1.0],
//         [6.6, 2.9, 4.6, 1.3],
//         [5.2, 2.7, 3.9, 1.4],
//     ]
//     .to_owned()
// }

pub fn x_clustering(size: &TestSize) -> Array2<f64> {
    match size {
        TestSize::Small => dataset_to_clustering_array(make_blobs(100, 10, 5)),
        TestSize::Medium => dataset_to_clustering_array(make_blobs(1000, 50, 10)),
        TestSize::Large => dataset_to_clustering_array(make_blobs(10000, 100, 20)),
    }
}

pub fn dataset_to_clustering_array(dataset: SCDataset<f32, f32>) -> Array2<f64> {
    Array2::from_shape_vec(
        (dataset.num_samples, dataset.num_features),
        dataset.data.iter().map(|elem| *elem as f64).collect(),
    )
        .unwrap()
        .to_owned()
}

pub fn get_smartcore_clustering_data(size: &TestSize) -> DenseMatrix<f64> {
    let x = x_clustering(size).to_owned();
    DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap())
}

pub fn get_linfa_clustering_data(size: &TestSize) -> Dataset<f64, ()> {
    Dataset::from(x_clustering(size).to_owned())
}

pub fn linfa_kmeans(dataset: &Dataset<f64, ()>) {
    let _model = LinfaKMeans::params(5)
        .max_n_iterations(10)
        .n_runs(1)
        .fit(dataset);
}

/// Run smartcore dbscan
/// ```
/// use smartcore_vs_linfa::{get_smartcore_clustering_data, smartcore_kmeans, TestSize};
/// smartcore_kmeans(&get_smartcore_clustering_data(&TestSize::Small));
/// ```

pub fn smartcore_kmeans(x: &DenseMatrix<f64>) -> SCKMeans<f64> {
    SCKMeans::fit(x, KMeansParameters::default().with_k(5).with_max_iter(10)).unwrap()
}

pub fn linfa_dbscan(dataset: &Array2<f64>) {
    let _model = LinfaDbscan::params(5).transform(dataset);
}

/// Run smartcore dbscan
/// ```
/// use smartcore_vs_linfa::{get_smartcore_clustering_data, smartcore_dbscan, TestSize};
/// smartcore_dbscan(&get_smartcore_clustering_data(&TestSize::Small));
/// ```
pub fn smartcore_dbscan(x: &DenseMatrix<f64>) -> SCDBSCAN<f64, Euclidian> {
    SCDBSCAN::fit(
        x,
        DBSCANParameters::default()
            .with_min_samples(5)
            .with_eps(1e-4),
    )
        .unwrap()
}
