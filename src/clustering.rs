use linfa::prelude::*;
use linfa_clustering::{Dbscan as LinfaDbscan, KMeans as LinfaKMeans};
use ndarray::{array, Array2};
use smartcore::{
    cluster::{
        dbscan::{DBSCANParameters, DBSCAN as SCDBSCAN},
        kmeans::{KMeans as SCKMeans, KMeansParameters},
    },
    linalg::naive::dense_matrix::DenseMatrix,
};

use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;

use super::TestSize;

pub fn x_clustering(_size: &TestSize) -> Array2<f64> {
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

// pub fn x_clustering2(_size: &TestSize) -> Array2<f64> {
//     let mut rng = Isaac64Rng::seed_from_u64(42);
//     let cent = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.]];
//     let x = generate_blobs(100, &cent, &mut rng);
//     x
// }

pub fn get_smartcore_clustering_data(size: &TestSize) -> DenseMatrix<f64> {
    let x = x_clustering(size).to_owned();
    DenseMatrix::from_array(x.shape()[0], x.shape()[1], x.as_slice().unwrap())
}

pub fn get_linfa_clustering_data(size: &TestSize) -> Dataset<f64, ()> {
    Dataset::from(x_clustering(size).to_owned())
}

pub fn linfa_kmeans(dataset: &Dataset<f64, ()>) {
    let _model = LinfaKMeans::params(2)
        .max_n_iterations(10)
        .n_runs(1)
        .fit(dataset);
}

/// Run smartcore dbscan
/// ```
/// use smartcore_vs_linfa::{get_smartcore_clustering_data, smartcore_kmeans, TestSize};
/// smartcore_kmeans(&get_smartcore_clustering_data(&TestSize::Small));
/// ```

pub fn smartcore_kmeans(x: &DenseMatrix<f64>) {
    let _model = SCKMeans::fit(x, KMeansParameters::default().with_k(2).with_max_iter(10));
}

pub fn linfa_dbscan(dataset: &Array2<f64>) {
    let _model = LinfaDbscan::params(2).transform(dataset);
}

/// Run smartcore dbscan
/// ```
/// use smartcore_vs_linfa::{get_smartcore_clustering_data, smartcore_dbscan, TestSize};
/// smartcore_dbscan(&get_smartcore_clustering_data(&TestSize::Small));
/// ```
pub fn smartcore_dbscan(x: &DenseMatrix<f64>) {
    let _model = SCDBSCAN::fit(
        x,
        DBSCANParameters::default()
            .with_min_samples(2)
            .with_eps(1e-4),
    );
}
