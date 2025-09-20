use linfa::prelude::*;
use linfa_clustering::{Dbscan as LinfaDbscan, KMeans as LinfaKMeans};
use ndarray::{Array2, Ix1};
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use smartcore::{
    cluster::{
        dbscan::{DBSCANParameters, DBSCAN as SCDBSCAN},
        kmeans::{KMeans as SCKMeans, KMeansParameters},
    },
    dataset::Dataset as SCDataset,
    linalg::basic::matrix::DenseMatrix,
    metrics::distance::euclidian::Euclidian,
};
use std::{collections::HashMap, sync::LazyLock};

use super::{seeded_rng, TestSize};
use crate::array2_to_dense_matrix;

/// Convert a `SmartCore` unsupervised dataset into a dense ndarray matrix.
///
/// # Panics
///
/// Panics if the dataset metadata does not align with the length of the
/// flattened feature vector.
#[must_use]
pub fn dataset_to_unsupervised_array(dataset: &SCDataset<f32, f32>) -> Array2<f64> {
    Array2::from_shape_vec(
        (dataset.num_samples, dataset.num_features),
        dataset.data.iter().map(|elem| f64::from(*elem)).collect(),
    )
    .unwrap()
}

type UnsupervisedCache = HashMap<TestSize, Array2<f64>>;

static UNSUPERVISED_DATA_CACHE: LazyLock<UnsupervisedCache> = LazyLock::new(|| {
    HashMap::from([
        (TestSize::Small, build_unsupervised_data(TestSize::Small)),
        (TestSize::Medium, build_unsupervised_data(TestSize::Medium)),
        (TestSize::Large, build_unsupervised_data(TestSize::Large)),
    ])
});

#[allow(clippy::cast_precision_loss)]
fn make_seeded_blobs(
    rng: &mut StdRng,
    num_samples: usize,
    num_features: usize,
    num_centers: usize,
) -> SCDataset<f32, f32> {
    let center_box = Uniform::from(-10.0..10.0);
    let cluster_std = 1.0;
    let mut centers: Vec<Vec<Normal<f32>>> = Vec::with_capacity(num_centers);

    for _ in 0..num_centers {
        centers.push(
            (0..num_features)
                .map(|_| Normal::new(center_box.sample(rng), cluster_std).unwrap())
                .collect(),
        );
    }

    let mut y: Vec<f32> = Vec::with_capacity(num_samples);
    let mut x: Vec<f32> = Vec::with_capacity(num_samples * num_features);

    for sample_index in 0..num_samples {
        let label = sample_index % num_centers;
        y.push(label as f32);
        for feature_index in 0..num_features {
            x.push(centers[label][feature_index].sample(rng));
        }
    }

    SCDataset {
        data: x,
        target: y,
        num_samples,
        num_features,
        feature_names: (0..num_features).map(|n| n.to_string()).collect(),
        target_names: vec!["label".to_string()],
        description: "Isotropic Gaussian blobs".to_string(),
    }
}

fn build_unsupervised_data(size: TestSize) -> Array2<f64> {
    let (num_samples, num_features, num_centers) = match size {
        TestSize::Small => (100, 10, 5),
        TestSize::Medium => (1000, 50, 10),
        TestSize::Large => (10000, 100, 20),
    };

    let mut rng = seeded_rng(size, "clustering");
    let dataset = make_seeded_blobs(&mut rng, num_samples, num_features, num_centers);
    dataset_to_unsupervised_array(&dataset)
}

fn unsupervised_cache(size: TestSize) -> &'static Array2<f64> {
    UNSUPERVISED_DATA_CACHE
        .get(&size)
        .expect("unsupervised data cache is populated for all test sizes")
}

#[must_use]
pub fn x_unsupervised(size: &TestSize) -> Array2<f64> {
    unsupervised_cache(*size).clone()
}

/// Retrieve `SmartCore`-compatible unsupervised inputs.
///
/// # Panics
///
/// Panics if the cached ndarray cannot be converted into a dense matrix.
#[must_use]
pub fn get_smartcore_unsupervised_data(size: &TestSize) -> DenseMatrix<f64> {
    let x = x_unsupervised(size);
    array2_to_dense_matrix(&x).expect("valid dense matrix conversion")
}

#[must_use]
pub fn get_linfa_unsupervised_data(size: &TestSize) -> Dataset<f64, (), Ix1> {
    Dataset::from(x_unsupervised(size))
}

/// Run linfa kmeans
///
/// # Panics
///
/// Panics if the Linfa K-Means solver fails to converge.
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
///
/// # Panics
///
/// Panics if the `SmartCore` K-Means solver fails to converge.
/// ```
/// use smartcore_vs_linfa::{get_smartcore_unsupervised_data, smartcore_kmeans, TestSize};
/// smartcore_kmeans(&get_smartcore_unsupervised_data(&TestSize::Small));
/// ```
pub fn smartcore_kmeans(x: &DenseMatrix<f64>) {
    let params = KMeansParameters::default().with_k(5).with_max_iter(10);
    SCKMeans::<f64, u32, DenseMatrix<f64>, Vec<u32>>::fit(x, params).unwrap();
}

/// Run linfa DBSCAN
///
/// # Panics
///
/// Panics if Linfa DBSCAN fails to process the dataset.
/// ```
/// use smartcore_vs_linfa::{linfa_dbscan, TestSize, x_unsupervised};
/// linfa_dbscan(&x_unsupervised(&TestSize::Small));
/// ```
pub fn linfa_dbscan(dataset: &Array2<f64>) {
    let _ = LinfaDbscan::params(5).transform(dataset);
}

/// Run smartcore dbscan
///
/// # Panics
///
/// Panics if `SmartCore` DBSCAN fails to process the dataset.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupervised_data_is_deterministic() {
        let data_a = super::build_unsupervised_data(TestSize::Small);
        let data_b = super::build_unsupervised_data(TestSize::Small);

        assert_eq!(data_a, data_b);
    }

    #[test]
    fn unsupervised_data_changes_with_size() {
        let small = super::build_unsupervised_data(TestSize::Small);
        let medium = super::build_unsupervised_data(TestSize::Medium);

        assert_ne!(small, medium);
    }
}
