use std::fmt::{Display, Formatter};

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use smartcore::{dataset::Dataset, error::Failed, linalg::basic::matrix::DenseMatrix};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TestSize {
    Small,
    Medium,
    Large,
}

impl Display for TestSize {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TestSize::Small => write!(f, "Small"),
            TestSize::Medium => write!(f, "Medium"),
            TestSize::Large => write!(f, "Large"),
        }
    }
}

fn scenario_seed(test_size: TestSize, scenario: &str) -> u64 {
    match (scenario, test_size) {
        ("regression", TestSize::Small) => 0xA11C_E511,
        ("regression", TestSize::Medium) => 0xA11C_E522,
        ("regression", TestSize::Large) => 0xA11C_E533,
        ("classification", TestSize::Small) => 0xC1A5_51F1,
        ("classification", TestSize::Medium) => 0xC1A5_5202,
        ("classification", TestSize::Large) => 0xC1A5_5303,
        ("clustering", TestSize::Small) => 0xC1C7_0111,
        ("clustering", TestSize::Medium) => 0xC1C7_0222,
        ("clustering", TestSize::Large) => 0xC1C7_0333,
        _ => panic!(
            "no deterministic seed registered for scenario '{scenario}' and test size '{test_size}'"
        ),
    }
}

/// Create a reproducible random number generator for a benchmarking scenario.
///
/// The `scenario` parameter distinguishes between the data domains used in this
/// crate (`"regression"`, `"classification"`, and `"clustering"`). Each
/// combination of [`TestSize`] and scenario maps to a stable `u64` seed to
/// ensure that generated datasets remain deterministic across runs.
///
/// # Panics
///
/// Panics if `scenario` is not one of the supported dataset domains.
///
/// # Examples
/// ```
/// use rand::RngCore;
/// use smartcore_vs_linfa::{seeded_rng, TestSize};
///
/// let mut rng_a = seeded_rng(TestSize::Small, "regression");
/// let mut rng_b = seeded_rng(TestSize::Small, "regression");
/// assert_eq!(rng_a.next_u64(), rng_b.next_u64());
/// ```
#[must_use]
pub fn seeded_rng(test_size: TestSize, scenario: &str) -> StdRng {
    StdRng::seed_from_u64(scenario_seed(test_size, scenario))
}

/// Generate a deterministic linear regression dataset.
///
/// # Panics
///
/// Panics if `noise` is not strictly positive because the underlying normal
/// distribution cannot be constructed.
///
/// # Examples
/// ```
/// use smartcore_vs_linfa::{make_regression, TestSize};
///
/// let dataset = make_regression(TestSize::Small, 4, 2, 0.1);
/// assert_eq!(dataset.num_samples, 4);
/// assert_eq!(dataset.num_features, 2);
/// ```
#[must_use]
pub fn make_regression(
    test_size: TestSize,
    num_samples: usize,
    num_features: usize,
    noise: f32,
) -> Dataset<f32, f32> {
    let noise = Normal::new(0.0, noise).unwrap();
    let mut rng = seeded_rng(test_size, "regression");

    let mut x: Vec<f32> = Vec::with_capacity(num_samples * num_features);
    let mut y: Vec<f32> = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let mut yi: f32 = 1.0;
        for _ in 0..num_features {
            let xi = noise.sample(&mut rng);
            x.push(xi);
            yi += xi;
        }
        y.push(yi);
    }

    Dataset {
        data: x,
        target: y,
        num_samples,
        num_features,
        feature_names: (0..num_features).map(|n| n.to_string()).collect(),
        target_names: vec!["label".to_string()],
        description: "Linearly-correlated dataset with noise".to_string(),
    }
}

/// Convert an [`ndarray::Array2`] into a [`DenseMatrix`] with column-major storage.
///
/// # Errors
///
/// Returns [`Failed`] if the supplied array cannot be converted into a `DenseMatrix` due to
/// inconsistent dimensions. This should never happen when the input array is well-formed.
pub fn array2_to_dense_matrix(data: &Array2<f64>) -> Result<DenseMatrix<f64>, Failed> {
    let row_vectors: Vec<Vec<f64>> = data.rows().into_iter().map(|row| row.to_vec()).collect();
    DenseMatrix::from_2d_vec(&row_vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn regression_generation_is_deterministic() {
        let dataset_a = make_regression(TestSize::Small, 10, 2, 0.5);
        let dataset_b = make_regression(TestSize::Small, 10, 2, 0.5);

        assert_eq!(dataset_a.data, dataset_b.data);
        assert_eq!(dataset_a.target, dataset_b.target);
    }

    #[test]
    fn seeded_rng_reuses_the_same_sequence() {
        let mut rng_a = seeded_rng(TestSize::Medium, "regression");
        let mut rng_b = seeded_rng(TestSize::Medium, "regression");

        assert_eq!(rng_a.next_u64(), rng_b.next_u64());
    }

    #[test]
    fn seeded_rng_differs_across_scenarios() {
        let mut regression_rng = seeded_rng(TestSize::Small, "regression");
        let mut classification_rng = seeded_rng(TestSize::Small, "classification");

        assert_ne!(regression_rng.next_u64(), classification_rng.next_u64());
    }
}
