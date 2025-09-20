use std::fmt::{Display, Formatter};

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

use ndarray::Array2;
use rand::prelude::*;
use rand_distr::Normal;

use smartcore::{dataset::Dataset, error::Failed, linalg::basic::matrix::DenseMatrix};

pub fn make_regression(num_samples: usize, num_features: usize, noise: f32) -> Dataset<f32, f32> {
    let noise = Normal::new(0.0, noise).unwrap();
    let mut rng = rand::thread_rng();

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
