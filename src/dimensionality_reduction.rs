use linfa::prelude::*;
use linfa_reduction::{Pca as LinfaPca};
use smartcore::{
    linalg::naive::dense_matrix::DenseMatrix,
    decomposition::pca::{PCA as SCPCA, PCAParameters},
};


/// Run linfa pca
/// ```
/// use smartcore_vs_linfa::{get_linfa_unsupervised_data, linfa_pca, TestSize};
/// linfa_pca(&get_linfa_unsupervised_data(&TestSize::Small));
/// ```
pub fn linfa_pca(dataset: &Dataset<f64, ()>) {
    LinfaPca::params(3).fit(dataset);
}

/// Run smartcore pca
/// ```
/// use smartcore_vs_linfa::{get_smartcore_unsupervised_data, smartcore_pca, TestSize};
/// smartcore_pca(&get_smartcore_unsupervised_data(&TestSize::Small));
/// ```
pub fn smartcore_pca(x: &DenseMatrix<f64>) {
    SCPCA::fit(x, PCAParameters::default().with_n_components(3));
}