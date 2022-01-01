use std::fmt::{Display, Formatter};

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


//! # Dataset Generators
//!
use rand::distributions::Uniform;
use rand::prelude::*;
use rand_distr::Normal;

use smartcore::dataset::Dataset;

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