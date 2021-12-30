use criterion::{black_box, criterion_group, criterion_main, Criterion};

// A benchmark function for linear regression
fn linear_regression_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Linear Regression");
    bm.bench_function("Smartcore", |b| {
        let (x, y) = smartcore_vs_linfa::get_smartcore_regression_data();
        b.iter(|| smartcore_vs_linfa::smartcore_linear_regression(black_box(&x), black_box(&y)))
    });
    bm.bench_function("Linfa", |b| {
        let dataset = smartcore_vs_linfa::get_linfa_regression_data();
        b.iter(|| smartcore_vs_linfa::linfa_linear_regression(black_box(&dataset)))
    });
}

// A benchmark function for logistic regression
fn logistic_regression_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Logistic Regression");
    bm.bench_function("Smartcore", |b| {
        let (x, y) = smartcore_vs_linfa::get_smartcore_classification_data();
        b.iter(|| {
            smartcore_vs_linfa::smartcore_logistic_regression(
                black_box(&x),
                black_box(&(y.iter().map(|&elem| elem as f64).collect())),
            )
        })
    });
    bm.bench_function("Linfa", |b| {
        let dataset = smartcore_vs_linfa::get_linfa_classification_data();
        b.iter(|| smartcore_vs_linfa::linfa_logistic_regression(black_box(&dataset)))
    });
}

// A benchmark function for logistic regression
fn kmeans_clustering_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("K-Means Clustering");
    bm.bench_function("Smartcore", |b| {
        b.iter(|| smartcore_vs_linfa::smartcore_kmeans(black_box(20)))
    });
    bm.bench_function("Linfa", |b| {
        b.iter(|| smartcore_vs_linfa::linfa_kmeans(black_box(20)))
    });
}

// Create a criterion group with default settings
criterion_group!(
    benches,
    linear_regression_benchmark,
    logistic_regression_benchmark,
    kmeans_clustering_benchmark,
);

// Generate the benchmark harness
criterion_main!(benches);
