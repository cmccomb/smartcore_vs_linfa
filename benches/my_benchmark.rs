use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, PlotConfiguration,
};

// A benchmark function for linear regression
fn linear_regression_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Linear Regression");
    bm.bench_function("Smartcore", |b| {
        let (x, y) =
            smartcore_vs_linfa::get_smartcore_regression_data(&smartcore_vs_linfa::TestSize::Small);
        b.iter(|| smartcore_vs_linfa::smartcore_linear_regression(black_box(&x), black_box(&y)))
    });
    bm.bench_function("Linfa", |b| {
        let dataset =
            smartcore_vs_linfa::get_linfa_regression_data(&smartcore_vs_linfa::TestSize::Small);
        b.iter(|| smartcore_vs_linfa::linfa_linear_regression(black_box(&dataset)))
    });
}

// A benchmark function for elastic net regression
fn elasticnet_regression_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Elastic Net");
    bm.bench_function("Smartcore", |b| {
        let (x, y) =
            smartcore_vs_linfa::get_smartcore_regression_data(&smartcore_vs_linfa::TestSize::Small);
        b.iter(|| smartcore_vs_linfa::smartcore_elasticnet_regression(black_box(&x), black_box(&y)))
    });
    bm.bench_function("Linfa", |b| {
        let dataset =
            smartcore_vs_linfa::get_linfa_regression_data(&smartcore_vs_linfa::TestSize::Small);
        b.iter(|| smartcore_vs_linfa::linfa_elasticnet_regression(black_box(&dataset)))
    });
}

// A benchmark function for logistic regression
fn logistic_regression_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Logistic Regression");
    bm.bench_function("Smartcore", |b| {
        let (x, y) = smartcore_vs_linfa::get_smartcore_classification_data(
            &smartcore_vs_linfa::TestSize::Small,
        );
        b.iter(|| {
            smartcore_vs_linfa::smartcore_logistic_regression(
                black_box(&x),
                black_box(&(y.iter().map(|&elem| elem as f64).collect())),
            )
        })
    });
    bm.bench_function("Linfa", |b| {
        let dataset =
            smartcore_vs_linfa::get_linfa_classification_data(&smartcore_vs_linfa::TestSize::Small);
        b.iter(|| smartcore_vs_linfa::linfa_logistic_regression(black_box(&dataset)))
    });
}

// A benchmark function for kmeans clustering
fn kmeans_clustering_benchmark(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut bm = c.benchmark_group("K-Means Clustering");
    // bm.plot_config(plot_config);
    bm.bench_function("Smartcore", |b| {
        let x =
            smartcore_vs_linfa::get_smartcore_clustering_data(&smartcore_vs_linfa::TestSize::Small);
        b.iter(|| smartcore_vs_linfa::smartcore_kmeans(black_box(&x)))
    });
    bm.bench_function("Linfa", |b| {
        let dataset =
            smartcore_vs_linfa::get_linfa_clustering_data(&smartcore_vs_linfa::TestSize::Small);
        b.iter(|| smartcore_vs_linfa::linfa_kmeans(black_box(&dataset)))
    });
}

// A benchmark function for kmeans clustering
fn dbscan_clustering_benchmark(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut bm = c.benchmark_group("DBSCAN Clustering");
    // bm.plot_config(plot_config);
    bm.bench_function("Smartcore", |b| {
        let x =
            smartcore_vs_linfa::get_smartcore_clustering_data(&smartcore_vs_linfa::TestSize::Small);
        b.iter(|| smartcore_vs_linfa::smartcore_dbscan(black_box(&x)))
    });
    bm.bench_function("Linfa", |b| {
        // let dataset =
        //     smartcore_vs_linfa::get_linfa_clustering_data(&smartcore_vs_linfa::TestSize::Small);
        let dataset = smartcore_vs_linfa::x_clustering(&smartcore_vs_linfa::TestSize::Small);
        b.iter(|| smartcore_vs_linfa::linfa_dbscan(black_box(&dataset)))
    });
}

// Create a criterion group with default settings
criterion_group!(
    benches,
    linear_regression_benchmark,
    elasticnet_regression_benchmark,
    logistic_regression_benchmark,
    svm_benchmark,
    kmeans_clustering_benchmark,
    dbscan_clustering_benchmark,
);

// Generate the benchmark harness
criterion_main!(benches);
