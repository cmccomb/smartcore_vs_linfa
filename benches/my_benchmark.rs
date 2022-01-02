use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, PlotConfiguration,
};
use smartcore_vs_linfa::TestSize;

// A benchmark function for linear regression
fn linear_regression_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Linear Regression");
    bm.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for test_size in [TestSize::Small, TestSize::Medium, TestSize::Large].iter() {
        bm.bench_function(format!("{}/Smart", test_size), |b| {
            let (x, y) = smartcore_vs_linfa::get_smartcore_regression_data(test_size);
            b.iter(|| smartcore_vs_linfa::smartcore_linear_regression(black_box(&x), black_box(&y)))
        });
        bm.bench_function(format!("{}/Linfa", test_size), |b| {
            let dataset = smartcore_vs_linfa::get_linfa_regression_data(test_size);
            b.iter(|| smartcore_vs_linfa::linfa_linear_regression(black_box(&dataset)))
        });
    }
}

// A benchmark function for elastic net regression
fn elasticnet_regression_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Elastic Net");
    bm.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for test_size in [TestSize::Small, TestSize::Medium, TestSize::Large].iter() {
        bm.bench_function(format!("{}/Smart", test_size), |b| {
            let (x, y) = smartcore_vs_linfa::get_smartcore_regression_data(test_size);
            b.iter(|| {
                smartcore_vs_linfa::smartcore_elasticnet_regression(black_box(&x), black_box(&y))
            })
        });
        bm.bench_function(format!("{}/Linfa", test_size), |b| {
            let dataset = smartcore_vs_linfa::get_linfa_regression_data(test_size);
            b.iter(|| smartcore_vs_linfa::linfa_elasticnet_regression(black_box(&dataset)))
        });
    }
}

// A benchmark function for logistic regression
fn svr_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Support Vector Regression");
    bm.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for test_size in [TestSize::Small, TestSize::Medium].iter() {
        bm.bench_function(format!("{}/Smart", test_size), |b| {
            let (x, y) = smartcore_vs_linfa::get_smartcore_regression_data(test_size);
            b.iter(|| smartcore_vs_linfa::smartcore_svm_regression(black_box(&x), black_box(&y)))
        });
        bm.bench_function(format!("{}/Linfa", test_size), |b| {
            let dataset = smartcore_vs_linfa::get_linfa_regression_data(test_size);
            b.iter(|| smartcore_vs_linfa::linfa_svm_regression(black_box(&dataset)))
        });
    }
}

// A benchmark function for logistic regression
fn logistic_regression_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Logistic Regression");
    bm.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for test_size in [TestSize::Small, TestSize::Medium, TestSize::Large].iter() {
        bm.bench_function(format!("{}/Smart", test_size), |b| {
            let (x, y) = smartcore_vs_linfa::get_smartcore_classification_data(test_size);
            b.iter(|| {
                smartcore_vs_linfa::smartcore_logistic_regression(black_box(&x), black_box(&y))
            })
        });
        bm.bench_function(format!("{}/Linfa", test_size), |b| {
            let dataset = smartcore_vs_linfa::get_linfa_classification_data(test_size);
            b.iter(|| smartcore_vs_linfa::linfa_logistic_regression(black_box(&dataset)))
        });
    }
}

// A benchmark function for logistic regression
fn decision_tree_classifier_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Decision Tree Classification");
    bm.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for test_size in [TestSize::Small, TestSize::Medium, TestSize::Large].iter() {
        bm.bench_function(format!("{}/Smart", test_size), |b| {
            let (x, y) = smartcore_vs_linfa::get_smartcore_classification_data(test_size);
            b.iter(|| {
                smartcore_vs_linfa::smartcore_decision_tree_classifier(black_box(&x), black_box(&y))
            })
        });
        bm.bench_function(format!("{}/Linfa", test_size), |b| {
            let dataset = smartcore_vs_linfa::get_linfa_classification_data(test_size);
            b.iter(|| smartcore_vs_linfa::linfa_decision_tree_classifier(black_box(&dataset)))
        });
    }
}

// A benchmark function for logistic regression
fn gnb_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Gaussian Naive Bayes");
    bm.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for test_size in [TestSize::Small, TestSize::Medium, TestSize::Large].iter() {
        bm.bench_function(format!("{}/Smart", test_size), |b| {
            let (x, y) = smartcore_vs_linfa::get_smartcore_classification_data(test_size);
            b.iter(|| smartcore_vs_linfa::smartcore_gnb_classifier(black_box(&x), black_box(&y)))
        });
        bm.bench_function(format!("{}/Linfa", test_size), |b| {
            let dataset = smartcore_vs_linfa::get_linfa_classification_data(test_size);
            b.iter(|| smartcore_vs_linfa::linfa_gnb_classifier(black_box(&dataset)))
        });
    }
}

// A benchmark function for logistic regression
fn svc_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Support Vector Classification");
    bm.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for test_size in [TestSize::Small, TestSize::Medium].iter() {
        bm.bench_function(format!("{}/Smart", test_size), |b| {
            let (x, y) = smartcore_vs_linfa::get_smartcore_classification_data(test_size);
            b.iter(|| smartcore_vs_linfa::smartcore_svm_classifier(black_box(&x), black_box(&y)))
        });
        bm.bench_function(format!("{}/Linfa", test_size), |b| {
            let dataset = smartcore_vs_linfa::get_linfa_classification_data_as_bool(test_size);
            b.iter(|| smartcore_vs_linfa::linfa_svm_classifier(black_box(&dataset)))
        });
    }
}

// A benchmark function for kmeans clustering
fn kmeans_clustering_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("K-Means Clustering");
    bm.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for test_size in [TestSize::Small, TestSize::Medium, TestSize::Large].iter() {
        bm.bench_function(format!("{}/Smart", test_size), |b| {
            let x = smartcore_vs_linfa::get_smartcore_unsupervised_data(test_size);
            b.iter(|| smartcore_vs_linfa::smartcore_kmeans(black_box(&x)))
        });
        bm.bench_function(format!("{}/Linfa", test_size), |b| {
            let dataset = smartcore_vs_linfa::get_linfa_unsupervised_data(test_size);
            b.iter(|| smartcore_vs_linfa::linfa_kmeans(black_box(&dataset)))
        });
    }
}

// A benchmark function for kmeans clustering
fn dbscan_clustering_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("DBSCAN Clustering");
    bm.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for test_size in [TestSize::Small, TestSize::Medium, TestSize::Large].iter() {
        bm.bench_function(format!("{}/Smart", test_size), |b| {
            let x = smartcore_vs_linfa::get_smartcore_unsupervised_data(test_size);
            b.iter(|| smartcore_vs_linfa::smartcore_dbscan(black_box(&x)))
        });
        bm.bench_function(format!("{}/Linfa", test_size), |b| {
            let dataset = smartcore_vs_linfa::x_unsupervised(test_size);
            b.iter(|| smartcore_vs_linfa::linfa_dbscan(black_box(&dataset)))
        });
    }
}

// A benchmark function for kmeans clustering
fn pca_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("PCA");
    bm.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for test_size in [TestSize::Small, TestSize::Medium, TestSize::Large].iter() {
        bm.bench_function(format!("{}/Smart", test_size), |b| {
            let x = smartcore_vs_linfa::get_smartcore_unsupervised_data(test_size);
            b.iter(|| smartcore_vs_linfa::smartcore_pca(black_box(&x)))
        });
        bm.bench_function(format!("{}/Linfa", test_size), |b| {
            let dataset = smartcore_vs_linfa::get_linfa_unsupervised_data(test_size);
            b.iter(|| smartcore_vs_linfa::linfa_pca(black_box(&dataset)))
        });
    }
}

// Create a criterion group with default settings
criterion_group!(
    benches,
    linear_regression_benchmark,
    elasticnet_regression_benchmark,
    svr_benchmark,
    logistic_regression_benchmark,
    decision_tree_classifier_benchmark,
    gnb_benchmark,
    svc_benchmark,
    kmeans_clustering_benchmark,
    dbscan_clustering_benchmark,
    pca_benchmark,
);

// Generate the benchmark harness
criterion_main!(benches);
