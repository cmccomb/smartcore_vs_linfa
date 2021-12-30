use criterion::{black_box, criterion_group, criterion_main, Criterion};

// A benchmark function for linear regression
fn linear_regression_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Linear Regression");
    bm.bench_function("Smartcore", |b| {
        b.iter(|| smartcore_vs_linfa::smartcore_linear_regression(black_box(20)))
    });
    bm.bench_function("Linfa", |b| {
        b.iter(|| smartcore_vs_linfa::linfa_linear_regression(black_box(20)))
    });
}

// A benchmark function for logistic regression
fn logistic_regression_benchmark(c: &mut Criterion) {
    let mut bm = c.benchmark_group("Logistic Regression");
    bm.bench_function("Smartcore", |b| {
        b.iter(|| smartcore_vs_linfa::smartcore_logistic_regression(black_box(20)))
    });
    bm.bench_function("Linfa", |b| {
        b.iter(|| smartcore_vs_linfa::linfa_logistic_regression(black_box(20)))
    });
}

// Create a criterion group with default settings
criterion_group!(
    benches,
    linear_regression_benchmark,
    logistic_regression_benchmark
);

// Generate the benchmark harness
criterion_main!(benches);
