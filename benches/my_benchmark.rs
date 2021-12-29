use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smartcore_vs_linfa::{linfa_linear_regression, smartcore_linear_regression};

fn criterion_benchmark(c: &mut Criterion) {
    let mut linear_regression = c.benchmark_group("linear_regression");
    linear_regression.bench_function("Smartcore", |b| {
        b.iter(|| smartcore_linear_regression(black_box(20)))
    });
    linear_regression.bench_function("Linfa", |b| {
        b.iter(|| linfa_linear_regression(black_box(20)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
