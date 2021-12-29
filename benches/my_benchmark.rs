use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smartcore_vs_linfa::{fibonacci1, fibonacci2};

fn criterion_benchmark(c: &mut Criterion) {
    let mut linear_regression = c.benchmark_group("linear_regression");
    linear_regression.bench_function("Bad Fib", |b| b.iter(|| fibonacci1(black_box(20))));
    linear_regression.bench_function("Good Fib", |b| b.iter(|| fibonacci2(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
