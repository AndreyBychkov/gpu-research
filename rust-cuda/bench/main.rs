#![feature(test)]

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput, black_box, Bencher};
use std::time::Duration;
use nanorand::{Rng, WyRand};





const MEASUREMENT_SECS: u64 = 10;

fn mul_bench(ben: &mut Bencher, n: usize) {

    let mut wyrand = WyRand::new();
    let mut lhs = vec![2.0f32; n];
    wyrand.fill(&mut lhs);
    let mut rhs = vec![0.0f32; n];
    wyrand.fill(&mut rhs);
    // let mut out = vec![0.0; n * n];
    ben.iter(|| {
        black_box(matmul_gpu(&lhs, &rhs));
    })
}


fn mul_run(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("mul");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let ns = [21000*21000];
    for n in ns.iter() {
        group.bench_with_input(BenchmarkId::new("cuda", n), n, |ben, &n| mul_bench(ben, n));
    }
    group.finish();
}

criterion_group!(benches, mul_run);
criterion_main!(benches);



