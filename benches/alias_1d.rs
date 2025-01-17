use criterion::{black_box, criterion_group, criterion_main, Criterion};
use discrete_sampling::distribution::Discrete1D;
use discrete_sampling::Alias1D;

use rand::{rngs::StdRng, Rng, SeedableRng};

fn alias_1d_build(c: &mut Criterion) {
    let mut distr = [0.0; 10_000];
    for (i, weight) in distr.iter_mut().enumerate() {
        *weight = (i + 1) as f32;
    }
    c.bench_function("alias_1d_build", |b| b.iter(|| { 
        <Alias1D<f32> as Discrete1D<f32>>::build(black_box(&distr))
    }));
}

fn alias_1d_sample_coherent(c: &mut Criterion) {
    let mut distr = [0.0; 100];
    for (i, weight) in distr.iter_mut().enumerate() {
        *weight = (i + 1) as f32;
    }
    let dist = <Alias1D<f32> as Discrete1D<f32>>::build(&distr);
    c.bench_function("alias_1d_sample_coherent", |b| b.iter(|| {
        let sample_count = 1000;
        for i in 0..sample_count {
            let i = i as f32 / sample_count as f32;
            dist.sample(black_box(i));
        }
    }));
}

fn alias_1d_sample_incoherent(c: &mut Criterion) {
    let mut distr = [0.0; 100];
    for (i, weight) in distr.iter_mut().enumerate() {
        *weight = (i + 1) as f32;
    }
    let dist = <Alias1D<f32> as Discrete1D<f32>>::build(&distr);
    let mut rng = StdRng::seed_from_u64(0);
    c.bench_function("alias_1d_sample_incoherent", |b| b.iter(|| {
        for _ in 0..1000 {
            let i = rng.r#gen::<f32>();
            dist.sample(black_box(i));
        }
    }));
}

criterion_group!(benches, alias_1d_build, alias_1d_sample_coherent, alias_1d_sample_incoherent);
criterion_main!(benches);

