use criterion::{black_box, criterion_group, criterion_main, Criterion};
use discrete_sampling::distribution::Discrete1D;
use discrete_sampling::Alias1D;

use rand::{rngs::StdRng, Rng, SeedableRng};

fn alias_1d_build(c: &mut Criterion) {
    c.bench_function("alias_1d_build", |b| b.iter(|| { 
        let mut distr = [0.0; 10_000];
        for (i, weight) in distr.iter_mut().enumerate() {
            *weight = (i + 1) as f32;
        }
        <Alias1D<f32> as Discrete1D<f32>>::build(black_box(&distr))
    }));
}

fn alias_1d_sample(c: &mut Criterion) {
    let mut distr = [0.0; 100];
    for (i, weight) in distr.iter_mut().enumerate() {
        *weight = (i + 1) as f32;
    }
    let dist = <Alias1D<f32> as Discrete1D<f32>>::build(&distr);
    let mut rng = StdRng::seed_from_u64(0);
    c.bench_function("alias_1d_sample", |b| b.iter(|| dist.sample(rng.gen::<f32>())));
}

criterion_group!(benches, alias_1d_build, alias_1d_sample);
criterion_main!(benches);

