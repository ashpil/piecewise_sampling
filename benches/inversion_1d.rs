use criterion::{black_box, criterion_group, criterion_main, Criterion};
use discrete_sampling::distribution::Discrete1D;
use discrete_sampling::Inversion1D;

use rand::{rngs::StdRng, Rng, SeedableRng};

fn inversion_1d_build(c: &mut Criterion) {
    c.bench_function("inversion_1d_build", |b| b.iter(|| <Inversion1D<f32> as Discrete1D<f32>>::build(black_box(&[1.0; 1_500_000]))));
}

fn inversion_1d_sample(c: &mut Criterion) {
    let dist = <Inversion1D<f32> as Discrete1D<f32>>::build(&[1.0; 1_000]);
    let mut rng = StdRng::seed_from_u64(0);
    c.bench_function("inversion_1d_sample", |b| b.iter(|| dist.sample(rng.r#gen::<f32>())));
}

criterion_group!(benches, inversion_1d_build, inversion_1d_sample);
criterion_main!(benches);

