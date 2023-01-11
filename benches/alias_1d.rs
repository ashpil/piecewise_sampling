use criterion::{black_box, criterion_group, criterion_main, Criterion};
use discrete_sampling::distribution::Discrete1D;
use discrete_sampling::Alias1D;

use rand::{rngs::StdRng, Rng, SeedableRng};

fn alias_1d_build(c: &mut Criterion) {
    c.bench_function("alias_1d_build", |b| b.iter(|| Alias1D::build(black_box(&[1.0; 1_500_000]))));
}

fn alias_1d_sample(c: &mut Criterion) {
    let dist = Alias1D::build(&[1.0; 1_000]);
    let mut rng = StdRng::seed_from_u64(0);
    c.bench_function("alias_1d_sample", |b| b.iter(|| dist.sample(rng.gen())));
}

criterion_group!(benches, alias_1d_build, alias_1d_sample);
criterion_main!(benches);

