use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pdf_maker::distribution::Distribution1D;
use pdf_maker::inversion::Inversion1D;

use rand::{rngs::StdRng, Rng, SeedableRng};

fn inversion_1d_build(c: &mut Criterion) {
    c.bench_function("inversion_1d_build", |b| b.iter(|| Inversion1D::build(black_box(&[1.0; 1_500_000]))));
}

fn inversion_1d_sample(c: &mut Criterion) {
    let dist = Inversion1D::build(&[1.0; 1_000]);
    let mut rng = StdRng::seed_from_u64(0);
    c.bench_function("inversion_1d_sample", |b| b.iter(|| dist.sample(rng.gen())));
}

criterion_group!(benches, inversion_1d_build, inversion_1d_sample);
criterion_main!(benches);

