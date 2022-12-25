use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pdf_maker::distribution::Distribution1D;
use pdf_maker::alias::Alias1D;

fn alias1d(c: &mut Criterion) {
    c.bench_function("alias1d", |b| b.iter(|| Alias1D::build(&[1.0; 1_000_000])));
}

criterion_group!(benches, alias1d);
criterion_main!(benches);
