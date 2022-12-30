use crate::data2d::Data2D;

// 1D piecewise constant distribution
pub trait Distribution1D {
    // constructor
    fn build(weights: &[f32]) -> Self;

    // takes in rand [0-1), returns (pdf, sampled idx)
    fn sample_discrete(&self, u: f32) -> (f32, usize);

    // takes in rand [0-1), returns (pdf, sampled [0-1))
    fn sample_continuous(&self, u: f32) -> (f32, f32);

    // takes in coord, returns pdf
    fn pdf(&self, u: usize) -> f32;

    // sum of all weights
    fn integral(&self) -> f32;

    // range of sampled idxs, should be len of weights
    fn size(&self) -> usize;
}

// 2D piecewise constant distribution
pub trait Distribution2D {
    // constructor
    fn build(weights: &Data2D<f32>) -> Self;

    // takes in rand [0-1)x[0-1), returns (pdf, sampled uv coords)
    fn sample_discrete(&self, uv: [f32; 2]) -> (f32, [usize; 2]);

    // takes in coords, returns pdf
    fn pdf(&self, uv: [usize; 2]) -> f32;

    fn width(&self) -> usize;
    fn height(&self) -> usize;

    // fills demo image with sample_count samples
    fn fill_demo_image(&self, demo: &mut Data2D<[f32; 3]>, rngs: impl Iterator<Item = [f32; 2]>) {
        for rng in rngs {
            let (_, [x, y]) = self.sample_discrete(rng);
            for is in -1..1 {
                for js in -1..1 {
                    let j = (y as i32 + js).clamp(0, (demo.height() - 1) as i32) as usize;
                    let i = (x as i32 + is).clamp(0, (demo.width() - 1) as i32) as usize;
                    demo[j][i] = [1000.0, 0.0, 0.0];
                }
            }
        }
    }
}

#[cfg(test)]
use {
    rand::{rngs::StdRng, Rng, SeedableRng},
    statrs::distribution::{ChiSquared, ContinuousCDF},
};

#[cfg(test)]
pub fn chisq_distribution_1d<D: Distribution1D>(expected: &[f32], sample_count: usize) {
    let dist = D::build(expected);
    let mut observed = vec![0.0f32; expected.len()].into_boxed_slice();
    let mut hist = vec![0u32; expected.len()].into_boxed_slice();
    let mut rng = StdRng::seed_from_u64(0);

    for _ in 0..sample_count {
        let (pdf, idx) = dist.sample_discrete(rng.gen());
        assert!((expected[idx] - pdf).abs() < 0.001);
        assert_eq!(pdf, dist.pdf(idx));
        hist[idx] += 1;
    }

    for (weight, obs) in hist.into_iter().zip(observed.iter_mut()) {
        *obs = (*weight as f32 / sample_count as f32) * dist.integral();
    }

    let mut chsq = 0.0;
    for (obs, exp) in observed.into_iter().zip(expected) {
        let diff = obs - exp;
        chsq += diff * diff / exp;
    }

    let pval = 1.0 - ChiSquared::new((expected.len() - 1) as f64).unwrap().cdf(chsq as f64);
    assert!(pval >= 0.99, "failed chi-squared statistical test, p = {}", pval);
}

#[cfg(test)]
macro_rules! distribution_1d_tests {
    ($impl:path) => {
        mod distribution1d {
            use crate::distribution::{
                Distribution1D,
                chisq_distribution_1d,
            };
            use $impl as Dist;

            #[test]
            fn basic() {
                chisq_distribution_1d::<Dist>(&[1.0, 1.0, 2.0, 4.0, 8.0], 1000);
            }

            #[test]
            fn uniform() {
                chisq_distribution_1d::<Dist>(&[1.0; 10_000], 1_000_000);
            }

            #[test]
            fn increasing() {
                let mut distr = [0.0; 100];
                for (i, weight) in distr.iter_mut().enumerate() {
                    *weight = (5 * (i + 1)) as f32;
                }
                chisq_distribution_1d::<Dist>(&distr, 100_000);
            }

            #[test]
            fn surjective() {
                let dist = Dist::build(&[1.0; 1_000]);
                let sample_count = 1000;
                let mut values = Vec::with_capacity(sample_count);
                for i in 0..sample_count {
                    let (_, x) = dist.sample_continuous(i as f32 / sample_count as f32);
                    values.push(x);
                }
                values.sort_floats();

                {
                    let mut last = *values.first().unwrap();
                    for i in 1..sample_count {
                        let current = values[i];
                        assert!((last - current).abs() <= (1.0 / sample_count as f32) * 1.001);
                        last = current;
                    }
                }
                let max = values.last().unwrap();
                let min = values.first().unwrap();
                assert!((1.0 - max).abs() < 0.01);
                assert!((0.0 - min).abs() < 0.01);
            }
        }
    }
}
#[cfg(test)]
pub(crate) use distribution_1d_tests;

