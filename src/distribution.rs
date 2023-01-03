use crate::data2d::Data2D;
use num_traits::real::Real;

// 1D piecewise constant distribution
// sampling functions are discrete
pub trait Distribution1D {
    type Weight: Real;

    // constructor
    fn build(weights: &[Self::Weight]) -> Self;

    // takes in rand [0-1), returns (pdf, sampled idx)
    fn sample(&self, u: Self::Weight) -> (Self::Weight, usize);

    // takes in coord, returns pdf
    fn pdf(&self, u: usize) -> Self::Weight;

    // sum of all weights
    fn integral(&self) -> Self::Weight;

    // range of sampled idxs, should be len of weights
    fn size(&self) -> usize;
}

pub trait ContinuousDistribution1D: Distribution1D {
    // takes in rand [0-1), returns (pdf, sampled [0-1))
    fn sample_continuous(&self, u: Self::Weight) -> (Self::Weight, Self::Weight);

    // inverse of above
    fn inverse_continuous(&self, u: Self::Weight) -> Self::Weight;
}

// 2D piecewise constant distribution
pub trait Distribution2D {
    // constructor
    fn build(weights: &Data2D<f32>) -> Self;

    // takes in rand [0-1)x[0-1), returns (pdf, sampled uv coords)
    fn sample(&self, uv: [f32; 2]) -> (f32, [usize; 2]);

    // takes in coords, returns pdf
    fn pdf(&self, uv: [usize; 2]) -> f32;

    fn width(&self) -> usize;
    fn height(&self) -> usize;

    // fills demo image with sample_count samples
    fn fill_demo_image(&self, demo: &mut Data2D<[f32; 3]>, rngs: impl Iterator<Item = [f32; 2]>) {
        for rng in rngs {
            let (_, [x, y]) = self.sample(rng);
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
    num_traits::{Zero, AsPrimitive},
};

#[cfg(test)]
pub fn chisq_distribution_1d<D: Distribution1D>(expected: &[D::Weight], sample_count: usize)
    where rand::distributions::Standard: rand::distributions::Distribution<D::Weight>,
        D::Weight: std::fmt::Debug + AsPrimitive<f64>,
        u32: AsPrimitive<D::Weight>,
        f64: AsPrimitive<D::Weight>,
        usize: AsPrimitive<D::Weight>,
{
    let dist = D::build(expected);
    let mut observed = vec![D::Weight::zero(); expected.len()].into_boxed_slice();
    let mut hist = vec![0u32; expected.len()].into_boxed_slice();
    let mut rng = StdRng::seed_from_u64(0);

    for _ in 0..sample_count {
        let (pdf, idx) = dist.sample(rng.gen());
        assert!((expected[idx] - pdf).abs() < 0.001.as_());
        assert_eq!(pdf, dist.pdf(idx));
        hist[idx] += 1;
    }

    for (weight, obs) in hist.into_iter().zip(observed.iter_mut()) {
        *obs = (weight.as_() / sample_count.as_()) * dist.integral();
    }

    let mut chsq = 0.0;
    let dof = expected.len() - 1;
    for (obs, exp) in observed.into_iter().zip(expected.into_iter()) {
        let diff = *obs - *exp;
        chsq += (diff * diff / *exp).as_();
    }

    let pval = 1.0 - ChiSquared::new(dof as f64).unwrap().cdf(chsq as f64);
    assert!(pval >= 0.99, "failed chi-squared statistical test, p = {}", pval);
}

#[cfg(test)]
pub fn test_inv_1d<D: ContinuousDistribution1D>(weights: &[D::Weight], sample_count: usize)
    where D::Weight: std::fmt::Display + 'static,
        f64: AsPrimitive<D::Weight>,
        usize: AsPrimitive<D::Weight>,
{
    let dist = D::build(&weights);

    for i in 0..sample_count {
        let x = i.as_() / sample_count.as_();
        let (_, y) = dist.sample_continuous(x);
        let inv = dist.inverse_continuous(y);
        assert!((inv - x).abs() < 0.0001.as_(), "{} original not equal to {} inverse of sample {}", x, inv, y);
    }
}

#[cfg(test)]
pub fn test_continuous_discrete_matching_1d<D: ContinuousDistribution1D>(weights: &[D::Weight], sample_count: usize)
    where D::Weight: AsPrimitive<usize> + std::fmt::Display,
        usize: AsPrimitive<D::Weight>,
{
    let dist = D::build(&weights);

    for i in 0..sample_count {
        let input = i.as_() / sample_count.as_();
        let (_, output_discrete) = dist.sample(input);
        let (_, output_continuous) = dist.sample_continuous(input);
        assert_eq!((output_continuous * dist.size().as_()).as_(), output_discrete);
    }
}

#[cfg(test)]
macro_rules! distribution_1d_tests {
    ($impl:path) => {
        mod distribution_1d {
            use crate::distribution::chisq_distribution_1d;
            use $impl as Dist;

            #[test]
            fn basic() {
                chisq_distribution_1d::<Dist<f32>>(&[1.0, 1.0, 2.0, 4.0, 8.0], 1000);
            }

            #[test]
            fn uniform() {
                chisq_distribution_1d::<Dist<f32>>(&[1.0; 10_000], 1_000_000);
            }

            #[test]
            fn increasing() {
                let mut distr = [0.0; 100];
                for (i, weight) in distr.iter_mut().enumerate() {
                    *weight = (5 * (i + 1)) as f32;
                }
                chisq_distribution_1d::<Dist<f32>>(&distr, 100_000);
            }
        }
    }
}
#[cfg(test)]
pub(crate) use distribution_1d_tests;

#[cfg(test)]
macro_rules! continuous_distribution_1d_tests {
    ($impl:path) => {
        mod continuous_distribution_1d {
            use crate::distribution::{
                Distribution1D,
                ContinuousDistribution1D,
                test_inv_1d,
                test_continuous_discrete_matching_1d,
            };
            use $impl as Dist;

            #[test]
            fn surjective() {
                let dist = Dist::<f32>::build(&[1.0; 1_000]);
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

            #[test]
            fn injective() {
                let dist = Dist::<f64>::build(&[1.0, 1.0, 2.0, 4.0, 8.0]);
                let sample_count = 1000;
                let mut values = Vec::with_capacity(sample_count);
                for i in 0..sample_count {
                    let (_, x) = dist.sample_continuous(i as f64 / sample_count as f64);
                    values.push(x);
                }
                values.sort_floats();
                {
                    let mut last = *values.first().unwrap();
                    for i in 1..sample_count {
                        let current = values[i];
                        assert_ne!(last, current);
                        last = current;
                    }
                }
            }

            #[test]
            fn inverse_uniform() {
                test_inv_1d::<Dist<f64>>(&[1.0; 1_000], 1000);
            }

            #[test]
            fn inverse_basic() {
                test_inv_1d::<Dist<f64>>(&[1.0, 2.0, 4.0, 8.0], 1000);
            }

            #[test]
            fn inverse_increasing() {
                let mut distr = [0.0; 100];
                for (i, weight) in distr.iter_mut().enumerate() {
                    *weight = (5 * (i + 1)) as f64;
                }
                test_inv_1d::<Dist<f64>>(&distr, 1000);
            }

            #[test]
            fn continuous_discrete_matching_uniform() {
                test_continuous_discrete_matching_1d::<Dist<f64>>(&[1.0; 1_000], 1000);
            }

            #[test]
            fn continuous_discrete_matching_basic() {
                test_continuous_discrete_matching_1d::<Dist<f64>>(&[1.0, 1.0, 2.0, 4.0, 8.0], 1000);
            }

            #[test]
            fn continuous_discrete_matching_increasing() {
                let mut distr = [0.0; 100];
                for (i, weight) in distr.iter_mut().enumerate() {
                    *weight = (5 * (i + 1)) as f64;
                }
                test_continuous_discrete_matching_1d::<Dist<f64>>(&distr, 1000);
            }
        }
    }
}
#[cfg(test)]
pub(crate) use continuous_distribution_1d_tests;
