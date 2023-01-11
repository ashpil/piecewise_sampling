use crate::data2d::Data2D;
use num_traits::{
    real::Real,
    Num,
    AsPrimitive,
};

// 1D piecewise constant distribution
// sampling functions are discrete
pub trait Discrete1D<R> {
    type Weight: Num; // type for weights, can be almost anything that has arithmetic ops

    // constructor
    fn build(weights: &[Self::Weight]) -> Self;

    // takes in rand [0-1), returns sampled idx
    fn sample(&self, u: R) -> usize;

    // takes in coord, returns pdf
    fn pdf(&self, u: usize) -> Self::Weight;

    // sum of all weights
    fn integral(&self) -> Self::Weight;

    // range of sampled idxs, should be len of weights
    fn size(&self) -> usize;
}

pub trait Continuous1D<R>: Discrete1D<R> {
    // takes in rand [0-1), returns sampled [0-1)
    fn sample_continuous(&self, u: R) -> R;

    // inverse of above
    fn inverse_continuous(&self, u: R) -> R;
}

// 2D piecewise constant distribution
pub trait Discrete2D<R> {
    type Weight: Num; // type for weights, can be almost anything that has arithmetic ops

    // constructor
    fn build(weights: &Data2D<Self::Weight>) -> Self;

    // takes in rand [0-1)x[0-1), returns sampled uv coords
    fn sample(&self, uv: [R; 2]) -> [usize; 2];

    // takes in coords, returns pdf
    fn pdf(&self, uv: [usize; 2]) -> Self::Weight;

    fn width(&self) -> usize;
    fn height(&self) -> usize;

    // fills demo image with sample_count samples
    fn fill_demo_image(&self, demo: &mut Data2D<[f32; 3]>, rngs: impl Iterator<Item = [R; 2]>) {
        for rng in rngs {
            let [x, y] = self.sample(rng);
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

pub trait Continuous2D<R>: Discrete2D<R> {
    // takes in rand [0-1), returns sampled [0-1)x[0-1)
    fn sample_continuous(&self, uv: [R; 2]) -> [R; 2];

    // inverse of above
    fn inverse_continuous(&self, uv: [R; 2]) -> [R; 2];
}

pub fn visualize_warping<D: Continuous2D<R>, R: Real + AsPrimitive<usize> + 'static>(distr: &D, block_count: usize) -> Data2D<[f32; 3]>
    where usize: AsPrimitive<R>,
    f64: AsPrimitive<R>,
{
    let mut out = Data2D::new_same(distr.width(), distr.height(), [0.0, 0.0, 0.0]);
    for j in 0..distr.height() {
        for i in 0..distr.width() {
            let input = [
                (i.as_() + 0.5.as_()) / distr.width().as_(),
                (j.as_() + 0.5.as_()) / distr.height().as_(),
            ];
            let [x, y] = distr.inverse_continuous(input);
            let x_scaled = (x * <usize as AsPrimitive<R>>::as_(block_count)).as_();
            let y_scaled = (y * <usize as AsPrimitive<R>>::as_(block_count)).as_();
            let tile = (x_scaled + block_count * y_scaled) as u64 + 1;
            out[j][i] = crate::utils::u64_to_color(tile);
        }
    }
    out
}

#[cfg(test)]
use {
    rand::{rngs::StdRng, Rng, SeedableRng},
    statrs::distribution::{ChiSquared, ContinuousCDF},
};

#[cfg(test)]
pub fn chisq_distribution_1d<D: Discrete1D<f64>>(expected: &[D::Weight], sample_count: usize)
    where D::Weight: std::fmt::Display + AsPrimitive<f64>,
        f64: AsPrimitive<D::Weight>,
{
    let dist = D::build(expected);
    let mut observed = vec![0.0f64; expected.len()].into_boxed_slice();
    let mut hist = vec![0usize; expected.len()].into_boxed_slice();
    let mut rng = StdRng::seed_from_u64(0);

    for _ in 0..sample_count {
        let idx = dist.sample(rng.gen::<f64>());
        hist[idx] += 1;
    }

    for (weight, obs) in hist.into_iter().zip(observed.iter_mut()) {
        *obs = ((*weight as f64) / (sample_count as f64)) * dist.integral().as_();
    }

    let mut chsq = 0.0;
    let dof = expected.len() - 1;
    for (obs, exp) in observed.into_iter().zip(expected.into_iter()) {
        let diff = *obs - exp.as_();
        chsq += diff * diff / exp.as_();
    }

    let pval = 1.0 - ChiSquared::new(dof as f64).unwrap().cdf(chsq as f64);
    assert!(pval >= 0.99, "failed chi-squared statistical test, p = {}", pval);
}

#[cfg(test)]
pub fn test_inv_1d<R: Real + 'static, D: Continuous1D<R>>(weights: &[D::Weight], sample_count: usize)
    where R: std::fmt::Display,
          usize: AsPrimitive<R>,
          f64: AsPrimitive<R>,
{
    let dist = D::build(&weights);

    for i in 0..sample_count {
        let x = i.as_() / sample_count.as_();
        let y = dist.sample_continuous(x);
        let inv = dist.inverse_continuous(y);
        assert!((inv - x).abs() < 0.01f64.as_(), "{} original not equal to {} inverse of sample {}", x, inv, y);
    }
}

#[cfg(test)]
pub fn test_continuous_discrete_matching_1d<R: Real, D: Continuous1D<R>>(weights: &[D::Weight], sample_count: usize)
    where R: std::fmt::Display + AsPrimitive<usize>,
          usize: AsPrimitive<R>,
{
    let dist = D::build(&weights);

    for i in 0..sample_count {
        let input = i.as_() / sample_count.as_();
        let output_discrete = dist.sample(input);
        let output_continuous = dist.sample_continuous(input);
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
                chisq_distribution_1d::<Dist<usize>>(&[1, 1, 2, 4, 8], 10_000);
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
                Discrete1D,
                Continuous1D,
                test_inv_1d,
                test_continuous_discrete_matching_1d,
            };
            use $impl as Dist;

            #[test]
            fn surjective() {
                let dist = <Dist::<f32> as Discrete1D<f32>>::build(&[1.0; 1_000]);
                let sample_count = 1000;
                let mut values = Vec::with_capacity(sample_count);
                for i in 0..sample_count {
                    let x = dist.sample_continuous(i as f32 / sample_count as f32);
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
                assert!((1.0f32 - max).abs() < 0.01);
                assert!((0.0f32 - min).abs() < 0.01);
            }

            #[test]
            fn injective() {
                let dist = <Dist::<f64> as Discrete1D<f64>>::build(&[1.0, 1.0, 2.0, 4.0, 8.0]);
                let sample_count = 1000;
                let mut values = Vec::with_capacity(sample_count);
                for i in 0..sample_count {
                    let x = dist.sample_continuous(i as f64 / sample_count as f64);
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
                test_inv_1d::<f64, Dist<f64>>(&[1.0; 1_000], 1000);
            }

            #[test]
            fn inverse_basic() {
                test_inv_1d::<f64, Dist<f64>>(&[1.0, 2.0, 4.0, 8.0], 1000);
            }

            #[test]
            fn inverse_increasing() {
                let mut distr = [0.0; 100];
                for (i, weight) in distr.iter_mut().enumerate() {
                    *weight = (5 * (i + 1)) as f64;
                }
                test_inv_1d::<f64, Dist<f64>>(&distr, 1000);
            }

            #[test]
            fn continuous_discrete_matching_uniform() {
                test_continuous_discrete_matching_1d::<f64, Dist<f64>>(&[1.0; 1_000], 1024);
            }

            #[test]
            fn continuous_discrete_matching_basic() {
                test_continuous_discrete_matching_1d::<f64, Dist<f64>>(&[1.0, 1.0, 2.0, 4.0, 8.0], 1024);
            }

            #[test]
            fn continuous_discrete_matching_increasing() {
                let mut distr = [0.0; 100];
                for (i, weight) in distr.iter_mut().enumerate() {
                    *weight = (5 * (i + 1)) as f64;
                }
                test_continuous_discrete_matching_1d::<f64, Dist<f64>>(&distr, 1024);
            }
        }
    }
}
#[cfg(test)]
pub(crate) use continuous_distribution_1d_tests;
