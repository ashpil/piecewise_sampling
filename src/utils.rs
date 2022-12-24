use num_traits::Float;

#[cfg(test)]
use {
    crate::Distribution1D,
    rand::{rngs::StdRng, Rng, SeedableRng},
    statrs::distribution::{ChiSquared, ContinuousCDF},
};

pub fn kahan_sum<F: Float>(input: impl IntoIterator<Item=F>) -> F {
    let mut sum = F::zero();
    let mut err = F::zero();
    for v in input {
        let y = v - err;
        let t = sum + y;
        err = (t - sum) - y;
        sum = t;
    }
    sum
}

#[cfg(test)]
pub fn test_distribution_1d<D: Distribution1D>(expected: &[f32], sample_count: usize) {
    let dist = D::build(expected);
    let mut observed = vec![0.0f32; expected.len()].into_boxed_slice();
    let mut hist = vec![0u32; expected.len()].into_boxed_slice();
    let mut rng = StdRng::seed_from_u64(0);

    for _ in 0..sample_count {
        let (pdf, idx) = dist.sample(rng.gen());
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

