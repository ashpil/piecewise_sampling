use crate::distribution::{
    Distribution1D,
    ContinuousDistribution1D,
};
use num_traits::{
    real::Real,
    cast,
};

pub struct Inversion1D<R: Real> {
    pub weight_sum: R,
    pub cdf: Box<[R]>,
}

impl<R: Real> Distribution1D for Inversion1D<R> {
    type Weight = R;

    fn build(weights: &[R]) -> Self {
        let mut cdf = vec![R::zero(); weights.len() + 1].into_boxed_slice();

        for (i, weight) in weights.iter().enumerate() {
            cdf[i + 1] = cdf[i] + *weight;
        }

        let weight_sum = cdf[weights.len()];

        for weight in cdf.iter_mut() {
            *weight = *weight / weight_sum;
        }

        Self {
            weight_sum,
            cdf,
        }
    }

    fn sample(&self, u: R) -> (R, usize) {
        let offset = self.cdf.partition_point(|p| *p <= u) - 1;
        let pdf = (self.cdf[offset + 1] - self.cdf[offset]) * self.weight_sum;
        (pdf, offset)
    }

    fn pdf(&self, u: usize) -> R {
        (self.cdf[u + 1] - self.cdf[u]) * self.weight_sum
    }

    fn integral(&self) -> R {
        self.weight_sum
    }

    fn size(&self) -> usize {
        self.cdf.len() - 1
    }
}

impl<R: Real> ContinuousDistribution1D for Inversion1D<R> {
    fn sample_continuous(&self, u: R) -> (R, R) {
        let (pdf, offset) = self.sample(u);
        let du = (u - self.cdf[offset]) / (self.cdf[offset + 1] - self.cdf[offset]);
        (pdf, (cast::<usize, R>(offset).unwrap() + du) / cast(self.size()).unwrap())
    }

    fn inverse_continuous(&self, u: R) -> R {
        let scaled: R = cast::<usize, R>(self.size()).unwrap() * u;
        let idx: usize = cast(scaled).unwrap();
        let delta = scaled - cast(idx).unwrap();
        (R::one() - delta) * self.cdf[idx] + delta * self.cdf[idx + 1]
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::distribution_1d_tests;
    use crate::distribution::continuous_distribution_1d_tests;

    distribution_1d_tests!(crate::inversion::Inversion1D);
    continuous_distribution_1d_tests!(crate::inversion::Inversion1D);
}

