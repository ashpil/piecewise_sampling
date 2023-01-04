use crate::distribution::{
    Distribution1D,
    ContinuousDistribution1D,
};
use num_traits::{
    real::Real,
    cast,
};

pub struct Inversion1D<R: Real> {
    pub cdf: Box<[R]>,
}

impl<R: Real> Distribution1D for Inversion1D<R> {
    type Weight = R;

    fn build(weights: &[R]) -> Self {
        let mut cdf = std::iter::once(R::zero()).chain(weights.iter().cloned()).collect::<Box<[R]>>();

        for i in 1..cdf.len() {
            cdf[i] = cdf[i - 1] + cdf[i];
        }

        Self {
            cdf,
        }
    }

    fn sample(&self, u: R) -> (R, usize) {
        let point = u * self.integral();
        let offset = self.cdf.partition_point(|p| *p <= point) - 1;
        let pdf = self.cdf[offset + 1] - self.cdf[offset];
        (pdf, offset)
    }

    fn pdf(&self, u: usize) -> R {
        self.cdf[u + 1] - self.cdf[u]
    }

    fn integral(&self) -> R {
        *self.cdf.last().unwrap()
    }

    fn size(&self) -> usize {
        self.cdf.len() - 1
    }
}

impl<R: Real> ContinuousDistribution1D for Inversion1D<R> {
    fn sample_continuous(&self, u: R) -> (R, R) {
        let (pdf, offset) = self.sample(u);
        let du = (u * self.integral() - self.cdf[offset]) / (self.cdf[offset + 1] - self.cdf[offset]);
        (pdf, (cast::<usize, R>(offset).unwrap() + du) / cast(self.size()).unwrap())
    }

    fn inverse_continuous(&self, u: R) -> R {
        let scaled: R = cast::<usize, R>(self.size()).unwrap() * u;
        let idx: usize = cast(scaled).unwrap();
        let delta = scaled - cast(idx).unwrap();
        crate::utils::lerp(delta, self.cdf[idx], self.cdf[idx + 1]) / self.integral()
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::distribution_1d_tests;
    use crate::distribution::continuous_distribution_1d_tests;

    distribution_1d_tests!(crate::inversion::Inversion1D);
    continuous_distribution_1d_tests!(crate::inversion::Inversion1D);
}

