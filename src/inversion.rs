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

fn build_fast<R: Real>(weights: &[R]) -> Inversion1D<R> {
    let mut cdf: Box<[std::mem::MaybeUninit<R>]> = Box::new_uninit_slice(weights.len() + 1);
    unsafe {
        cdf[0].as_mut_ptr().write(R::zero());
    }

    for (i, weight) in weights.iter().enumerate() {
        unsafe {
            cdf[i + 1].as_mut_ptr().write(cdf[i].assume_init() + *weight);
        }
    }

    Inversion1D {
        cdf: unsafe { cdf.assume_init() },
    }
}

impl<R: Real> Distribution1D for Inversion1D<R> {
    type Weight = R;

    fn build(weights: &[R]) -> Self {
        return build_fast(weights);
        let mut cdf = Vec::with_capacity(weights.len() + 1);
        cdf.push(R::zero());

        for (i, weight) in weights.iter().enumerate() {
            cdf.push(cdf[i] + *weight);
        }

        Self {
            cdf: cdf.into_boxed_slice(),
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

