use crate::distribution::{
    Discrete1D,
    Discrete1DPdf,
    Continuous1D,
};
use num_traits::{
    Num,
    real::Real,
    AsPrimitive,
};
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

pub type Inversion2D<R> = crate::Adapter2D<Inversion1D<R>>;

#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize))]
pub struct Inversion1D<W> {
    pub cdf: Box<[W]>,
}

impl<W: Num + AsPrimitive<R>, R: Real + 'static> Discrete1D<R> for Inversion1D<W> {
    type Weight = W;

    fn build(weights: &[W]) -> Self {
        let mut cdf = core::iter::once(W::zero()).chain(weights.iter().cloned()).collect::<Box<[W]>>();

        for i in 1..cdf.len() {
            cdf[i] = cdf[i - 1] + cdf[i];
        }

        Self {
            cdf,
        }
    }

    fn sample(&self, u: R) -> usize {
        let point = u * self.integral().as_();
        let offset = self.cdf.partition_point(|p| p.as_() <= point) - 1;
        offset
    }

    fn integral(&self) -> W {
        *self.cdf.last().unwrap()
    }

    fn size(&self) -> usize {
        self.cdf.len() - 1
    }
}

impl<W: Num + AsPrimitive<R>, R: Real + 'static> Discrete1DPdf<R> for Inversion1D<W> {
    fn pdf(&self, u: usize) -> W {
        self.cdf[u + 1] - self.cdf[u]
    }
}

impl<W: Num + AsPrimitive<R>, R: Real + AsPrimitive<usize> + 'static> Continuous1D<R> for Inversion1D<W>
    where usize: AsPrimitive<R>,
{
    fn sample_continuous(&self, u: R) -> R {
        let offset = self.sample(u);
        let du = (u * self.integral().as_() - self.cdf[offset].as_()) / (self.cdf[offset + 1].as_() - self.cdf[offset].as_());
        (offset.as_() + du) / self.size().as_()
    }

    fn inverse_continuous(&self, u: R) -> R {
        let scaled: R = self.size().as_() * u;
        let idx: usize = scaled.as_();
        let delta = scaled - idx.as_();
        crate::utils::lerp(delta, self.cdf[idx].as_(), self.cdf[idx + 1].as_()) / self.integral().as_()
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::distribution_1d_tests;
    use crate::distribution::continuous_distribution_1d_tests;

    distribution_1d_tests!(crate::inversion::Inversion1D);
    continuous_distribution_1d_tests!(crate::inversion::Inversion1D);
}

