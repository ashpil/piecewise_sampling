use crate::{distribution::{
    Discrete1D, Discrete1DPdf
}, utils};
use num_traits::{
    Num,
    real::Real,
    AsPrimitive,
};
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

struct Reservoir<Candidate: Clone + Copy, N: Num + AsPrimitive<R>, R: Real + 'static> {
    selected: Option<Candidate>,
    weight_sum: N,
    rand: R,
}

impl<'a, Candidate: Clone + Copy, N: Num + AsPrimitive<R>, R: Real + 'static> Reservoir<Candidate, N, R> {
    fn empty(rand: R) -> Self {
        Self {
            selected: None,
            weight_sum: N::zero(),
            rand,
        }
    }

    fn update(&mut self, new_candidate: Candidate, new_weight: N) {
        self.weight_sum = self.weight_sum + new_weight;
        self.selected = if self.rand * self.weight_sum.as_() < new_weight.as_() {
            self.rand = self.weight_sum.as_() * self.rand / new_weight.as_();
            Some(new_candidate)
        } else {
            self.rand = (self.rand * self.weight_sum.as_() - new_weight.as_()) / (self.weight_sum - new_weight).as_();
            self.selected
        };
    }
}

pub type Reservoir2D<R> = crate::Adapter2D<Reservoir1D<R>>;

#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize))]
pub struct Reservoir1D<W> {
    pub weights: Box<[W]>,
}

impl<W: Num + AsPrimitive<R> + PartialOrd, R: Real + AsPrimitive<W> + 'static> Discrete1D<R> for Reservoir1D<W> {
    type Weight = W;

    fn build(weights: &[W]) -> Self {
        Self {
            weights: weights.to_vec().into_boxed_slice(),
        }
    }

    fn sample(&self, u: R) -> usize {
        let mut r = Reservoir::empty(u);
        for (i, weight) in self.weights.iter().enumerate() {
            r.update(i, *weight);
        }
        r.selected.unwrap().clone()
    }

    fn integral(&self) -> W {
        <W as utils::Sum>::sum(self.weights.iter().cloned())
    }

    fn size(&self) -> usize {
        self.weights.len()
    }
}

impl<W: Num + AsPrimitive<R> + PartialOrd, R: Real + AsPrimitive<W> + 'static> Discrete1DPdf<R> for Reservoir1D<W> {
    fn pdf(&self, u: usize) -> W {
        self.weights[u]
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::distribution_1d_tests;

    distribution_1d_tests!(crate::reservoir::Reservoir1D);
}

