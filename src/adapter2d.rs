use crate::data2d::Data2D;
use crate::distribution::{
    Discrete1D,
    Discrete1DPdf,
    Discrete2D,
    Discrete2DPdf,
    Continuous1D,
    Continuous2D,
};
use num_traits::{
    real::Real,
    AsPrimitive,
};

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    vec::Vec,
};

#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize))]
pub struct Adapter2D<D> {
    pub marginal: D,
    pub conditional: Box<[D]>,
}

impl<D: Discrete1D<R>, R> Discrete2D<R> for Adapter2D<D> {
    type Weight = D::Weight;

    fn build(weights: &Data2D<D::Weight>) -> Self {
        let mut conditional = Vec::with_capacity(weights.height());
        let mut marginal_weights = Vec::with_capacity(weights.height());

        for row in weights.iter() {
            let table = D::build(row);
            marginal_weights.push(table.integral());
            conditional.push(table);
        }

        let marginal = D::build(&marginal_weights);

        Self {
            marginal,
            conditional: conditional.into_boxed_slice(),
        }
    }

    fn sample(&self, [u, v]: [R; 2]) -> [usize; 2] {
        let y = self.marginal.sample(v);
        let x = self.conditional[y].sample(u);

        [x, y]
    }

    fn integral(&self) -> D::Weight {
        self.marginal.integral()
    }

    fn height(&self) -> usize {
        self.conditional.len()
    }

    fn width(&self) -> usize {
        self.conditional[0].size()
    }
}

impl<D: Discrete1DPdf<R>, R> Discrete2DPdf<R> for Adapter2D<D> {
    fn pdf(&self, [u, v]: [usize; 2]) -> D::Weight {
        let pdf_y = self.marginal.pdf(v);
        let pdf_x = self.conditional[v].pdf(u);

        pdf_y * pdf_x
    }
}

impl<D: Continuous1D<R>, R: Real + AsPrimitive<usize> + 'static> Continuous2D<R> for Adapter2D<D>
    where usize: AsPrimitive<R>,
{
    fn sample_continuous(&self, [u, v]: [R; 2]) -> [R; 2] {
        let y = self.marginal.sample_continuous(v);
        let offset_y = (y * <usize as AsPrimitive<R>>::as_(self.height())).as_();
        let x = self.conditional[offset_y].sample_continuous(u);

        [x, y]
    }

    fn invert_continuous(&self, [u, v]: [R; 2]) -> [R; 2] {
        let y = self.marginal.invert_continuous(v);
        let offset_y = (y * <usize as AsPrimitive<R>>::as_(self.height())).as_();
        let x = self.conditional[offset_y].invert_continuous(u);

        [x, y]
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::distribution_2d_tests;

    pub type Inversion2D<R> = crate::Adapter2D<crate::Inversion1D<R>>;

    distribution_2d_tests!(crate::adapter2d::tests::Inversion2D);
}

