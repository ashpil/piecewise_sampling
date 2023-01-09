use crate::data2d::Data2D;
use crate::distribution::{
    Discrete1D,
    Discrete2D,
    Continuous1D,
    Continuous2D,
};
use num_traits::cast;

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    vec::Vec,
};

#[derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
pub struct Adapter2D<D: Discrete1D> {
    pub marginal: D,
    pub conditional: Box<[D]>,
}

impl<D: Discrete1D> Discrete2D for Adapter2D<D> {
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

    fn sample(&self, [u, v]: [D::Weight; 2]) -> [usize; 2] {
        let y = self.marginal.sample(v);
        let x = self.conditional[y].sample(u);

        [x, y]
    }

    fn pdf(&self, [u, v]: [usize; 2]) -> D::Weight {
        let pdf_y = self.marginal.pdf(v);
        let pdf_x = self.conditional[v].pdf(u);

        pdf_y * pdf_x
    }

    fn height(&self) -> usize {
        self.conditional.len()
    }

    fn width(&self) -> usize {
        self.conditional[0].size()
    }
}

impl<D: Continuous1D> Continuous2D for Adapter2D<D> {
    fn sample_continuous(&self, [u, v]: [D::Weight; 2]) -> [D::Weight; 2] {
        let y = self.marginal.sample_continuous(v);
        let offset_y = cast::<D::Weight, usize>(y * cast(self.height()).unwrap()).unwrap();
        let x = self.conditional[offset_y].sample_continuous(u);

        [x, y]
    }

    fn inverse_continuous(&self, [u, v]: [D::Weight; 2]) -> [D::Weight; 2] {
        let y = self.marginal.inverse_continuous(v);
        let offset_y = cast::<D::Weight, usize>(y * cast(self.height()).unwrap()).unwrap();
        let x = self.conditional[offset_y].inverse_continuous(u);

        [x, y]
    }
}

