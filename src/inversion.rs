use crate::distribution::{Distribution1D, Distribution2D};

pub struct Inversion1D {
    pub weight_sum: f32,
    pub cdf: Box<[f32]>,
}

impl Distribution1D for Inversion1D {
    fn build(weights: &[f32]) -> Self {
        let mut cdf = vec![0.0; weights.len() + 1].into_boxed_slice();

        for (i, weight) in weights.iter().enumerate() {
            cdf[i + 1] = cdf[i] + weight;
        }

        let weight_sum = cdf[weights.len()];

        for weight in cdf.iter_mut() {
            *weight /= weight_sum;
        }

        Self {
            weight_sum,
            cdf,
        }
    }

    fn sample(&self, u: f32) -> (f32, usize) {
        fn find_interval(a: &[f32], val: f32) -> usize {
            a.partition_point(|p| *p <= val) - 1
        }
        let offset = find_interval(&self.cdf, u);
        let pdf = (self.cdf[offset + 1] - self.cdf[offset]) * self.weight_sum;
        (pdf, offset)
    }

    fn pdf(&self, u: usize) -> f32 {
        (self.cdf[u + 1] - self.cdf[u]) * self.weight_sum
    }

    fn integral(&self) -> f32 {
        return self.weight_sum;
    }
}

pub struct Inversion2D {
    pub marginal: Inversion1D,
    pub conditional: Box<[Inversion1D]>,
}

impl Distribution2D for Inversion2D {
    fn build(weights: &[Vec<f32>]) -> Self {
        let mut conditional = Vec::with_capacity(weights.len());
        let mut marginal_weights = Vec::with_capacity(weights.len());
        for row in weights {
            let table = Inversion1D::build(row);
            marginal_weights.push(table.integral());
            conditional.push(table);
        }

        let marginal = Inversion1D::build(&marginal_weights);

        Self {
            marginal,
            conditional: conditional.into_boxed_slice(),
        }
    }

    fn sample(&self, [u, v]: [f32; 2]) -> (f32, [usize; 2]) {
        let (pdf_y, y) = self.marginal.sample(u);
        let (pdf_x, x) = self.conditional[y].sample(v);

        (pdf_x * pdf_y, [x, y])
    }

    fn pdf(&self, [u, v]: [usize; 2]) -> f32 {
        let pdf_y = self.marginal.pdf(v);
        let pdf_x = self.conditional[v].pdf(u);

        return pdf_y * pdf_x;
    }
}

#[cfg(test)]
mod tests {
    use super::Inversion1D;
    use crate::utils::test_distribution_1d;

    #[test]
    fn basic1d() {
        test_distribution_1d::<Inversion1D>(&[1.0, 1.0, 2.0, 4.0, 8.0], 1000);
    }

    #[test]
    fn uniform1d() {
        test_distribution_1d::<Inversion1D>(&[1.0; 10_000], 1_000_000);
    }

    #[test]
    fn increasing1d() {
        let mut distr = [0.0; 100];
        for (i, weight) in distr.iter_mut().enumerate() {
            *weight = (5 * (i + 1)) as f32;
        }
        test_distribution_1d::<Inversion1D>(&distr, 100_000);
    }
}

