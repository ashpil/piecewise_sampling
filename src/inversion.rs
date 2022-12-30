use crate::distribution::Distribution1D;

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

    fn sample_discrete(&self, u: f32) -> (f32, usize) {
        let offset = self.cdf.partition_point(|p| *p <= u) - 1;
        let pdf = (self.cdf[offset + 1] - self.cdf[offset]) * self.weight_sum;
        (pdf, offset)
    }

    fn sample_continuous(&self, u: f32) -> (f32, f32) {
        let (pdf, offset) = self.sample_discrete(u);
        let du = (u - self.cdf[offset]) / (self.cdf[offset + 1] - self.cdf[offset]);
        (pdf, (offset as f32 + du) / self.cdf.len() as f32)
    }

    fn pdf(&self, u: usize) -> f32 {
        (self.cdf[u + 1] - self.cdf[u]) * self.weight_sum
    }

    fn integral(&self) -> f32 {
        self.weight_sum
    }

    fn size(&self) -> usize {
        self.cdf.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::distribution_1d_tests;
    distribution_1d_tests!(crate::inversion::Inversion1D);
}

