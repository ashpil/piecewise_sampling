use crate::{Distribution1D, Distribution2D};

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
    pub width: usize,
    pub height: usize,

    pub conditional_pdfs_integrals: Vec<Vec<f32>>,
    pub conditional_cdfs: Vec<Vec<f32>>,

    pub marginal_pdf_integral: Vec<f32>,
    pub marginal_cdf: Vec<f32>,
}

impl Distribution2D for Inversion2D {
    fn build(image: &[Vec<f32>]) -> Self {
        let height = image.len();
        let width = image[0].len();

        let mut conditional_pdfs_integrals: Vec<Vec<f32>> = vec![vec![0.0; width + 1]; height];
        let mut conditional_cdfs = vec![vec![0.0; width + 1]; height];

        let mut marginal_pdf_integral = vec![0.0; height + 1];
        let mut marginal_cdf = vec![0.0; height + 1];

        for (row_index, row) in conditional_pdfs_integrals.iter_mut().enumerate() {
            for (col_index, pixel) in row[0..width].iter_mut().enumerate() {
                *pixel = image[row_index][col_index];
            }
        }

        for (j, row) in conditional_pdfs_integrals.iter_mut().enumerate() {
            for i in 1..width + 1 {
                conditional_cdfs[j][i] = conditional_cdfs[j][i - 1] + row[i - 1] / width as f32;
            }

            row[width] = conditional_cdfs[j][width];
            for i in 0..width + 1 {
                conditional_cdfs[j][i] /= row[width];
            }
        }

        for (i, v) in marginal_pdf_integral[..height].iter_mut().enumerate() {
            *v = conditional_pdfs_integrals[i][width];
        }

        for i in 1..height + 1 {
            marginal_cdf[i] = marginal_cdf[i - 1] + marginal_pdf_integral[i - 1] / height as f32;
        }

        marginal_pdf_integral[height] = marginal_cdf[height];
        for i in 0..height + 1 {
            marginal_cdf[i] /= marginal_pdf_integral[height];
        }

        Self {
            width,
            height,

            conditional_pdfs_integrals,
            conditional_cdfs,

            marginal_pdf_integral,
            marginal_cdf,
        }
    }

    fn sample(&self, [u, v]: [f32; 2]) -> (f32, [usize; 2]) {
        fn find_interval(a: &[f32], val: f32) -> usize {
            a.partition_point(|p| *p <= val) - 1
        }

        let offset_v = find_interval(&self.marginal_cdf, v);
        let pdf_v = self.marginal_pdf_integral[offset_v] / self.marginal_pdf_integral[self.height];

        let offset_u = find_interval(&self.conditional_cdfs[offset_v], u);
        let pdf_u = self.conditional_pdfs_integrals[offset_v][offset_u] / self.conditional_pdfs_integrals[offset_v][self.width];

        (pdf_v * pdf_u, [offset_u, offset_v])
    }

    fn pdf(&self, [u, v]: [usize; 2]) -> f32 {
        self.conditional_pdfs_integrals[v][u] / self.marginal_pdf_integral[self.height]
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

