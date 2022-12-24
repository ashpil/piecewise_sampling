use crate::Distribution2D;

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
        // perform binary search
        fn find_interval(a: &[f32], val: f32) -> usize {
            let mut first = 0;
            let mut len = a.len();

            while len > 0 {
                let half = len >> 1;
                let middle = first + half;
                if a[middle] <= val {
                    first = middle + 1;
                    len -= half + 1;
                } else {
                    len = half;
                }
            }

            (first - 1).clamp(0, a.len() - 2)
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

