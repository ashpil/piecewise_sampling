use exr::prelude::*;
use crate::{luminance, Sampler};

pub struct InversionSampler {
    pub width: usize,
    pub height: usize,

    pub rgb_image: Vec<Vec<[f32; 3]>>,

    pub conditional_pdfs_integrals: Vec<Vec<f32>>,
    pub conditional_cdfs: Vec<Vec<f32>>,

    pub marginal_pdf_integral: Vec<f32>,
    pub marginal_cdf: Vec<f32>,
}

impl InversionSampler {
    pub fn new(filepath: impl AsRef<std::path::Path>) -> Self {
        let rgb_image = read_first_rgba_layer_from_file(filepath, |resolution, _| {
            let width = resolution.width();
            let height = resolution.height();
            vec![vec![[0.0, 0.0, 0.0]; width]; height]
        }, |buffer, pos, (r, g, b, _): (f32, f32, f32, f32)| {
            buffer[pos.y()][pos.x()] = [r, g, b];
        }).unwrap().layer_data.channel_data.pixels;

        let big_height = rgb_image.len();
        let big_width = rgb_image[0].len();

        let factor = 4;

        let width = big_width / factor;
        let height = big_height / factor;

        let mut conditional_pdfs_integrals: Vec<Vec<f32>> = vec![vec![0.0; width + 1]; height];
        let mut conditional_cdfs = vec![vec![0.0; width + 1]; height];

        let mut marginal_pdf_integral = vec![0.0; height + 1];
        let mut marginal_cdf = vec![0.0; height + 1];

        // calculate downsampled image
        for (row_index, row) in conditional_pdfs_integrals.iter_mut().enumerate() {
            for (col_index, pixel) in row[0..width].iter_mut().enumerate() {
                for i in (row_index * factor)..((row_index + 1) * factor) {
                    for j in (col_index * factor)..((col_index + 1) * factor) {
                        let new_pixel = luminance(rgb_image[i][j], (i as f32 + 0.5) / (big_height as f32));
                        *pixel = (*pixel).max(new_pixel);
                    }
                }
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

        InversionSampler {
            width,
            height,

            rgb_image,

            conditional_pdfs_integrals,
            conditional_cdfs,

            marginal_pdf_integral,
            marginal_cdf,
        }
    }
}

impl Sampler for InversionSampler {
    fn sample(&self, [u, v]: [f32; 2]) -> (f32, [f32; 2]) {
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
        let dv = (v - self.marginal_cdf[offset_v]) / (self.marginal_cdf[offset_v + 1] - self.marginal_cdf[offset_v]);
        let pdf_v = self.marginal_pdf_integral[offset_v] / self.marginal_pdf_integral[self.height];
        let v_result = (offset_v as f32 + dv) / self.height as f32;

        let offset_u = find_interval(&self.conditional_cdfs[offset_v], u);
        let du = (u - self.conditional_cdfs[offset_v][offset_u]) / (self.conditional_cdfs[offset_v][offset_u + 1] - self.conditional_cdfs[offset_v][offset_u]);
        let pdf_u = self.conditional_pdfs_integrals[offset_v][offset_u] / self.conditional_pdfs_integrals[offset_v][self.width];
        let u_result = (offset_u as f32 + du) / self.width as f32;

        (pdf_v * pdf_u, [u_result, v_result])
    }

    fn pdf(&self, [u, v]: [f32; 2]) -> f32 {
        let iu = ((u * self.width as f32) as usize).clamp(0, self.width - 1);
        let iv = ((v * self.height as f32) as usize).clamp(0, self.height - 1);

        self.conditional_pdfs_integrals[iv][iu] / self.marginal_pdf_integral[self.height]
    }
}

