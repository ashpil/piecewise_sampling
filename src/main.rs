use exr::prelude::*;
use byteorder::{WriteBytesExt, LittleEndian};
use rand::{rngs::StdRng, SeedableRng, Rng};

struct Sampler {
    width: usize,
    height: usize,

    rgb_image: Vec<Vec<[f32; 3]>>,

    conditional_pdfs: Vec<Vec<f32>>,
    conditional_cdfs: Vec<Vec<f32>>,

    marginal_pdf: Vec<f32>,
    marginal_cdf: Vec<f32>,
}

impl Sampler {
    pub fn new(filepath: impl AsRef<std::path::Path>) -> Self {
        let mut sampler = read_first_rgba_layer_from_file(filepath, |resolution, _| {
            let width = resolution.width();
            let height = resolution.height();
            Self {
                width,
                height,

                rgb_image: vec![vec![[0.0, 0.0, 0.0]; width]; height],

                conditional_pdfs: vec![vec![0.0; width]; height],
                conditional_cdfs: vec![vec![0.0; width + 1]; height],

                marginal_pdf: vec![0.0; height],
                marginal_cdf: vec![0.0; height + 1],
            }
        }, |buffer, pos, (r, g, b, _): (f32, f32, f32, f32)| {
            buffer.rgb_image[pos.y()][pos.x()] = [r, g, b];

            let luminance = r * 0.2126 + g * 0.7152 + b * 0.0722;
            let sin_theta = (std::f32::consts::PI * (pos.y() as f32 + 0.5) / (buffer.height as f32)).sin();
            buffer.conditional_pdfs[pos.y()][pos.x()] = luminance * sin_theta;
        }).unwrap().layer_data.channel_data.pixels;

        let mut conditional_integrals = vec![0.0; sampler.height];

        for (j, row) in sampler.conditional_pdfs.iter_mut().enumerate() {
            for i in 1..sampler.width + 1 {
                sampler.conditional_cdfs[j][i] = sampler.conditional_cdfs[j][i - 1] + row[i - 1] / sampler.width as f32;
            }

            conditional_integrals[j] = sampler.conditional_cdfs[j][sampler.width];
            for i in 0..sampler.width + 1 {
                sampler.conditional_cdfs[j][i] /= conditional_integrals[j];
            }

            for i in 0..sampler.width {
                row[i] /= conditional_integrals[j];
            }
        }

        for (i, v) in sampler.marginal_pdf.iter_mut().enumerate() {
            *v = conditional_integrals[i];
        }

        for i in 1..sampler.height + 1 {
            sampler.marginal_cdf[i] = sampler.marginal_cdf[i - 1] + sampler.marginal_pdf[i - 1] / sampler.height as f32;
        }

        let marginal_integral = sampler.marginal_cdf[sampler.height];
        for i in 0..sampler.height + 1 {
            sampler.marginal_cdf[i] /= marginal_integral;
        }
        for i in 0..sampler.height {
            sampler.marginal_pdf[i] /= marginal_integral;
        }

        sampler
    }

    pub fn sample(&self, u: f32, v: f32) -> (f32, [f32; 2]) {
        let offset_v = find_interval(&self.marginal_cdf, v);
        let dv = (v - self.marginal_cdf[offset_v]) / (self.marginal_cdf[offset_v + 1] - self.marginal_cdf[offset_v]);
        let pdf_v = self.marginal_pdf[offset_v];
        let v_result = (offset_v as f32 + dv) / self.height as f32;

        let offset_u = find_interval(&self.conditional_cdfs[offset_v], u);
        let du = (u - self.conditional_cdfs[offset_v][offset_u]) / (self.conditional_cdfs[offset_v][offset_u + 1] - self.conditional_cdfs[offset_v][offset_u]);
        let pdf_u = self.conditional_pdfs[offset_v][offset_u];
        let u_result = (offset_u as f32 + du) / self.width as f32;

        (pdf_v * pdf_u, [u_result, v_result])
    }
}

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

fn main() {

    let sampler = Sampler::new("image.exr");

    let mut rng = StdRng::seed_from_u64(0);
    let mut copy = sampler.rgb_image.clone();

    for _ in 0..100000 {
        let (_, [x, y]) = sampler.sample(rng.gen(), rng.gen());
        for i in 0..1 {
            for j in 0..1 {
                let height = ((y * sampler.height as f32) as i32 + i).clamp(0, (sampler.height - 1) as i32);
                let width = ((x * sampler.width as f32) as i32 + j).clamp(0, (sampler.width - 1) as i32);
                copy[height as usize][width as usize] = [1000.0, 0.0, 0.0];
            }
        }
    }

    write_rgb_file("sampled.exr", sampler.width, sampler.height, |x, y| {
        let p = copy[y][x];
        (p[0], p[1], p[2])
    }).unwrap();

    {
        let mut buffer = std::fs::File::create("output/conditional_pdfs").unwrap();
        println!("Writing conditional pdfs {}x{}", sampler.conditional_pdfs[0].len(), sampler.conditional_pdfs.len());
        for row in sampler.conditional_pdfs {
            for pixel in row {
                buffer.write_f32::<LittleEndian>(pixel).unwrap();
            }
        }
    }
    {
        let mut buffer = std::fs::File::create("output/conditional_cdfs").unwrap();
        println!("Writing conditional cdfs {}x{}", sampler.conditional_cdfs[0].len(), sampler.conditional_cdfs.len());
        for row in sampler.conditional_cdfs {
            for pixel in row {
                buffer.write_f32::<LittleEndian>(pixel).unwrap();
            }
        }
    }
    {
        let mut buffer = std::fs::File::create("output/marginal_pdf").unwrap();
        println!("Writing marginal pdf {}", sampler.marginal_pdf.len());
        for pixel in sampler.marginal_pdf {
            buffer.write_f32::<LittleEndian>(pixel).unwrap();
        }
    }
    {
        let mut buffer = std::fs::File::create("output/marginal_cdf").unwrap();
        println!("Writing marginal cdf {}", sampler.marginal_cdf.len());
        for pixel in sampler.marginal_cdf {
            buffer.write_f32::<LittleEndian>(pixel).unwrap();
        }
    }


    println!("Done!");
}
