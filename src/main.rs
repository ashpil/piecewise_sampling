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

        let mut conditional_pdfs: Vec<Vec<f32>> = vec![vec![0.0; width]; height];
        let mut conditional_cdfs = vec![vec![0.0; width + 1]; height];

        let mut marginal_pdf = vec![0.0; height];
        let mut marginal_cdf = vec![0.0; height + 1];

        for (row_index, row) in conditional_pdfs.iter_mut().enumerate() {
            for (col_index, pixel) in row.iter_mut().enumerate() {
                for i in (row_index * factor)..((row_index + 1) * factor) {
                    for j in (col_index * factor)..((col_index + 1) * factor) {
                        let [r, g, b] = rgb_image[i][j];
                        let luminance = r * 0.2126 + g * 0.7152 + b * 0.0722;
                        let sin_theta = (std::f32::consts::PI * (i as f32 + 0.5) / (big_height as f32)).sin();
                        *pixel = (*pixel).max(luminance * sin_theta);
                    }
                }
            }
        }

        write_rgb_file("hmm.exr", width, height, |x, y| {
            let p = conditional_pdfs[y][x];
            (p, p, p)
        }).unwrap();

        let mut conditional_integrals = vec![0.0; height];

        for (j, row) in conditional_pdfs.iter_mut().enumerate() {
            for i in 1..width + 1 {
                conditional_cdfs[j][i] = conditional_cdfs[j][i - 1] + row[i - 1] / width as f32;
            }

            conditional_integrals[j] = conditional_cdfs[j][width];
            for i in 0..width + 1 {
                conditional_cdfs[j][i] /= conditional_integrals[j];
            }

            for i in 0..width {
                row[i] /= conditional_integrals[j];
            }
        }

        for (i, v) in marginal_pdf.iter_mut().enumerate() {
            *v = conditional_integrals[i];
        }

        for i in 1..height + 1 {
            marginal_cdf[i] = marginal_cdf[i - 1] + marginal_pdf[i - 1] / height as f32;
        }

        let marginal_integral = marginal_cdf[height];
        for i in 0..height + 1 {
            marginal_cdf[i] /= marginal_integral;
        }
        for i in 0..height {
            marginal_pdf[i] /= marginal_integral;
        }

        Sampler {
            width,
            height,

            rgb_image,

            conditional_pdfs,
            conditional_cdfs,

            marginal_pdf,
            marginal_cdf,
        }
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
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: pdf_maker filename.exr.");
        eprintln!("Result in output/ directory.");
        std::process::exit(1);
    }

    let sampler = Sampler::new(args[1].clone());

    let mut rng = StdRng::seed_from_u64(0);
    let mut copy = sampler.rgb_image.clone();

    let big_width = sampler.rgb_image[0].len();
    let big_height = sampler.rgb_image.len();

    for _ in 0..100000 {
        let (_, [x, y]) = sampler.sample(rng.gen(), rng.gen());
        for i in -2..2 {
            for j in -2..2 {
                let height = ((y * big_height as f32) as i32 + i).clamp(0, (big_height - 1) as i32);
                let width = ((x * big_width as f32) as i32 + j).clamp(0, (big_width - 1) as i32);
                copy[height as usize][width as usize] = [1000.0, 0.0, 0.0];
            }
        }
    }

    let _ = std::fs::create_dir("output");

    write_rgb_file("output/demo.exr", big_width, big_height, |x, y| {
        let p = copy[y][x];
        (p[0], p[1], p[2])
    }).unwrap();

    {
        let mut buffer = std::fs::File::create("output/conditional_pdfs.raw").unwrap();
        println!("Writing conditional pdfs {}x{}", sampler.conditional_pdfs[0].len(), sampler.conditional_pdfs.len());
        for row in sampler.conditional_pdfs {
            for pixel in row {
                buffer.write_f32::<LittleEndian>(pixel).unwrap();
            }
        }
    }
    {
        let mut buffer = std::fs::File::create("output/conditional_cdfs.raw").unwrap();
        println!("Writing conditional cdfs {}x{}", sampler.conditional_cdfs[0].len(), sampler.conditional_cdfs.len());
        for row in sampler.conditional_cdfs {
            for pixel in row {
                buffer.write_f32::<LittleEndian>(pixel).unwrap();
            }
        }
    }
    {
        let mut buffer = std::fs::File::create("output/marginal_pdf.raw").unwrap();
        println!("Writing marginal pdf {}", sampler.marginal_pdf.len());
        for pixel in sampler.marginal_pdf {
            buffer.write_f32::<LittleEndian>(pixel).unwrap();
        }
    }
    {
        let mut buffer = std::fs::File::create("output/marginal_cdf.raw").unwrap();
        println!("Writing marginal cdf {}", sampler.marginal_cdf.len());
        for pixel in sampler.marginal_cdf {
            buffer.write_f32::<LittleEndian>(pixel).unwrap();
        }
    }


    println!("Done!");
}
