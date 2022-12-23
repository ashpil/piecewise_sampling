#![feature(iter_array_chunks)]

use exr::prelude::*;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod inversion;
mod alias;

use inversion::Inversion2D;
use alias::Alias2D;

pub trait Distribution1D {
    // takes in rand [0-1), returns (pdf, selected [0-1))
    fn sample(&self, u: f32) -> (f32, f32);

    // takes in coord [0-1), returns pdf
    fn pdf(&self, u: f32) -> f32;
}

pub trait Distribution2D {
    // takes in rand [0-1)x[0-1), returns (pdf, uv coords)
    fn sample(&self, uv: [f32; 2]) -> (f32, [f32; 2]);

    // takes in coords [0-1)x[0-1), returns pdf
    fn pdf(&self, uv: [f32; 2]) -> f32;

    // fills demo image with sample_count samples
    fn fill_demo_image(&self, demo: &mut Vec<Vec<[f32; 3]>>, rngs: impl Iterator<Item = [f32; 2]>) {
        let width = demo[0].len();
        let height = demo.len();

        for rng in rngs {
            let (pdf, [x, y]) = self.sample(rng);
            let mpdf = self.pdf([x, y]);
            if (pdf - mpdf).abs() > 0.01 {
                panic!("Something wrong: got {} as pdf, but reconstructed {} at {}x{}", pdf, mpdf, x, y);
            }
            for is in -1..1 {
                for js in -1..1 {
                    let j = ((y * height as f32) as i32 + js).clamp(0, (height - 1) as i32) as usize;
                    let i = ((x * width as f32) as i32 + is).clamp(0, (width - 1) as i32) as usize;
                    demo[j][i] = [1000.0, 0.0, 0.0];
                }
            }
        }
    }
}

// calculates luminance from pixel and height in [0..1]
fn luminance([r, g, b]: [f32; 3]) -> f32 {
    return r * 0.2126 + g * 0.7152 + b * 0.0722;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: pdf_maker filename.exr.");
        std::process::exit(1);
    }

    let source_image = read_first_rgba_layer_from_file(args[1].clone(), |resolution, _| {
        let width = resolution.width();
        let height = resolution.height();
        vec![vec![[0.0, 0.0, 0.0]; width]; height]
    }, |buffer, pos, (r, g, b, _): (f32, f32, f32, f32)| {
        buffer[pos.y()][pos.x()] = [r, g, b];
    }).unwrap().layer_data.channel_data.pixels;

    // assuming environment map
    let density_image = source_image.iter().enumerate().map(|(row_idx, row)| {
        let sin_theta = (std::f32::consts::PI * (row_idx as f32 + 0.5) / (source_image.len() as f32)).sin();
        row.iter().map(|c| luminance(*c) * sin_theta).collect()
    }).collect();

    let sample_count = 100_000;
    let stratify = true;
    let rands: Vec<[f32; 2]> = if stratify {
        let mut rands = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            rands.push([sobol_burley::sample(i as u32, 0, 0), sobol_burley::sample(i as u32, 1, 0)]);
        }
        rands
    } else {
        StdRng::seed_from_u64(0).sample_iter(rand::distributions::Uniform::new(0.0, 1.0)).take(sample_count * 2).array_chunks::<2>().collect()
    };

    {
        let preprocess_start = Instant::now();
        let sampler = Inversion2D::new(&density_image, 1);
        println!("Took {} seconds for inversion method preprocess", preprocess_start.elapsed().as_secs_f32());

        let mut demo_image = source_image.clone();
        let sampling_start = Instant::now();
        sampler.fill_demo_image(&mut demo_image, rands.clone().into_iter());
        println!("Took {} seconds for inversion method sampling", sampling_start.elapsed().as_secs_f32());

        write_rgb_file("inversion_demo.exr", demo_image[0].len(), demo_image.len(), |x, y| {
            let p = demo_image[y][x];
            (p[0], p[1], p[2])
        }).unwrap();
    }

    {
        let preprocess_start = Instant::now();
        let sampler = Alias2D::new(&density_image);
        println!("Took {} seconds for alias method preprocess", preprocess_start.elapsed().as_secs_f32());

        let mut demo_image = source_image.clone();
        let start = Instant::now();
        sampler.fill_demo_image(&mut demo_image, rands.clone().into_iter());
        println!("Took {} seconds for alias method sampling", start.elapsed().as_secs_f32());

        write_rgb_file("alias_demo.exr", demo_image[0].len(), demo_image.len(), |x, y| {
            let p = demo_image[y][x];
            (p[0], p[1], p[2])
        }).unwrap();
    }

    println!("Done!");
}
