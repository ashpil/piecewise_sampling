#![feature(iter_array_chunks)]

use exr::prelude::*;
use std::time::Instant;

use sobol::Sobol;
use sobol::params::JoeKuoD6;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod inversion_sampler;
use inversion_sampler::InversionSampler;

mod alias_sampler;
use alias_sampler::AliasSampler;

pub trait Sampler {
    // takes in rand uv, returns (pdf, uv coords)
    fn sample(&self, uv: [f32; 2]) -> (f32, [f32; 2]);

    // takes in uv coords, returns pdf
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

    let luminance_image = source_image.iter().map(|r| r.iter().map(|c| luminance(*c)).collect()).collect();

    let sample_count = 10_000;
    let stratify = false;
    let rands: Vec<[f32; 2]> = if stratify {
         Sobol::<f32>::new(2, &JoeKuoD6::minimal()).take(sample_count).map(|v| [v[0], v[1]]).collect()
    } else {
        StdRng::seed_from_u64(0).sample_iter(rand::distributions::Uniform::new(0.0, 1.0)).take(sample_count * 2).array_chunks::<2>().collect()
    };

    {
        let preprocess_start = Instant::now();
        let sampler = InversionSampler::new(&luminance_image, 1);
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
        let sampler = AliasSampler::new(&luminance_image);
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
