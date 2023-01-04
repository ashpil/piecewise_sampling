#![feature(iter_array_chunks)]

use exr::prelude::*;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use pdf_maker::data2d::Data2D;
use pdf_maker::distribution::ContinuousDistribution2D;
use pdf_maker::inversion::Inversion1D;
use pdf_maker::alias::ContinuousAlias1D;
use pdf_maker::hierarchical::Hierarchical1D;
use pdf_maker::adapter2d::Adapter2D;

fn luminance([r, g, b]: [f32; 3]) -> f32 {
    r * 0.2126 + g * 0.7152 + b * 0.0722
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
        Data2D::new_same(width, height, [0.0, 0.0, 0.0])
    }, |buffer, pos, (r, g, b, _): (f32, f32, f32, f32)| {
        buffer[pos.y()][pos.x()] = [r, g, b];
    }).unwrap().layer_data.channel_data.pixels;

    // assuming environment map
    let mut density_image = Data2D::new_same(source_image.width(), source_image.height(), 0.0);
    for (row_idx, row) in source_image.iter().enumerate() {
        let sin_theta = (std::f32::consts::PI * (row_idx as f32 + 0.5) / (source_image.height() as f32)).sin();
        for (col_idx, pixel) in row.iter().enumerate() {
            density_image[row_idx][col_idx] = luminance(*pixel) * sin_theta;
        }
    }

    let sample_count = 65_536;
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

    fn demo_distribution<D: ContinuousDistribution2D<Weight=f32>>(name: &str, source_image: &Data2D<[f32; 3]>, weights: &Data2D<f32>, rands: &Vec<[f32; 2]>) {

        println!("{} method", name);

        let sampler = {
            let preprocess_start = Instant::now();
            let sampler = D::build(weights);
            println!("  {} seconds for build", preprocess_start.elapsed().as_secs_f32());
            sampler
        };

        {
            let mut demo_image = source_image.clone();

            let sampling_start = Instant::now();
            sampler.fill_demo_image(&mut demo_image, rands.clone().into_iter());
            println!("  {} seconds for sampling", sampling_start.elapsed().as_secs_f32());

            write_rgb_file(format!("{}_demo.exr", name), demo_image.width(), demo_image.height(), |x, y| {
                let p = demo_image[y][x];
                (p[0], p[1], p[2])
            }).unwrap();
        }

        {
            let warping_start = Instant::now();
            let warping = sampler.visualize_warping(16);
            println!("  {} seconds for warping visualization", warping_start.elapsed().as_secs_f32());

            write_rgb_file(format!("{}_warping.exr", name), warping.width(), warping.height(), |x, y| {
                let p = warping[y][x];
                (p[0], p[1], p[2])
            }).unwrap();
        }
    }

    demo_distribution::<Adapter2D::<Inversion1D<f32>>>("Inversion", &source_image, &density_image, &rands);
    demo_distribution::<Adapter2D::<ContinuousAlias1D<f32>>>("Alias", &source_image, &density_image, &rands);
    demo_distribution::<Adapter2D::<Hierarchical1D<f32>>>("Hierarchical", &source_image, &density_image, &rands);

    println!("Done!");
}
