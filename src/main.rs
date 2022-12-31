#![feature(iter_array_chunks)]

use exr::prelude::*;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use pdf_maker::data2d::Data2D;
use pdf_maker::distribution::Distribution2D;
use pdf_maker::inversion::Inversion1D;
use pdf_maker::alias::Alias1D;
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

    {
        let preprocess_start = Instant::now();
        let sampler = Adapter2D::<Inversion1D>::build(&density_image);
        println!("Took {} seconds for inversion method preprocess", preprocess_start.elapsed().as_secs_f32());

        let mut demo_image = source_image.clone();
        let sampling_start = Instant::now();
        sampler.fill_demo_image(&mut demo_image, rands.clone().into_iter());
        println!("Took {} seconds for inversion method sampling", sampling_start.elapsed().as_secs_f32());

        //let warp = sampler.visualize_warping();

        write_rgb_file("inversion_demo.exr", demo_image.width(), demo_image.height(), |x, y| {
            let p = demo_image[y][x];
            (p[0], p[1], p[2])
        }).unwrap();

        //write_rgb_file("inversion_warping.exr", warp.width(), warp.height(), |x, y| {
        //    let p = warp[y][x];
        //    (p, p, p)
        //}).unwrap();
    }

    {
        let preprocess_start = Instant::now();
        let sampler = Adapter2D::<Alias1D>::build(&density_image);
        println!("Took {} seconds for alias method preprocess", preprocess_start.elapsed().as_secs_f32());

        let mut demo_image = source_image.clone();
        let start = Instant::now();
        sampler.fill_demo_image(&mut demo_image, rands.clone().into_iter());
        println!("Took {} seconds for alias method sampling", start.elapsed().as_secs_f32());

        let warp = sampler.visualize_warping();

        write_rgb_file("alias_demo.exr", demo_image.width(), demo_image.height(), |x, y| {
            let p = demo_image[y][x];
            (p[0], p[1], p[2])
        }).unwrap();

        write_rgb_file("alias_warping.exr", warp.width(), warp.height(), |x, y| {
            let p = warp[y][x];
            (p, p, p)
        }).unwrap();
    }

    println!("Done!");
}
