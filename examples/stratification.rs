#![feature(iter_array_chunks)]

use exr::prelude::*;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use discrete_sampling::Data2D;
use discrete_sampling::Alias2D;
use discrete_sampling::Inversion2D;
use discrete_sampling::Hierarchical2D;
use discrete_sampling::distribution::Discrete2D;

fn luminance([r, g, b]: [f32; 3]) -> f32 {
    r * 0.2126 + g * 0.7152 + b * 0.0722
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: stratification <filename.exr>");
        std::process::exit(1);
    }

    let source_image_path = args[1].clone();

    let source_image = read_first_rgba_layer_from_file(source_image_path, |resolution, _| {
        let width = resolution.width();
        let height = resolution.height();
        Data2D::new_same(width, height, [0.0, 0.0, 0.0])
    }, |buffer, pos, (r, g, b, _): (f32, f32, f32, f32)| {
        buffer[[pos.x(), pos.y()]] = [r, g, b];
    }).unwrap().layer_data.channel_data.pixels;

    let mut density_image = Data2D::new_same(source_image.width(), source_image.height(), 0.0);
    for (row_idx, row) in source_image.iter().enumerate() {
        for (col_idx, pixel) in row.iter().enumerate() {
            density_image[[col_idx, row_idx]] = luminance(*pixel);
        }
    }

    let sample_count = source_image.width() + source_image.height(); // empirically seems to be a reasonable default
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

    fn demo_distribution<D: Discrete2D<f32, Weight=f32>>(out_name: &str, source_image: &Data2D<[f32; 3]>, weights: &Data2D<f32>, rands: &[[f32; 2]]) {
        let sampler = D::build(weights);

        let mut demo_image = source_image.clone();

        sampler.fill_demo_image(&mut demo_image, rands.iter().cloned());

        write_rgb_file(out_name, demo_image.width(), demo_image.height(), |x, y| {
            let p = demo_image[[x, y]];
            (p[0], p[1], p[2])
        }).unwrap();
        println!("Wrote {}", out_name);
    }

    demo_distribution::<Inversion2D<f32>>("inversion_demo.exr", &source_image, &density_image, &rands);
    demo_distribution::<Alias2D<f32>>("alias_demo.exr", &source_image, &density_image, &rands);
    demo_distribution::<Hierarchical2D<f32>>("hierarchical_demo.exr", &source_image, &density_image, &rands);
}

