#![feature(iter_array_chunks)]

use exr::prelude::*;
use std::time::Instant;

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
        eprintln!("Usage: compare_perf <filename.exr>");
        std::process::exit(1);
    }

    let density_image = read_first_rgba_layer_from_file(args[1].clone(), |resolution, _| {
        let width = resolution.width();
        let height = resolution.height();
        Data2D::new_same(width, height, 0.0)
    }, |buffer, pos, (r, g, b, _): (f32, f32, f32, f32)| {
        buffer[[pos.x(), pos.y()]] = luminance([r, g, b]);
    }).unwrap().layer_data.channel_data.pixels;

    fn sample_perf<D: Discrete2D<f32, Weight=f32>>(name: &str, weights: &Data2D<f32>) {
        println!("{} method", name);

        let sample_count_per_dimension = 1000;
        let total_sample_count = sample_count_per_dimension * sample_count_per_dimension;
        let incoherent: Vec<[f32; 2]> = StdRng::seed_from_u64(0).sample_iter(rand::distributions::Uniform::new(0.0, 1.0)).take(total_sample_count * 2).array_chunks::<2>().collect();
        let coherent: Vec<[f32; 2]> = (0..sample_count_per_dimension).map(|j| (0..sample_count_per_dimension).map(move |i| [(i as f32) / sample_count_per_dimension as f32, (j as f32) / sample_count_per_dimension as f32])).flatten().collect();

        let sampler = {
            let preprocess_start = Instant::now();
            let sampler = D::build(weights);
            println!("  {: >3}ms for build", preprocess_start.elapsed().as_millis());
            sampler
        };

        // TODO: investigate why black_box only make a noticable difference for alias
        {
            let sampling_start = Instant::now();
            for value in coherent {
                let input = std::hint::black_box(value);
                let output = sampler.sample(input);
                std::hint::black_box(output);
            }
            let elapsed = sampling_start.elapsed();
            println!("  {: >3}ns per sample for coherent sampling", elapsed.as_nanos() / total_sample_count as u128);
        }

        {
            let sampling_start = Instant::now();
            for value in incoherent {
                let input = std::hint::black_box(value);
                let output = sampler.sample(input);
                std::hint::black_box(output);
            }
            let elapsed = sampling_start.elapsed();
            println!("  {: >3}ns per sample for incoherent sampling", elapsed.as_nanos() / total_sample_count as u128);
        }
    }

    sample_perf::<Inversion2D<f32>>("Inversion", &density_image);
    sample_perf::<Alias2D<f32>>("Alias", &density_image);
    sample_perf::<Hierarchical2D<f32>>("Hierarchical", &density_image);
}

