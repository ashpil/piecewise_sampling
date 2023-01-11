use exr::prelude::*;
use std::time::Instant;

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
        buffer[pos.y()][pos.x()] = luminance([r, g, b]);
    }).unwrap().layer_data.channel_data.pixels;

    fn sample_perf<D: Discrete2D<f32, Weight=f32>>(name: &str, weights: &Data2D<f32>) {
        println!("{} method", name);

        let sampler = {
            let preprocess_start = Instant::now();
            let sampler = D::build(weights);
            println!("  {} seconds for build", preprocess_start.elapsed().as_secs_f32());
            sampler
        };

        {
            let sampling_start = Instant::now();
            let size = 1000;
            for j in 0..size {
                for i in 0..size {
                    let input = [(i as f32) / size as f32, (j as f32) / size as f32];
                    let output = sampler.sample(input);
                    std::hint::black_box(output); // TODO: why do we get a big perf difference with our without this for alias only?
                }
            }
            println!("  {} seconds for sampling", sampling_start.elapsed().as_secs_f32());
        }
    }

    sample_perf::<Inversion2D<f32>>("Inversion", &density_image);
    sample_perf::<Alias2D<f32>>("Alias", &density_image);
    sample_perf::<Hierarchical2D<f32>>("Hierarchical", &density_image);
}

