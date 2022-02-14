use exr::prelude::*;
use byteorder::{WriteBytesExt, LittleEndian};
use rand::{rngs::StdRng, SeedableRng, Rng};
use std::time::Instant;

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
    fn fill_demo_image(&self, demo: &mut Vec<Vec<[f32; 3]>>, rng: &mut impl Rng, sample_count: usize) {
        let width = demo[0].len();
        let height = demo.len();

        for _ in 0..sample_count {
            let (pdf, [x, y]) = self.sample([rng.gen(), rng.gen()]);
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


fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: pdf_maker filename.exr.");
        eprintln!("Result in output/ directory.");
        std::process::exit(1);
    }

    let mut rgb_image = read_first_rgba_layer_from_file(args[1].clone(), |resolution, _| {
        let width = resolution.width();
        let height = resolution.height();
        vec![vec![[0.0, 0.0, 0.0]; width]; height]
    }, |buffer, pos, (r, g, b, _): (f32, f32, f32, f32)| {
        buffer[pos.y()][pos.x()] = [r, g, b];
    }).unwrap().layer_data.channel_data.pixels;
    let mut rgb_image2 = rgb_image.clone();

    let sampler = InversionSampler::new(&rgb_image, 4);
    let sampler2 = AliasSampler::new(&rgb_image2);

    let mut rng = StdRng::seed_from_u64(0);

    let now1 = Instant::now();
    sampler.fill_demo_image(&mut rgb_image, &mut rng, 100000);
    println!("Took {} for inversion method", now1.elapsed().as_secs_f32());

    let now2 = Instant::now();
    sampler2.fill_demo_image(&mut rgb_image2, &mut rng, 100000);
    println!("Took {} for alias method", now2.elapsed().as_secs_f32());


    // if error, likely output dir already created, so we ignore that
    let _ = std::fs::create_dir("output");

    write_rgb_file("output/demo.exr", rgb_image[0].len(), rgb_image.len(), |x, y| {
        let p = rgb_image[y][x];
        (p[0], p[1], p[2])
    }).unwrap();

    write_rgb_file("output/demo2.exr", rgb_image[0].len(), rgb_image.len(), |x, y| {
        let p = rgb_image2[y][x];
        (p[0], p[1], p[2])
    }).unwrap();

    fn write_f32_to_file(name: &str, data: Vec<Vec<f32>>) {
        let mut buffer = std::fs::File::create(format!("output/{}.raw", name.replace(" ", "_"))).unwrap();
        println!("Writing f32s {} {}x{}", name, data[0].len(), data.len());
        for row in data {
            for pixel in row {
                buffer.write_f32::<LittleEndian>(pixel).unwrap();
            }
        }
    }

    fn write_u32_to_file(name: &str, data: Vec<u32>) {
        let mut buffer = std::fs::File::create(format!("output/{}.raw", name.replace(" ", "_"))).unwrap();
        println!("Writing u32s {} {}", name, data.len());
        for pixel in data {
            buffer.write_u32::<LittleEndian>(pixel).unwrap();
        }
    }

    write_f32_to_file("conditional pdfs integrals", sampler.conditional_pdfs_integrals);
    write_f32_to_file("conditional cdfs", sampler.conditional_cdfs);
    write_f32_to_file("marginal pdf integral", vec![sampler.marginal_pdf_integral]);
    write_f32_to_file("marginal cdf", vec![sampler.marginal_cdf]);

    write_f32_to_file("tau table", vec![sampler2.tau_table]);
    write_f32_to_file("pdf table", vec![sampler2.pdf_table]);
    write_u32_to_file("i table", sampler2.i_table);
    write_u32_to_file("j table", sampler2.j_table);

    println!("Done!");
}
