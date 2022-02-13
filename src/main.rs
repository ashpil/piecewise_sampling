use exr::prelude::*;
use byteorder::{WriteBytesExt, LittleEndian};
use rand::{rngs::StdRng, SeedableRng, Rng};

mod inversion_sampler;
use inversion_sampler::InversionSampler;

// calculates luminance from pixel and height in [0..1]
pub fn luminance([r, g, b]: [f32; 3], height: f32) -> f32 {
    let luminance = r * 0.2126 + g * 0.7152 + b * 0.0722;
    let sin_theta = (std::f32::consts::PI * height).sin();
    luminance * sin_theta
}

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
                panic!("Something wrong: got {} as pdf, but reconstructed {}", pdf, mpdf);
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

    let sampler = InversionSampler::new(args[1].clone());

    let mut rng = StdRng::seed_from_u64(0);
    let mut demo = sampler.rgb_image.clone();

    sampler.fill_demo_image(&mut demo, &mut rng, 100000);

    // if error, likely output dir already created, so we ignore that
    let _ = std::fs::create_dir("output");

    write_rgb_file("output/demo.exr", sampler.rgb_image[0].len(), sampler.rgb_image.len(), |x, y| {
        let p = demo[y][x];
        (p[0], p[1], p[2])
    }).unwrap();

    fn write_bytes_to_file(name: &str, data: Vec<Vec<f32>>) {
        let mut buffer = std::fs::File::create(format!("output/{}.raw", name.replace(" ", "_"))).unwrap();
        println!("Writing {} {}x{}", name, data[0].len(), data.len());
        for row in data {
            for pixel in row {
                buffer.write_f32::<LittleEndian>(pixel).unwrap();
            }
        }
    }

    write_bytes_to_file("conditional pdfs integrals", sampler.conditional_pdfs_integrals);
    write_bytes_to_file("conditional cdfs", sampler.conditional_cdfs);
    write_bytes_to_file("marginal pdf integral", vec![sampler.marginal_pdf_integral]);
    write_bytes_to_file("marginal cdf", vec![sampler.marginal_cdf]);

    println!("Done!");
}
