use crate::Sampler;

pub struct AliasSampler {
    pub width: usize,
    pub height: usize,

    table: Vec<(f32, usize, usize)>,
}

// calculates luminance from pixel and height in [0..1]
fn luminance([r, g, b]: [f32; 3], height: f32) -> f32 {
    let luminance = (r as f32) * 0.2126 + (g as f32) * 0.7152 + (b as f32) * 0.0722;
    let sin_theta = (std::f32::consts::PI * height).sin();
    luminance * sin_theta
}

// sort when we know only first element is unsorted
fn sort_first(vec: &mut Vec<(f32, usize)>) {
    // only sort of non-empty
    if let Some((el, v0)) = vec.get(0).cloned() {
        let mut next_index = 1;
        while let Some((next_el, v1)) = vec.get(next_index).cloned() {
            if el >= next_el {
                break;
            } else {
                vec[next_index - 1] = (next_el, v1);
                next_index += 1;
            }
        }
        vec[next_index - 1] = (el, v0);
    }
}

impl AliasSampler {
    // not super efficient but not ungodly terrible construction
    pub fn new(rgb_image: &Vec<Vec<[f32; 3]>>) -> Self {
        let width = rgb_image[0].len();
        let height = rgb_image.len();

        let mut samples = Vec::with_capacity(width * height);
        let mut sum = 0.0;
        let mut c = 0.0; // compensation for Kahan sum
        for (row_index, row) in rgb_image.iter().enumerate() {
            for (col_index, pixel) in row.iter().enumerate() {
                let grayscale_pixel = luminance(*pixel, (row_index as f32 + 0.5) / (height as f32));
                let index = (row_index * width) + col_index;
                samples.push((grayscale_pixel, index));

                let y = grayscale_pixel - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
        }
        for (pixel, _) in &mut samples {
            *pixel /= sum;
        }

        let avg = 1.0 / (width * height) as f32;

        let sort_fn = |(a, _): &(f32, _), (b, _): &(f32, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal).reverse();
        samples.sort_unstable_by(sort_fn);

        let mut table = Vec::with_capacity(width * height);

        // all the math here is in f32s as what we're doing is
        // terrible for floating point precision
        while let Some((w_i, i)) = samples.pop() {
            let len = samples.len();
            if let Some((w_j, j)) = samples.get_mut(0) {
                let t = w_i / avg;
                table.push((t, i, *j));
                *w_j -= avg - w_i;
                if (avg - w_i) < 0.0 {
                    eprintln!("alias: Too many floating point errors at {} to end. Average {}, last {}", len, avg, w_i);
                    eprintln!("alias: Continuing with 1s - should be fine if {} isn't massive.", len);
                    break;
                }
            }
            sort_first(&mut samples);
        }
        while let Some((_, j)) = samples.get(0).cloned() {
            table.push((1.0, j, 0));
            samples.pop();
        }

        AliasSampler {
            width,
            height,

            table,
        }
    }
}


impl Sampler for AliasSampler {
    fn sample(&self, [u, v]: [f32; 2]) -> (f32, [f32; 2]) {
        let table_index = (self.table.len() as f32 * u) as usize;
        let (t, i, j) = self.table[table_index];

        let index = if v < t { i } else { j };

        let pdf = 0.0;

        let j = index / self.width;
        let i = index - (j * self.width);

        (pdf, [(i as f32) / (self.width as f32), (j as f32) / (self.height as f32)])
    }

    fn pdf(&self, _: [f32; 2]) -> f32 {
        0.0 // TODO
    }
}

