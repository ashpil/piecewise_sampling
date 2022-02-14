use crate::Sampler;

pub struct AliasSampler {
    pub width: usize,
    pub height: usize,

    pub tau_table: Vec<f32>,
    pub pdf_table: Vec<f32>,
    pub i_table: Vec<u32>,
    pub j_table: Vec<u32>,
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

        let pdf_table = samples.iter().map(|(s, _)| *s).collect();

        let avg = 1.0 / (width * height) as f32;

        samples.sort_unstable_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal).reverse());

        let mut tau_table = Vec::with_capacity(width * height);
        let mut i_table = Vec::with_capacity(width * height);
        let mut j_table = Vec::with_capacity(width * height);

        // all the math here is in f32s as what we're doing is
        // terrible for floating point precision
        while let Some((w_i, i)) = samples.pop() {
            let len = samples.len();
            if let Some((w_j, j)) = samples.get_mut(0) {
                let tau = w_i / avg;
                tau_table.push(tau);
                i_table.push(i as u32);
                j_table.push(*j as u32);
                *w_j -= avg - w_i;
                if (avg - w_i) < 0.0 {
                    eprintln!("alias: Too many floating point errors at {} to end. Average {}, last {}", len, avg, w_i);
                    eprintln!("alias: Continuing with 1s - should be fine if {} isn't massive.", len);
                    break;
                }
            }
            sort_first(&mut samples);
        }
        while let Some((_, i)) = samples.get(0).cloned() {
            tau_table.push(1.0);
            i_table.push(i as u32);
            j_table.push(0);
            samples.pop();
        }

        AliasSampler {
            width,
            height,

            tau_table,
            pdf_table,
            i_table,
            j_table,
        }
    }
}


impl Sampler for AliasSampler {
    fn sample(&self, [u, v]: [f32; 2]) -> (f32, [f32; 2]) {
        let table_index = (self.tau_table.len() as f32 * u) as usize;
        let tau = self.tau_table[table_index];

        let index = if v < tau { self.i_table[table_index] } else { self.j_table[table_index] };
        let pdf = self.pdf_table[index as usize];

        let y = index / self.width as u32;
        let x = index - (y * self.width as u32);


        (pdf, [(x as f32) / (self.width as f32), (y as f32) / (self.height as f32)])
    }

    fn pdf(&self, [u, v]: [f32; 2]) -> f32 {
        let x = (u * self.width as f32) as usize;
        let y = (v * self.height as f32) as usize;

        self.pdf_table[(y * self.width) + x]
    }
}

