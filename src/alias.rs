use crate::{Distribution1D, Distribution2D};
use crate::utils;

#[derive(Clone, Copy)]
pub struct Entry {
    pdf: f32,
    select: f32,
    alias: u32,
}

pub struct Alias1D {
    pub entries: Vec<Entry>,
}

impl Alias1D {
    // Vose O(n)
    pub fn new(weights: &Vec<f32>) -> Self {
        let n = weights.len();

        // due to the fact that we use f32s, multiplying a [0-1) f32 by about 2 million or so
        // will give us numbers rounded to nearest float, which might be the next integer over, not
        // the actual one
        // would be nice if rust supported other float rounding modes...
        assert!(n < 2_000_000, "Alias1D not relaible for distributions with more than 2,000,000 elements");

        let mut running_weights = vec![0.0; n];
        let mut entries = vec![Entry { pdf: 0.0, select: 0.0, alias: 0 }; n];

        let mut small = Vec::new();
        let mut large = Vec::new();

        let weight_sum = utils::kahan_sum(weights.iter().cloned());

        for (i, weight) in weights.iter().enumerate() {
            let adjusted_weight = (weight * n as f32) / weight_sum;
            entries[i].pdf = adjusted_weight;
            running_weights[i] = adjusted_weight;
            if adjusted_weight < 1.0 {
                small.push(i as u32);
            } else {
                large.push(i as u32);
            }
        }

        while !small.is_empty() && !large.is_empty() {
            let l = small.pop().unwrap();
            let g = large.pop().unwrap();

            entries[l as usize].alias = g;
            entries[l as usize].select = running_weights[l as usize];
            running_weights[g as usize] = (running_weights[g as usize] + running_weights[l as usize]) - 1.0;

            if running_weights[g as usize] < 1.0 {
                small.push(g);
            } else {
                large.push(g);
            }
        }

        while let Some(g) = large.pop() {
            entries[g as usize].select = 1.0;
        }

        while let Some(l) = small.pop() {
            entries[l as usize].select = 1.0;
        }

        Self {
            entries,
        }
    }
}

impl Distribution1D for Alias1D {
    fn sample(&self, u: f32) -> (f32, f32) {
        let scaled: f32 = (self.entries.len() as f32) * u;
        let mut index = scaled as u32;
        let mut entry = self.entries[index as usize];
        let v = scaled - index as f32;
        if entry.select <= v {
            index = entry.alias;
            entry = self.entries[entry.alias as usize];
        }

        (entry.pdf, (index as f32) / (self.entries.len() as f32))
    }

    fn pdf(&self, u: f32) -> f32 {
        let idx = (u * self.entries.len() as f32) as usize;
        self.entries[idx].pdf
    }
}

pub struct Alias2D {
    pub width: usize,
    pub height: usize,

    pub one: Alias1D,
}

impl Alias2D {
    pub fn new(image: &Vec<Vec<f32>>) -> Self {
        let height = image.len();
        let width = image[0].len();

        let one = Alias1D::new(&image.iter().flatten().cloned().collect());

        Self {
            width,
            height,

            one,
        }
    }
}


impl Distribution2D for Alias2D {
    fn sample(&self, [u, v]: [f32; 2]) -> (f32, [f32; 2]) {
        let (pdf, coord) = self.one.sample(u);
        let n = self.width * self.height;
        let index = (n as f32 * coord) as usize;

        let y = index / self.width;
        let x = index % self.width;

        (pdf, [(x as f32) / (self.width as f32), (y as f32) / (self.height as f32)])
    }

    fn pdf(&self, [u, v]: [f32; 2]) -> f32 {
        let n = self.width * self.height;

        let x = (u * self.width as f32) as usize;
        let y = (v * self.height as f32) as usize;

        let index = (y * self.width) + x;
        self.one.pdf(index as f32 / n as f32)
    }
}

