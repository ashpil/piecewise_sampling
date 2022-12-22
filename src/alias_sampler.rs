use crate::Sampler;

#[derive(Clone, Copy)]
pub struct Entry {
    pdf: f32,
    select: f32,
    alias: u32,
}

pub struct AliasSampler {
    pub width: usize,
    pub height: usize,

    pub entries: Vec<Entry>,
}

impl AliasSampler {
    // Vose O(n)
    pub fn new(image: &Vec<Vec<f32>>) -> Self {
        let height = image.len();
        let width = image[0].len();
        let n = width * height;

        let mut running_weights = vec![0.0; n];

        let mut entries = vec![Entry { pdf: 0.0, select: 0.0, alias: 0 }; n];

        let mut small = Vec::new();
        let mut large = Vec::new();

        let weight_sum: f32 = image.iter().enumerate().map(|(i, row)| row.iter().sum::<f32>() * (std::f32::consts::PI * (i as f32 + 0.5) / (height as f32)).sin()).sum();

        for (row_idx, row) in image.iter().enumerate() {
            for (col_idx, weight) in row.iter().enumerate() {
                let adjusted_weight = ((weight * n as f32) / weight_sum) * (std::f32::consts::PI * (row_idx as f32 + 0.5) / (height as f32)).sin();
                let i = row_idx * width + col_idx;
                entries[i].pdf = adjusted_weight;
                running_weights[i] = adjusted_weight;
                if adjusted_weight < 1.0 {
                    small.push(i as u32);
                } else {
                    large.push(i as u32);
                }
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

        AliasSampler {
            width,
            height,

            entries,
        }
    }
}


impl Sampler for AliasSampler {
    fn sample(&self, [u, v]: [f32; 2]) -> (f32, [f32; 2]) {
        let n = self.width * self.height;
        let mut index = (n as f32 * u) as u32;
        let mut entry = self.entries[index as usize];
        if entry.select < v {
            index = entry.alias;
            entry = self.entries[entry.alias as usize];
        }

        let y = index / self.width as u32;
        let x = index - (y * self.width as u32);

        let pdf = entry.pdf;
        (pdf, [(x as f32) / (self.width as f32), (y as f32) / (self.height as f32)])
    }

    fn pdf(&self, [u, v]: [f32; 2]) -> f32 {
        let x = (u * self.width as f32) as usize;
        let y = (v * self.height as f32) as usize;

        self.entries[(y * self.width) + x].pdf
    }
}

