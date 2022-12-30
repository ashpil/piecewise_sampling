use crate::distribution::Distribution1D;
use crate::utils;

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pdf: f32,
    select: f32,
    alias: u32,
}

pub struct Alias1D {
    pub weight_sum: f32,
    pub entries: Box<[Entry]>,
}

impl Distribution1D for Alias1D {
    // Vose O(n)
    fn build(weights: &[f32]) -> Self {
        let n = weights.len();

        // due to the fact that we use f32s, multiplying a [0-1) f32 by about 2 million or so
        // will give us numbers rounded to nearest float, which might be the next integer over, not
        // the actual one
        // would be nice if rust supported other float rounding modes...
        assert!(n < 2_000_000, "Alias1D not reliable for distributions with more than 2,000,000 elements");

        let mut entries = vec![Entry { pdf: 0.0, select: 0.0, alias: 0 }; n].into_boxed_slice();

        let mut small = Vec::new();
        let mut large = Vec::new();

        let weight_sum = utils::kahan_sum(weights.iter().cloned());

        for (i, weight) in weights.iter().enumerate() {
            let adjusted_weight = (weight * n as f32) / weight_sum;
            entries[i].pdf = *weight;
            entries[i].select = adjusted_weight;
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
            entries[g as usize].select = (entries[g as usize].select + entries[l as usize].select) - 1.0;

            if entries[g as usize].select < 1.0 {
                small.push(g);
            } else {
                large.push(g);
            }
        }

        // the select for entries in `large` should already all be >= 1.0, so 
        // we don't need to update them here
        //
        //while let Some(g) = large.pop() {
        //    entries[g as usize].select = 1.0;
        //}

        // these are actually large but are in small due to float error
        // should be slightly less than 1.0, we need to make sure they're 1.0
        while let Some(l) = small.pop() {
            entries[l as usize].select = 1.0;
        }

        Self {
            weight_sum,
            entries,
        }
    }

    fn sample_discrete(&self, u: f32) -> (f32, usize) {
        let scaled: f32 = (self.entries.len() as f32) * u;
        let mut index = scaled as usize;
        let mut entry = self.entries[index];
        let v = scaled - index as f32;
        if entry.select < v {
            index = entry.alias as usize;
            entry = self.entries[entry.alias as usize];
        }

        (entry.pdf, index)
    }

    fn sample_continuous(&self, u: f32) -> (f32, f32) {
        let scaled: f32 = (self.entries.len() as f32) * u;
        let mut index = scaled as usize;
        let mut entry = self.entries[index];
        let v = scaled - index as f32;
        let mut du = v / entry.select;

        if entry.select < v {
            index = entry.alias as usize;
            entry = self.entries[entry.alias as usize];
            du = v / (1.0 - entry.select);
        }

        (entry.pdf, (index as f32 + du) / self.entries.len() as f32)
    }

    fn pdf(&self, u: usize) -> f32 {
        self.entries[u].pdf
    }

    fn integral(&self) -> f32 {
        self.weight_sum
    }

    fn size(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::distribution_1d_tests;
    distribution_1d_tests!(crate::alias::Alias1D);
}

