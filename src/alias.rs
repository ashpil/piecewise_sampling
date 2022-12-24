use crate::{Distribution1D, Distribution2D};
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
        assert!(n < 2_000_000, "Alias1D not relaible for distributions with more than 2,000,000 elements");

        let mut running_weights = vec![0.0; n];
        let mut entries = vec![Entry { pdf: 0.0, select: 0.0, alias: 0 }; n];

        let mut small = Vec::new();
        let mut large = Vec::new();

        let weight_sum = utils::kahan_sum(weights.iter().cloned());

        for (i, weight) in weights.iter().enumerate() {
            let adjusted_weight = (weight * n as f32) / weight_sum;
            entries[i].pdf = *weight;
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
            weight_sum,
            entries: entries.into_boxed_slice(),
        }
    }

    fn sample(&self, u: f32) -> (f32, usize) {
        let scaled: f32 = (self.entries.len() as f32) * u;
        let mut index = scaled as usize;
        let mut entry = self.entries[index];
        let v = scaled - index as f32;
        if entry.select <= v {
            index = entry.alias as usize;
            entry = self.entries[entry.alias as usize];
        }

        (entry.pdf, index)
    }

    fn pdf(&self, u: usize) -> f32 {
        self.entries[u].pdf
    }
}

pub struct Alias2D {
    pub marginal: Alias1D,
    pub conditional: Box<[Alias1D]>,
}

impl Distribution2D for Alias2D {
    fn build(weights: &[Vec<f32>]) -> Self {
        let mut conditional = Vec::with_capacity(weights.len());
        let mut marginal_weights = Vec::with_capacity(weights.len());
        for row in weights {
            let table = Alias1D::build(row);
            marginal_weights.push(table.weight_sum);
            conditional.push(table);
        }

        let marginal = Alias1D::build(&marginal_weights);

        Self {
            marginal,
            conditional: conditional.into_boxed_slice(),
        }
    }

    fn sample(&self, [u, v]: [f32; 2]) -> (f32, [usize; 2]) {
        let (pdf_y, y) = self.marginal.sample(u);
        let (pdf_x, x) = self.conditional[y].sample(v);

        (pdf_x * pdf_y, [x, y])
    }

    fn pdf(&self, [u, v]: [usize; 2]) -> f32 {
        let pdf_y = self.marginal.pdf(v);
        let pdf_x = self.conditional[v].pdf(u);

        return pdf_y * pdf_x;
    }
}

#[cfg(test)]
mod tests {
    use super::Alias1D;
    use crate::Distribution1D;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    fn test_distribution(expected: &[f32], sample_count: usize) {
        let dist = Alias1D::build(expected);
        let mut observed = vec![0.0f32; expected.len()].into_boxed_slice();
        let mut hist = vec![0u32; expected.len()].into_boxed_slice();
        let mut rng = StdRng::seed_from_u64(0);

        for _ in 0..sample_count {
            let (pdf, idx) = dist.sample(rng.gen());
            assert_eq!(expected[idx], pdf);
            assert_eq!(pdf, dist.pdf(idx));
            hist[idx] += 1;
        }

        for (weight, obs) in hist.into_iter().zip(observed.iter_mut()) {
            *obs = (*weight as f32 / sample_count as f32) * dist.weight_sum;
        }

        let mut chsq = 0.0;
        for (obs, exp) in observed.into_iter().zip(expected) {
            let diff = obs - exp;
            chsq += diff * diff / exp;
        }

        let pval = 1.0 - ChiSquared::new((expected.len() - 1) as f64).unwrap().cdf(chsq as f64);
        assert!(pval >= 0.99, "failed chi-squared statistical test, p = {}", pval);
    }

    #[test]
    fn basic1d() {
        test_distribution(&[1.0, 1.0, 2.0, 4.0, 8.0], 1000);
    }

    #[test]
    fn uniform1d() {
        test_distribution(&[1.0; 10_000], 1_000_000);
    }

    #[test]
    fn increasing1d() {
        let mut distr = [0.0; 100];
        for (i, weight) in distr.iter_mut().enumerate() {
            *weight = (5 * (i + 1)) as f32;
        }
        test_distribution(&distr, 100_000);
    }
}


