use crate::distribution::{
    Distribution1D,
    ContinuousDistribution1D,
};
use crate::utils;
use num_traits::real::Real;
use num_traits::AsPrimitive;

#[derive(Clone, Copy, Debug)]
pub struct Entry<R: Real> {
    pdf: R,
    select: R,
    alias: u32,
}

pub struct Alias1D<R: Real> {
    pub weight_sum: R,
    pub entries: Box<[Entry<R>]>,
}

impl<R: Real + AsPrimitive<usize> + 'static> Distribution1D for Alias1D<R>
    where usize: AsPrimitive<R>,
{
    type Weight = R;

    // Vose O(n)
    fn build(weights: &[R]) -> Self {
        let n = weights.len();

        // due to the fact that we use f32s, multiplying a [0-1) f32 by about 2 million or so
        // will give us numbers rounded to nearest float, which might be the next integer over, not
        // the actual one
        // would be nice if rust supported other float rounding modes...
        // TODO: disable if not f32
        assert!(n < 2_000_000, "Alias1D not reliable for distributions with more than 2,000,000 elements");

        let mut entries = vec![Entry { pdf: R::zero(), select: R::zero(), alias: 0 }; n].into_boxed_slice();

        let mut small = Vec::new();
        let mut large = Vec::new();

        // this may not be necessary if not f32, TODO: conditionally disable
        let weight_sum = utils::kahan_sum(weights.iter().cloned());

        for (i, weight) in weights.iter().enumerate() {
            let adjusted_weight = (*weight * n.as_()) / weight_sum;
            entries[i].pdf = *weight;
            entries[i].select = adjusted_weight;
            if adjusted_weight < R::one() {
                small.push(i as u32);
            } else {
                large.push(i as u32);
            }
        }

        while !small.is_empty() && !large.is_empty() {
            let l = small.pop().unwrap();
            let g = large.pop().unwrap();

            entries[l as usize].alias = g;
            entries[g as usize].select = (entries[g as usize].select + entries[l as usize].select) - R::one();

            if entries[g as usize].select < R::one() {
                small.push(g);
            } else {
                large.push(g);
            }
        }

        // the select for entries in `large` should already all be >= 1.0, so 
        // we don't need to update them here
        //while let Some(g) = large.pop() {
        //    entries[g as usize].select = R::one();
        //}

        // these are actually large but are in small due to float error
        // they are currently slightly less than 1.0, we need to make sure they're 1.0
        while let Some(l) = small.pop() {
            entries[l as usize].select = R::one();
        }

        Self {
            weight_sum,
            entries,
        }
    }

    fn sample(&self, u: R) -> (R, usize) {
        let scaled: R = self.entries.len().as_() * u;
        let mut index: usize = scaled.as_();
        let mut entry = self.entries[index];
        let v = scaled - index.as_();
        if entry.select <= v {
            index = entry.alias as usize;
            entry = self.entries[entry.alias as usize];
        }

        (entry.pdf, index)
    }

    fn pdf(&self, u: usize) -> R {
        self.entries[u].pdf
    }

    fn integral(&self) -> R {
        self.weight_sum
    }

    fn size(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ContinuousEntry<R: Real> {
    pdf: R,
    select: R,
    alias: u32,
    own_region: [R; 2], // which region of own entry do we sample
    alias_region: [R; 2], // which region of alias entry do we sample
}

pub struct ContinuousAlias1D<R: Real> {
    pub weight_sum: R,
    pub entries: Box<[ContinuousEntry<R>]>,
}

impl<R: Real + AsPrimitive<usize> + 'static> Distribution1D for ContinuousAlias1D<R>
    where usize: AsPrimitive<R>,
{
    type Weight = R;

    fn build(weights: &[R]) -> Self {
        let n = weights.len();

        // due to the fact that we use f32s, multiplying a [0-1) f32 by about 2 million or so
        // will give us numbers rounded to nearest float, which might be the next integer over, not
        // the actual one
        // would be nice if rust supported other float rounding modes...
        // TODO: disable if not f32
        assert!(n < 2_000_000, "Alias1D not reliable for distributions with more than 2,000,000 elements");

        let mut entries = vec![ContinuousEntry { pdf: R::zero(), select: R::zero(), alias: 0, own_region: [R::zero(), R::one()], alias_region: [R::zero(), R::zero()] }; n].into_boxed_slice();
        let mut adjusted_weights = vec![R::zero(); n].into_boxed_slice();

        let mut small = Vec::new();
        let mut large = Vec::new();

        // this may not be necessary if not f32, TODO: conditionally disable
        let weight_sum = utils::kahan_sum(weights.iter().cloned());

        for (i, weight) in weights.iter().enumerate() {
            let adjusted_weight = (*weight * n.as_()) / weight_sum;
            adjusted_weights[i] = adjusted_weight;
            entries[i].pdf = *weight;
            entries[i].select = adjusted_weight;
            if adjusted_weight < R::one() {
                small.push(i as u32);
            } else {
                large.push(i as u32);
            }
        }

        // always create a < 1.0 entry here
        while !small.is_empty() && !large.is_empty() {
            let less = small.pop().unwrap();
            let more = large.pop().unwrap();

            entries[less as usize].alias = more;
            let mut region = entries[more as usize].own_region;
            region[0] = region[1] - (R::one() - entries[less as usize].select) / adjusted_weights[more as usize];
            entries[less as usize].alias_region = region;
            entries[more as usize].own_region[1] = region[0];

            // more numerically stable version of this
            // entries[more as usize].select -= (R::one() - entries[less as usize].select);
            entries[more as usize].select = (entries[more as usize].select + entries[less as usize].select) - R::one();

            if entries[more as usize].select < R::one() {
                small.push(more);
            } else {
                large.push(more);
            }
        }

        // the select for entries in `large` should already all be >= 1.0, so 
        // technically we don't need to update them here,
        // but we use division by select in our continuous sampling methods, so we
        // still need to make sure it's exactly one
        while let Some(g) = large.pop() {
            entries[g as usize].select = R::one();
        }

        // these are actually large but are in small due to float error
        // should be slightly less than 1.0, we need to make sure they're 1.0
        while let Some(l) = small.pop() {
            entries[l as usize].select = R::one();
        }

        Self {
            weight_sum,
            entries,
        }
    }

    fn sample(&self, u: R) -> (R, usize) {
        let scaled: R = self.entries.len().as_() * u;
        let mut index: usize = scaled.as_();
        let mut entry = self.entries[index];
        let v = scaled - index.as_();
        if entry.select <= v {
            index = entry.alias as usize;
            entry = self.entries[entry.alias as usize];
        }

        (entry.pdf, index)
    }

    fn pdf(&self, u: usize) -> R {
        self.entries[u].pdf
    }

    fn integral(&self) -> R {
        self.weight_sum
    }

    fn size(&self) -> usize {
        self.entries.len()
    }
}

impl<R: Real + AsPrimitive<usize> + 'static> ContinuousDistribution1D for ContinuousAlias1D<R> where usize: AsPrimitive<R> {
    fn sample_continuous(&self, u: R) -> (R, R) {
        let scaled: R = self.entries.len().as_() * u;
        let initial_index: usize = scaled.as_();
        let initial_entry = self.entries[initial_index];
        let v = scaled - initial_index.as_();

        let (pdf, index, du) = if initial_entry.select <= v {
            // selected alias of initial entry
            let v_remapped = (v - initial_entry.select) / (R::one() - initial_entry.select);

            let pdf = self.entries[initial_entry.alias as usize].pdf;

            let du = v_remapped * (initial_entry.alias_region[1] - initial_entry.alias_region[0]) + initial_entry.alias_region[0];
            let index = initial_entry.alias as usize;
            (pdf, index, du)
        } else {
            // selected initial entry
            let v_remapped = v / initial_entry.select;

            let pdf = initial_entry.pdf;

            let index = initial_index;
            let du = v_remapped * (initial_entry.own_region[1] - initial_entry.own_region[0]) + initial_entry.own_region[0];

            (pdf, index, du)
        };


        (pdf, (index.as_() + du) / self.size().as_())
    }

    fn inverse_continuous(&self, u: R) -> R {
        let scaled: R = self.entries.len().as_() * u;
        let index: usize = scaled.as_();
        let v = scaled - index.as_();
        let entry = self.entries[index];

        let du;
        if entry.select == R::one() {
            if entry.own_region[1] == R::one() {
                du = v;
            } else {
                todo!()
            }
        } else {
            du = v * entry.select;
        }

        (index.as_() + du) / self.size().as_()
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::distribution_1d_tests;
    use crate::distribution::continuous_distribution_1d_tests;

    distribution_1d_tests!(crate::alias::Alias1D);
    continuous_distribution_1d_tests!(crate::alias::ContinuousAlias1D);
}

