use crate::distribution::{
    Discrete1D,
    Continuous1D,
};
use crate::utils;
use num_traits::{
    Num,
    real::Real,
    AsPrimitive,
};

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    vec::Vec,
    vec,
};

pub type Alias2D<R> = crate::Adapter2D<Alias1D<R>>;

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize))]
pub struct Entry<W> {
    select: W,
    alias: u32,
}

#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize))]
pub struct Alias1D<W> {
    pub weight_sum: W,
    pub entries: Box<[Entry<W>]>,
}

impl<W: Num + PartialOrd + AsPrimitive<R>, R: Real + AsPrimitive<usize> + 'static> Discrete1D<R> for Alias1D<W>
    where usize: AsPrimitive<R>,
          usize: AsPrimitive<W>,
{
    type Weight = W;

    // Vose O(n)
    fn build(weights: &[W]) -> Self {
        let n = weights.len();

        let is_f32 = core::any::TypeId::of::<f32>() == core::any::TypeId::of::<W>();

        if is_f32 {
            // due to the fact that we use f32s, multiplying a [0-1) f32 by about 2 million or so
            // will give us numbers rounded to nearest float, which might be the next integer over, not
            // the actual one
            // would be nice if rust supported other float rounding modes...
            assert!(n < 2_000_000, "Alias1D on f32s not reliable for distributions with more than 2,000,000 elements");
        }

        let mut entries = vec![Entry { select: W::zero(), alias: 0 }; n].into_boxed_slice();

        let mut small = Vec::new();
        let mut large = Vec::new();

        let weight_sum = if is_f32 {
            utils::kahan_sum(weights.iter().cloned())
        } else {
            let mut sum = W::zero();
            for weight in weights {
                sum = sum + *weight;
            }
            sum
        };

        for (i, weight) in weights.iter().enumerate() {
            let adjusted_weight = *weight * n.as_();
            entries[i].select = adjusted_weight;
            if adjusted_weight < weight_sum {
                small.push(i as u32);
            } else {
                large.push(i as u32);
            }
        }

        while !small.is_empty() && !large.is_empty() {
            let l = small.pop().unwrap();
            let g = large.pop().unwrap();

            entries[l as usize].alias = g;
            entries[g as usize].select = (entries[g as usize].select + entries[l as usize].select) - weight_sum;

            if entries[g as usize].select < weight_sum {
                small.push(g);
            } else {
                large.push(g);
            }
        }

        // the select for entries in `large` should already all be >= sum, so 
        // we don't need to update them here
        // while let Some(g) = large.pop() {
        //     entries[g as usize].select = weight_sum;
        // }

        // these are actually large but are in small due to float error
        // they are currently slightly less than the weight sum, we need to make sure they're the weight sum
        while let Some(l) = small.pop() {
            entries[l as usize].select = weight_sum;
        }

        Self {
            weight_sum,
            entries,
        }
    }

    fn sample(&self, u: R) -> usize {
        let scaled: R = <usize as AsPrimitive<R>>::as_(self.entries.len()) * u;
        let mut index: usize = scaled.as_();
        let entry = self.entries[index];
        let v = (scaled - index.as_()) * self.weight_sum.as_();
        if entry.select.as_() <= v {
            index = entry.alias as usize;
        }

        index
    }

    fn integral(&self) -> W {
        self.weight_sum
    }

    fn size(&self) -> usize {
        self.entries.len()
    }
}

pub type ContinuousAlias2D<W> = crate::Adapter2D<ContinuousAlias1D<W>>;

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize))]
pub struct ContinuousEntry<W: Real> {
    select: W,
    alias: u32,
    own_region: [W; 2], // which region of own entry do we sample
    alias_region: [W; 2], // which region of alias entry do we sample
}

#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize))]
pub struct ContinuousAlias1D<W: Real> {
    pub weight_sum: W,
    pub entries: Box<[ContinuousEntry<W>]>,
}

impl<W: Real + AsPrimitive<usize>> Discrete1D<W> for ContinuousAlias1D<W>
    where usize: AsPrimitive<W>,
{
    type Weight = W;

    fn build(weights: &[W]) -> Self {
        let n = weights.len();

        // due to the fact that we use f32s, multiplying a [0-1) f32 by about 2 million or so
        // will give us numbers rounded to nearest float, which might be the next integer over, not
        // the actual one
        // would be nice if rust supported other float rounding modes...
        // TODO: disable if not f32
        assert!(n < 2_000_000, "Alias1D not reliable for distributions with more than 2,000,000 elements");

        let mut entries = vec![ContinuousEntry { select: W::zero(), alias: 0, own_region: [W::zero(), W::one()], alias_region: [W::zero(), W::zero()] }; n].into_boxed_slice();
        let mut adjusted_weights = vec![W::zero(); n].into_boxed_slice();

        let mut small = Vec::new();
        let mut large = Vec::new();

        // this may not be necessary if not f32, TODO: conditionally disable
        let weight_sum = utils::kahan_sum(weights.iter().cloned());

        for (i, weight) in weights.iter().enumerate() {
            let adjusted_weight = (*weight * n.as_()) / weight_sum;
            adjusted_weights[i] = adjusted_weight;
            entries[i].select = adjusted_weight;
            if adjusted_weight < W::one() {
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
            region[0] = region[1] - (W::one() - entries[less as usize].select) / adjusted_weights[more as usize];
            entries[less as usize].alias_region = region;
            entries[more as usize].own_region[1] = region[0];

            // more numerically stable version of this
            // entries[more as usize].select -= (W::one() - entries[less as usize].select);
            entries[more as usize].select = (entries[more as usize].select + entries[less as usize].select) - W::one();

            if entries[more as usize].select < W::one() {
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
            entries[g as usize].select = W::one();
        }

        // these are actually large but are in small due to float error
        // should be slightly less than 1.0, we need to make sure they're 1.0
        while let Some(l) = small.pop() {
            entries[l as usize].select = W::one();
        }

        Self {
            weight_sum,
            entries,
        }
    }

    fn sample(&self, u: W) -> usize {
        let scaled: W = self.entries.len().as_() * u;
        let mut index: usize = scaled.as_();
        let entry = self.entries[index];
        let v = scaled - index.as_();
        if entry.select <= v {
            index = entry.alias as usize;
        }

        index
    }

    fn integral(&self) -> W {
        self.weight_sum
    }

    fn size(&self) -> usize {
        self.entries.len()
    }
}

impl<W: Real + AsPrimitive<usize>> Continuous1D<W> for ContinuousAlias1D<W>
    where usize: AsPrimitive<W>,
{
    fn sample_continuous(&self, u: W) -> W {
        let scaled: W = self.entries.len().as_() * u;
        let initial_index: usize = scaled.as_();
        let initial_entry = self.entries[initial_index];
        let v = scaled - initial_index.as_();

        let (index, du) = if initial_entry.select <= v {
            // selected alias of initial entry
            let v_remapped = (v - initial_entry.select) / (W::one() - initial_entry.select);

            let du = v_remapped * (initial_entry.alias_region[1] - initial_entry.alias_region[0]) + initial_entry.alias_region[0];
            let index = initial_entry.alias as usize;
            (index, du)
        } else {
            // selected initial entry
            let v_remapped = v / initial_entry.select;

            let index = initial_index;
            let du = v_remapped * (initial_entry.own_region[1] - initial_entry.own_region[0]) + initial_entry.own_region[0];

            (index, du)
        };


        (index.as_() + du) / self.size().as_()
    }

    // O(n) at worst
    fn inverse_continuous(&self, u: W) -> W {
        let scaled: W = self.entries.len().as_() * u;
        let initial_index: usize = scaled.as_();
        let initial_entry = self.entries[initial_index];
        let v = scaled - initial_index.as_();

        let (index, du) = if initial_entry.own_region[0] == W::zero() && initial_entry.own_region[1] == W::one() {
            (initial_index, v * initial_entry.select)
        } else {
            if v <= initial_entry.own_region[1] {
                (initial_index, (v / initial_entry.own_region[1]) * initial_entry.select)
            } else {
                let mut index = None;
                let mut du = None;
                for (entry_idx, entry) in self.entries.iter().enumerate() {
                    if entry.alias == initial_index as u32 {
                        if entry.alias_region[0] <= v && v <= entry.alias_region[1] {
                            let v_remapped = (v - entry.alias_region[0]) / (entry.alias_region[1] - entry.alias_region[0]);
                            index = Some(entry_idx);
                            du = Some(v_remapped * (W::one() - entry.select) + entry.select);
                        }
                    }
                }
                (index.unwrap(), du.unwrap())
            }
        };

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

