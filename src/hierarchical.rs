// Adapted from Matt Pharr's implementation

use crate::distribution::{
    Continuous1D,
    Continuous2D,
    Discrete1D,
    Discrete1DPdf,
    Discrete2D,
    Discrete2DPdf,
};
use crate::data2d::Data2D;
use crate::utils::lerp;
use num_traits::{
    Num,
    real::Real,
    Zero,
    AsPrimitive,
};

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    vec,
};

// returns pdf, selected idx
// remaps u to [0-1) range
fn select_remap<N: Num + PartialOrd + AsPrimitive<R>, R: Real + 'static>(weights: [N; 2], rand: &mut R) -> bool {
    let weight_sum = weights[0] + weights[1];
    let weight_1_r = weights[0].as_();
    let weight_2_r = weights[1].as_();
    let weight_sum_r = weight_sum.as_();
    let new_rand: R = *rand * weight_sum_r;
    if new_rand < weight_1_r {
        *rand = new_rand / weight_1_r;
        false
    } else {
        *rand = (new_rand - weight_1_r) / weight_2_r;
        true
    }
}

fn get_or_zero<Z: Zero + Copy>(v: &[Z], idx: usize) -> Z {
    v.get(idx).copied().unwrap_or(Z::zero())
}

fn get_or_zero_2d<Z: Zero + Copy>(v: &Data2D<Z>, idx: [usize; 2]) -> Z {
    v.get(idx).copied().unwrap_or(Z::zero())
}

#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize))]
pub struct Hierarchical1D<W> {
    levels: Box<[Box<[W]>]>,
}

impl<W: Num + PartialOrd + AsPrimitive<R>, R: Real + 'static> Discrete1D<R> for Hierarchical1D<W> {
    type Weight = W;

    fn build(weights: &[W]) -> Self {
        let level_count = weights.len().next_power_of_two().ilog2() as usize;
        let mut levels = vec![Default::default(); level_count].into_boxed_slice();

        levels[level_count - 1] = weights.to_vec().into_boxed_slice();

        let mut n = weights.len();
        let mut prev_level_idx = level_count;
        while n > 2 {
            prev_level_idx -= 1;
            n = n.div_ceil(2);
            let mut level = vec![W::zero(); n].into_boxed_slice();
            for (i, l) in level.iter_mut().enumerate() {
                *l =
                    get_or_zero(&levels[prev_level_idx], 2 * i + 0) +
                    get_or_zero(&levels[prev_level_idx], 2 * i + 1);
            }
            levels[prev_level_idx - 1] = level;
        }

        Self {
            levels,
        }
    }

    fn sample(&self, mut u: R) -> usize {
        let mut idx = 0;

        for level in self.levels.iter() {
            idx *= 2;
            let weights = [
                get_or_zero(level, idx + 0),
                get_or_zero(level, idx + 1),
            ];
            idx = idx + select_remap(weights, &mut u) as usize;
        }
        idx
    }

    fn integral(&self) -> W {
        let first = self.levels.first().unwrap();
        first[0] + first[1]
    }

    fn size(&self) -> usize {
        self.levels.last().unwrap().len()
    }
}

impl<W: Num + PartialOrd + AsPrimitive<R>, R: Real + 'static> Discrete1DPdf<R> for Hierarchical1D<W> {
    fn pdf(&self, u: usize) -> W {
        self.levels.first().unwrap()[u]
    }
}

impl<W: Num + PartialOrd + AsPrimitive<R>, R: Real + 'static> Continuous1D<R> for Hierarchical1D<W> where usize: AsPrimitive<R>,
{
    fn sample_continuous(&self, mut u: R) -> R {
        let mut idx = 0;

        for level in self.levels.iter() {
            idx *= 2;
            let weights = [
                get_or_zero(level, idx + 0),
                get_or_zero(level, idx + 1),
            ];
            idx = idx + select_remap(weights, &mut u) as usize;
        }
        (idx.as_() + u) / self.size().as_()
    }

    fn invert_continuous(&self, u: R) -> R {
        let mut out = [R::zero(), R::one()];
        let mut bounds = [
            R::zero(),
            self.size().next_power_of_two().as_() / self.size().as_(),
        ];
        let mut idx = 0;

        for level in self.levels.iter() {
            idx *= 2;

            let bounds_mid = (bounds[0] + bounds[1]) / 2.as_();

            let more = u < bounds_mid;
            let weights = [get_or_zero(level, idx + 0).as_(), get_or_zero(level, idx + 1).as_()];
            out[more as usize] = lerp(weights[0] / (weights[0] + weights[1]), out[0], out[1]);
            bounds[more as usize] = bounds_mid;
            idx += (!more) as usize;
        }

        let delta = (u - bounds[0]) / (bounds[1] - bounds[0]);
        lerp(delta, out[0], out[1])
    }
}

#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize))]
pub struct Hierarchical2D<W> {
    levels: Box<[Data2D<W>]>,
}

impl<W: Num + PartialOrd + AsPrimitive<R>, R: Real + 'static> Discrete2D<R> for Hierarchical2D<W> {
    type Weight = W;

    fn build(weights: &Data2D<W>) -> Self {
        let max_size = weights.width().max(weights.height());
        let level_count = max_size.next_power_of_two().ilog2() as usize;
        let mut levels = vec![Default::default(); level_count].into_boxed_slice();

        levels[level_count - 1] = weights.clone();

        let mut nx = weights.width();
        let mut ny = weights.height();
        let mut prev_level_idx = level_count;
        while nx > 2 || ny > 2 {
            prev_level_idx -= 1;
            nx = nx.div_ceil(2);
            ny = ny.div_ceil(2);
            let mut level = Data2D::new_same(nx, ny, W::zero());
            for y in 0..ny {
                for x in 0..nx {
                    level[[x, y]] = 
                        get_or_zero_2d(&levels[prev_level_idx], [2 * x + 0, 2 * y + 0]) +
                        get_or_zero_2d(&levels[prev_level_idx], [2 * x + 1, 2 * y + 0]) +
                        get_or_zero_2d(&levels[prev_level_idx], [2 * x + 0, 2 * y + 1]) +
                        get_or_zero_2d(&levels[prev_level_idx], [2 * x + 1, 2 * y + 1]);
                }
            }
            levels[prev_level_idx - 1] = level;
        }

        Self {
            levels,
        }
    }

    fn sample(&self, [mut u, mut v]: [R; 2]) -> [usize; 2] {
        let mut idx = [0; 2];

        for (i, level) in self.levels.iter().enumerate() {
            if i > 0 && self.levels[i].width() > self.levels[i - 1].width() { idx[0] *= 2 }
            if i > 0 && self.levels[i].height() > self.levels[i - 1].height() { idx[1] *= 2 }

            let weights = [
                [get_or_zero_2d(level, [idx[0] + 0, idx[1] + 0]), get_or_zero_2d(level, [idx[0] + 0, idx[1] + 1])],
                [get_or_zero_2d(level, [idx[0] + 1, idx[1] + 0]), get_or_zero_2d(level, [idx[0] + 1, idx[1] + 1])],
            ];

            let weights_x = [weights[0][0] + weights[0][1], weights[1][0] + weights[1][1]];
            let selected_x = select_remap(weights_x, &mut u) as usize;

            let weights_y = [weights[selected_x][0], weights[selected_x][1]];
            let selected_y = select_remap(weights_y, &mut v) as usize;

            idx[0] += selected_x;
            idx[1] += selected_y;
        }
        idx
    }

    fn integral(&self) -> W {
        let mut sum = W::zero();
        for l in self.levels.first().unwrap().iter().flatten() {
            sum = sum + *l;
        }
        sum
    }

    fn width(&self) -> usize {
        self.levels.last().unwrap().width()
    }

    fn height(&self) -> usize {
        self.levels.last().unwrap().height()
    }
}

impl<W: Num + PartialOrd + AsPrimitive<R>, R: Real + 'static> Discrete2DPdf<R> for Hierarchical2D<W> {
    fn pdf(&self, [u, v]: [usize; 2]) -> W {
        self.levels.first().unwrap()[[u, v]]
    }
}

impl<W: Num + PartialOrd + AsPrimitive<R>, R: Real + 'static> Continuous2D<R> for Hierarchical2D<W>
    where usize: AsPrimitive<R>,
{
    fn sample_continuous(&self, [mut u, mut v]: [R; 2]) -> [R; 2] {
        let mut idx = [0; 2];

        for (i, level) in self.levels.iter().enumerate() {
            if i > 0 && self.levels[i].width() > self.levels[i - 1].width() { idx[0] *= 2 }
            if i > 0 && self.levels[i].height() > self.levels[i - 1].height() { idx[1] *= 2 }

            let weights = [
                [get_or_zero_2d(level, [idx[0] + 0, idx[1] + 0]), get_or_zero_2d(level, [idx[0] + 0, idx[1] + 1])],
                [get_or_zero_2d(level, [idx[0] + 1, idx[1] + 0]), get_or_zero_2d(level, [idx[0] + 1, idx[1] + 1])],
            ];

            let weights_x = [weights[0][0] + weights[0][1], weights[1][0] + weights[1][1]];
            let selected_x = select_remap(weights_x, &mut u) as usize;

            let weights_y = [weights[selected_x][0], weights[selected_x][1]];
            let selected_y = select_remap(weights_y, &mut v) as usize;

            idx[0] += selected_x;
            idx[1] += selected_y;
        }
        let idx_normalized = [
            (idx[0].as_() + u) / self.width().as_(),
            (idx[1].as_() + v) / self.height().as_(),
        ];
        idx_normalized
    }

    fn invert_continuous(&self, [u, v]: [R; 2]) -> [R; 2] {
        let mut out_u = [R::zero(), R::one()];
        let mut out_v = [R::zero(), R::one()];
        let mut bounds_u = [
            R::zero(),
            self.width().next_power_of_two().as_() / self.width().as_(),
        ];
        let mut bounds_v = [
            R::zero(),
            self.height().next_power_of_two().as_() / self.height().as_(),
        ];
        let mut idx = [0; 2];

        for (i, level) in self.levels.iter().enumerate() {
            if i > 0 && self.levels[i].width() > self.levels[i - 1].width() { idx[0] *= 2 }
            if i > 0 && self.levels[i].height() > self.levels[i - 1].height() { idx[1] *= 2 }

            if level.width() > 1 {
                let bounds_mid = (bounds_u[0] + bounds_u[1]) / 2.as_();

                let weights = [
                    (get_or_zero_2d(level, [idx[0] + 0, idx[1] + 0]) + get_or_zero_2d(level, [idx[0] + 0, idx[1] + 1])).as_(),
                    (get_or_zero_2d(level, [idx[0] + 1, idx[1] + 0]) + get_or_zero_2d(level, [idx[0] + 1, idx[1] + 1])).as_(),
                ];
                let more = u < bounds_mid;
                out_u[more as usize] = lerp(weights[0] / (weights[0] + weights[1]), out_u[0], out_u[1]);
                bounds_u[more as usize] = bounds_mid;
                idx[0] += (!more) as usize;
            }

            if level.height() > 1 {
                let bounds_mid = (bounds_v[0] + bounds_v[1]) / 2.as_();

                let more = v < bounds_mid;
                let weights = [
                    get_or_zero_2d(level, [idx[0] + 0, idx[1] + 0]).as_(),
                    get_or_zero_2d(level, [idx[0] + 0, idx[1] + 1]).as_(),
                ];
                out_v[more as usize] = lerp(weights[0] / (weights[0] + weights[1]), out_v[0], out_v[1]);
                bounds_v[more as usize] = bounds_mid;
                idx[1] += (!more) as usize;
            }
        }

        let delta_u = (u - bounds_u[0]) / (bounds_u[1] - bounds_u[0]);
        let delta_v = (v - bounds_v[0]) / (bounds_v[1] - bounds_v[0]);
        [lerp(delta_u, out_u[0], out_u[1]), lerp(delta_v, out_v[0], out_v[1])]
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::distribution_1d_tests;
    use crate::distribution::continuous_distribution_1d_tests;
    use crate::distribution::distribution_2d_tests;

    distribution_1d_tests!(crate::hierarchical::Hierarchical1D);
    continuous_distribution_1d_tests!(crate::hierarchical::Hierarchical1D);

    distribution_2d_tests!(crate::hierarchical::Hierarchical2D);
}

