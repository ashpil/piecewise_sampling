// Adapted from Matt Pharr's implementation

use crate::distribution::{
    ContinuousDistribution1D,
    Distribution1D,
};
use num_traits::{
    real::Real,
    Num,
    NumCast,
    Zero,
    cast,
};
use crate::utils::lerp;

// returns pdf, selected idx
// remaps u to [0-1) range
fn select_remap<N: Num + NumCast + PartialOrd + Copy, R: Real>(weights: [N; 2], rand: &mut R) -> (R, u8) {
    let weight_sum = weights[0] + weights[1];
    let weight_1_r = cast::<N, R>(weights[0]).unwrap();
    let weight_2_r = cast::<N, R>(weights[1]).unwrap();
    let weight_sum_r = cast::<N, R>(weight_sum).unwrap();
    let new_rand: R = *rand * weight_sum_r;
    if new_rand < weight_1_r {
        *rand = new_rand / weight_1_r;
        (weight_1_r / weight_sum_r, 0)
    } else {
        *rand = (new_rand - weight_1_r) / weight_2_r;
        (weight_2_r / weight_sum_r, 1)
    }
}

fn get_or_zero<Z: Zero + Copy>(v: &[Z], idx: usize) -> Z {
    v.get(idx).copied().unwrap_or(Z::zero())
}

pub struct Hierarchical1D<R: Real> {
    levels: Box<[Box<[R]>]>,
}

impl<R: Real> Distribution1D for Hierarchical1D<R> {
    type Weight = R;

    fn build(weights: &[R]) -> Self {
        let level_count = weights.len().next_power_of_two().ilog2() as usize;
        let mut levels = vec![Default::default(); level_count].into_boxed_slice();

        levels[level_count - 1] = weights.to_vec().into_boxed_slice();

        let mut n = weights.len();
        let mut prev_level_idx = level_count;
        while n > 2 {
            prev_level_idx -= 1;
            n = n.div_ceil(2);
            let mut level = vec![R::zero(); n].into_boxed_slice();
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

    fn sample(&self, mut u: R) -> (R, usize) {
        let mut pdf = self.integral();
        let mut idx = 0;

        for level in self.levels.iter() {
            idx *= 2;
            let probs = [
                get_or_zero(level, idx + 0),
                get_or_zero(level, idx + 1),
            ];
            let (level_pdf, level_idx) = select_remap(probs, &mut u);
            idx = idx + level_idx as usize;
            pdf = pdf * level_pdf;
        }
        (pdf, idx)
    }

    fn pdf(&self, mut u: usize) -> R {
        let mut pdf = self.integral();

        for level in self.levels.iter().rev() {
            let v = get_or_zero(level, u);

            if v == R::zero() { return R::zero() }

            let ue = if u % 2 == 1 {
                u - 1 
            } else {
                u
            };
            pdf = pdf * v / (get_or_zero(level, ue) + get_or_zero(level, ue + 1));

            u /= 2;
        }

        pdf
    }

    fn integral(&self) -> R {
        let first = self.levels.first().unwrap();
        first[0] + first[1]
    }

    fn size(&self) -> usize {
        self.levels.last().unwrap().len()
    }
}

impl<R: Real> ContinuousDistribution1D for Hierarchical1D<R> {
    fn sample_continuous(&self, mut u: R) -> (R, R) {
        let mut pdf = self.integral();
        let mut idx = 0;

        for level in self.levels.iter() {
            idx *= 2;
            let probs = [
                get_or_zero(level, idx + 0),
                get_or_zero(level, idx + 1),
            ];
            let (level_pdf, level_idx) = select_remap(probs, &mut u);
            idx = idx + level_idx as usize;
            pdf = pdf * level_pdf;
        }
        (pdf, (cast::<usize, R>(idx).unwrap() + u) / cast(self.size()).unwrap())
    }

    fn inverse_continuous(&self, u: R) -> R {
        let mut out = [R::zero(), R::one()];
        let mut bounds = [
            R::zero(),
            cast::<usize, R>(self.size().next_power_of_two()).unwrap() / cast::<usize, R>(self.size()).unwrap()
        ];
        let mut idx = 0;

        for level in self.levels.iter() {
            idx *= 2;

            let probs = [get_or_zero(level, idx + 0), get_or_zero(level, idx + 1)];
            let bounds_mid = (bounds[0] + bounds[1]) / cast(2).unwrap();

            let more = u < bounds_mid;
            out[more as usize] = lerp(probs[0] / (probs[0] + probs[1]), out[0], out[1]);
            bounds[more as usize] = bounds_mid;
            idx += (!more) as usize;
        }

        let delta = (u - bounds[0]) / (bounds[1] - bounds[0]);
        lerp(delta, out[0], out[1])
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::distribution_1d_tests;
    use crate::distribution::continuous_distribution_1d_tests;

    distribution_1d_tests!(crate::hierarchical::Hierarchical1D);
    continuous_distribution_1d_tests!(crate::hierarchical::Hierarchical1D);
}

