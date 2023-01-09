// Adapted from Matt Pharr's implementation

use crate::distribution::{
    Continuous1D,
    Continuous2D,
    Discrete1D,
    Discrete2D,
};
use crate::data2d::Data2D;
use crate::utils::lerp;
use num_traits::{
    real::Real,
    Num,
    NumCast,
    Zero,
    cast,
};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;


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

fn get_or_zero_2d<Z: Zero + Copy>(v: &Data2D<Z>, x: usize, y: usize) -> Z {
    v.get(y).and_then(|s| s.get(x)).copied().unwrap_or(Z::zero())
}

#[derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
pub struct Hierarchical1D<R: Real> {
    levels: Box<[Box<[R]>]>,
}

impl<R: Real> Discrete1D for Hierarchical1D<R> {
    type Weight = R;

    fn build(weights: &[R]) -> Self {
        let level_count = weights.len().next_power_of_two().ilog2() as usize;
        let mut levels = alloc::vec![Default::default(); level_count].into_boxed_slice();

        levels[level_count - 1] = weights.to_vec().into_boxed_slice();

        let mut n = weights.len();
        let mut prev_level_idx = level_count;
        while n > 2 {
            prev_level_idx -= 1;
            n = n.div_ceil(2);
            let mut level = alloc::vec![R::zero(); n].into_boxed_slice();
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

impl<R: Real> Continuous1D for Hierarchical1D<R> {
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

#[derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
pub struct Hierarchical2D<R: Real> {
    levels: Box<[Data2D<R>]>,
}

impl<R: Real> Hierarchical2D<R> {
    fn integral(&self) -> R {
        let mut sum = R::zero();
        for l in self.levels.first().unwrap().iter().flatten() {
            sum = sum + *l;
        }
        sum
    }
}

impl<R: Real> Discrete2D for Hierarchical2D<R> {
    type Weight = R;

    fn build(weights: &Data2D<R>) -> Self {
        let max_size = weights.width().max(weights.height());
        let level_count = max_size.next_power_of_two().ilog2() as usize;
        let mut levels = alloc::vec![Default::default(); level_count].into_boxed_slice();

        levels[level_count - 1] = weights.clone();

        let mut nx = weights.width();
        let mut ny = weights.height();
        let mut prev_level_idx = level_count;
        while nx > 2 || ny > 2 {
            prev_level_idx -= 1;
            nx = nx.div_ceil(2);
            ny = ny.div_ceil(2);
            let mut level = Data2D::new_same(nx, ny, R::zero());
            for y in 0..ny {
                for x in 0..nx {
                    level[y][x] = 
                        get_or_zero_2d(&levels[prev_level_idx], 2 * x + 0, 2 * y + 0) +
                        get_or_zero_2d(&levels[prev_level_idx], 2 * x + 1, 2 * y + 0) +
                        get_or_zero_2d(&levels[prev_level_idx], 2 * x + 0, 2 * y + 1) +
                        get_or_zero_2d(&levels[prev_level_idx], 2 * x + 1, 2 * y + 1);
                }
            }
            levels[prev_level_idx - 1] = level;
        }

        Self {
            levels,
        }
    }

    fn sample(&self, [mut u, mut v]: [R; 2]) -> (R, [usize; 2]) {
        let mut pdf = self.integral();
        let mut idx = [0; 2];

        for (i, level) in self.levels.iter().enumerate() {
            if i > 0 && self.levels[i].width() > self.levels[i - 1].width() { idx[0] *= 2 }
            if i > 0 && self.levels[i].height() > self.levels[i - 1].height() { idx[1] *= 2 }

            let probs_x = [
                get_or_zero_2d(level, idx[0] + 0, idx[1] + 0) + get_or_zero_2d(level, idx[0] + 0, idx[1] + 1),
                get_or_zero_2d(level, idx[0] + 1, idx[1] + 0) + get_or_zero_2d(level, idx[0] + 1, idx[1] + 1),
            ];
            let (pdf_x, idx_x) = select_remap(probs_x, &mut u);
            idx[0] = idx[0] + idx_x as usize;
            pdf = pdf * pdf_x;

            let probs_y = [
                get_or_zero_2d(level, idx[0] + 0, idx[1] + 0),
                get_or_zero_2d(level, idx[0] + 0, idx[1] + 1),
            ];
            let (pdf_y, idx_y) = select_remap(probs_y, &mut v);
            idx[1] = idx[1] + idx_y as usize;
            pdf = pdf * pdf_y;
        }
        (pdf, idx)
    }

    fn pdf(&self, [mut u, mut v]: [usize; 2]) -> R {
        let mut pdf = self.integral();

        for (i, level) in self.levels.iter().enumerate().rev() {
            let x = get_or_zero_2d(level, u, v);
            if x == R::zero() { return R::zero() }

            let ue = if u % 2 == 1 {
                u - 1 
            } else {
                u
            };
            let ve = if v % 2 == 1 {
                v - 1 
            } else {
                v
            };
            pdf = pdf * x / (
                get_or_zero_2d(level, ue + 0, ve + 0) +
                get_or_zero_2d(level, ue + 1, ve + 0) +
                get_or_zero_2d(level, ue + 0, ve + 1) +
                get_or_zero_2d(level, ue + 1, ve + 1)
            );

            if i > 0 && self.levels[i - 1].width() < self.levels[i].width() { u /= 2 }
            if i > 0 && self.levels[i - 1].height() < self.levels[i].height() { v /= 2 }
        }

        pdf
    }

    fn width(&self) -> usize {
        self.levels.last().unwrap().width()
    }

    fn height(&self) -> usize {
        self.levels.last().unwrap().height()
    }
}

impl<R: Real> Continuous2D for Hierarchical2D<R> {
    fn sample_continuous(&self, [mut u, mut v]: [R; 2]) -> (R, [R; 2]) {
        let mut pdf = self.integral();
        let mut idx = [0; 2];

        for (i, level) in self.levels.iter().enumerate() {
            if i > 0 && self.levels[i].width() > self.levels[i - 1].width() { idx[0] *= 2 }
            if i > 0 && self.levels[i].height() > self.levels[i - 1].height() { idx[1] *= 2 }

            let probs_x = [
                get_or_zero_2d(level, idx[0] + 0, idx[1] + 0) + get_or_zero_2d(level, idx[0] + 0, idx[1] + 1),
                get_or_zero_2d(level, idx[0] + 1, idx[1] + 0) + get_or_zero_2d(level, idx[0] + 1, idx[1] + 1),
            ];
            let (pdf_x, idx_x) = select_remap(probs_x, &mut u);
            idx[0] = idx[0] + idx_x as usize;
            pdf = pdf * pdf_x;

            let probs_y = [
                get_or_zero_2d(level, idx[0] + 0, idx[1] + 0),
                get_or_zero_2d(level, idx[0] + 0, idx[1] + 1),
            ];
            let (pdf_y, idx_y) = select_remap(probs_y, &mut v);
            idx[1] = idx[1] + idx_y as usize;
            pdf = pdf * pdf_y;
        }
        let idx_normalized = [
            (cast::<usize, R>(idx[0]).unwrap() + u) / cast(self.width()).unwrap(),
            (cast::<usize, R>(idx[1]).unwrap() + v) / cast(self.height()).unwrap(),
        ];
        (pdf, idx_normalized)
    }

    fn inverse_continuous(&self, [u, v]: [R; 2]) -> [R; 2] {
        let mut out_u = [R::zero(), R::one()];
        let mut out_v = [R::zero(), R::one()];
        let mut bounds_u = [
            R::zero(),
            cast::<usize, R>(self.width().next_power_of_two()).unwrap() / cast::<usize, R>(self.width()).unwrap()
        ];
        let mut bounds_v = [
            R::zero(),
            cast::<usize, R>(self.height().next_power_of_two()).unwrap() / cast::<usize, R>(self.height()).unwrap()
        ];
        let mut idx = [0; 2];

        for (i, level) in self.levels.iter().enumerate() {
            if i > 0 && self.levels[i].width() > self.levels[i - 1].width() { idx[0] *= 2 }
            if i > 0 && self.levels[i].height() > self.levels[i - 1].height() { idx[1] *= 2 }

            if level.width() > 1 {
                let probs = [
                    get_or_zero_2d(level, idx[0] + 0, idx[1] + 0) + get_or_zero_2d(level, idx[0] + 0, idx[1] + 1),
                    get_or_zero_2d(level, idx[0] + 1, idx[1] + 0) + get_or_zero_2d(level, idx[0] + 1, idx[1] + 1),
                ];
                let bounds_mid = (bounds_u[0] + bounds_u[1]) / cast(2).unwrap();

                let more = u < bounds_mid;
                out_u[more as usize] = lerp(probs[0] / (probs[0] + probs[1]), out_u[0], out_u[1]);
                bounds_u[more as usize] = bounds_mid;
                idx[0] += (!more) as usize;
            }

            if level.height() > 1 {
                let probs = [
                    get_or_zero_2d(level, idx[0] + 0, idx[1] + 0),
                    get_or_zero_2d(level, idx[0] + 0, idx[1] + 1),
                ];
                let bounds_mid = (bounds_v[0] + bounds_v[1]) / cast(2).unwrap();

                let more = v < bounds_mid;
                out_v[more as usize] = lerp(probs[0] / (probs[0] + probs[1]), out_v[0], out_v[1]);
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

    distribution_1d_tests!(crate::hierarchical::Hierarchical1D);
    continuous_distribution_1d_tests!(crate::hierarchical::Hierarchical1D);

    #[test]
    fn build_2d() {
        use crate::distribution::Distribution2D;
        let width = 17;
        let height = 16;
        let mut dist = crate::data2d::Data2D::new_same(width, height, 0.0);
        for j in 0..height {
            for i in 0..width {
                dist[j][i] = 2.0;
            }
        }
        crate::hierarchical::Hierarchical2D::build(&dist);
    }
}

