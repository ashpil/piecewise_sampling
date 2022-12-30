use crate::data2d::Data2D;

// 1D piecewise constant distribution
pub trait Distribution1D {
    // constructor
    fn build(weights: &[f32]) -> Self;

    // takes in rand [0-1), returns (pdf, selected idx)
    fn sample_discrete(&self, u: f32) -> (f32, usize);

    // takes in coord, returns pdf
    fn pdf(&self, u: usize) -> f32;

    // sum of all weights
    fn integral(&self) -> f32;

    // range of sampled idxs, should be len of weights
    fn size(&self) -> usize;
}

// 2D piecewise constant distribution
pub trait Distribution2D {
    // constructor
    fn build(weights: &Data2D<f32>) -> Self;

    // takes in rand [0-1)x[0-1), returns (pdf, uv coords)
    fn sample_discrete(&self, uv: [f32; 2]) -> (f32, [usize; 2]);

    // takes in coords, returns pdf
    fn pdf(&self, uv: [usize; 2]) -> f32;

    fn width(&self) -> usize;
    fn height(&self) -> usize;

    // fills demo image with sample_count samples
    fn fill_demo_image(&self, demo: &mut Data2D<[f32; 3]>, rngs: impl Iterator<Item = [f32; 2]>) {
        for rng in rngs {
            let (_, [x, y]) = self.sample_discrete(rng);
            for is in -1..1 {
                for js in -1..1 {
                    let j = (y as i32 + js).clamp(0, (demo.height() - 1) as i32) as usize;
                    let i = (x as i32 + is).clamp(0, (demo.width() - 1) as i32) as usize;
                    demo[j][i] = [1000.0, 0.0, 0.0];
                }
            }
        }
    }
}

