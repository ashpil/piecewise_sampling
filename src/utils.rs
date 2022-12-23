use num_traits::Float;

pub fn kahan_sum<F: Float>(input: impl IntoIterator<Item=F>) -> F {
    let mut sum = F::zero();
    let mut err = F::zero();
    for v in input {
        let y = v - err;
        let t = sum + y;
        err = (t - sum) - y;
        sum = t;
    }
    sum
}

