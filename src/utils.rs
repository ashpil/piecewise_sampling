use num_traits::real::Real;

// unlikely to be necessary if not float, but why not,
// let it be used on arbitrary reals
pub fn kahan_sum<R: Real>(input: impl IntoIterator<Item=R>) -> R {
    let mut sum = R::zero();
    let mut err = R::zero();
    for v in input {
        let y = v - err;
        let t = sum + y;
        err = (t - sum) - y;
        sum = t;
    }
    sum
}

