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

// from pbrt
pub fn radical_inverse<R: Real>(base_index: usize, mut a: u64) -> R {
    let primes = [ 2, 3, 5, 7, 11 ];

    let base = primes[base_index];

    let inv_base = R::one() / num_traits::cast(base).unwrap();
    let mut inv_base_m = R::one();
    let mut reversed_digits = 0;

    let limit = !0 / base - base;
    while a != 0 && reversed_digits < limit {
        let next = a / base;
        let digit = a - next * base;
        reversed_digits = reversed_digits * base + digit;
        inv_base_m = inv_base_m * inv_base;
        a = next;
    }
    num_traits::cast::<u64, R>(reversed_digits).unwrap() * inv_base_m
}

pub fn u64_to_color(a: u64) -> [f32; 3] {
    [
        radical_inverse(0, a),
        radical_inverse(1, a),
        radical_inverse(2, a),
    ]
}

