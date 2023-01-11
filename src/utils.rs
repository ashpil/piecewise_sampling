use num_traits::{
    Zero,
    One,
    NumOps,
    real::Real,
};

// a little bit of a workaround to allow SIMD summation for all types
// SIMD summation for floats is actually usually significantly more accurate
// than normal summation
pub trait Sum {
    fn sum(input: impl IntoIterator<Item=Self>) -> Self;
}

impl Sum for f32 {
    fn sum(input: impl IntoIterator<Item=Self>) -> Self {
        let mut sum = 0.0;
        for v in input {
            sum = unsafe { core::intrinsics::fadd_fast(sum, v) };
        }
        sum
    }
}

impl Sum for f64 {
    fn sum(input: impl IntoIterator<Item=Self>) -> Self {
        let mut sum = 0.0;
        for v in input {
            sum = unsafe { core::intrinsics::fadd_fast(sum, v) };
        }
        sum
    }
}

impl<F: core::ops::Add + Zero> Sum for F {
    default fn sum(input: impl IntoIterator<Item=F>) -> F {
        let mut sum = F::zero();
        for v in input {
            sum = sum + v;
        }
        sum
    }
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

pub fn lerp<T: One + NumOps + Copy>(by: T, from: T, to: T) -> T {
    (T::one() - by) * from + by * to 
}

