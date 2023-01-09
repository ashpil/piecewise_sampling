#![cfg_attr(not(test), no_std)]

#![feature(sort_floats)]
#![feature(int_roundings)]

#[cfg(not(feature = "std"))]
extern crate alloc;

mod utils;

pub mod data2d;
pub mod distribution;
pub mod inversion;
pub mod alias;
pub mod adapter2d;
pub mod hierarchical;
