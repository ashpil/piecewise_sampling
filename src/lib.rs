#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

#![feature(sort_floats)]
#![feature(int_roundings)]

#[cfg(not(feature = "std"))]
extern crate alloc;

mod utils;

pub mod distribution;

mod data2d;
pub use data2d::Data2D;

mod inversion;
pub use inversion::Inversion1D;
pub use inversion::Inversion2D;

mod alias;
pub use alias::Alias1D;
pub use alias::Alias2D;
pub use alias::ContinuousAlias1D;
pub use alias::ContinuousAlias2D;

mod adapter2d;
pub use adapter2d::Adapter2D;

mod hierarchical;
pub use hierarchical::Hierarchical1D;
pub use hierarchical::Hierarchical2D;
