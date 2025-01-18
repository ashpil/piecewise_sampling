#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    vec::Vec,
    vec,
};

#[derive(Clone)]
pub struct Data2D<T> {
    buffer: Box<[T]>,
    width: usize,
}

impl<T: core::fmt::Debug> core::fmt::Debug for Data2D<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> Default for Data2D<T> {
    fn default() -> Self {
        Self {
            buffer: Vec::new().into_boxed_slice(),
            width: 0,
        }
    }
}

impl<T: Clone> Data2D<T> {
    pub fn new_same(width: usize, height: usize, same: T) -> Self {
        Self {
            buffer: vec![same; width * height].into_boxed_slice(),
            width,
        }
    }
}

impl<T> core::ops::Index<[usize; 2]> for Data2D<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &Self::Output {
        &self.buffer[idx[1] * self.width + idx[0]]
    }
}

impl<T> core::ops::IndexMut<[usize; 2]> for Data2D<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        &mut self.buffer[idx[1] * self.width + idx[0]]
    }
}

impl<T> Data2D<T> {
    pub fn insert(&mut self, x: usize, y: usize, data: T) {
        self.buffer[y * self.width + x] = data;
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.buffer.len() / self.width
    }

    pub fn iter(&self) -> core::slice::ChunksExact<T> {
        self.buffer.chunks_exact(self.width)
    }

    pub fn iter_mut(&mut self) -> core::slice::ChunksExactMut<T> {
        self.buffer.chunks_exact_mut(self.width)
    }

    pub fn get(&self, idx: [usize; 2]) -> Option<&T> {
        self.buffer.get(idx[1] * self.width..(idx[1] + 1) * self.width).and_then(|s| s.get(idx[0]))
    }
}

