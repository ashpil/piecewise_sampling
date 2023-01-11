#[cfg(not(feature = "std"))]
use alloc::{
    vec::Vec,
    vec,
};

#[derive(Clone)]
pub struct Data2D<T> {
    buffer: Vec<T>,
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
            buffer: Vec::new(),
            width: 0,
        }
    }
}

impl<T: Clone> Data2D<T> {
    pub fn new_same(width: usize, height: usize, same: T) -> Self {
        Self {
            buffer: vec![same; width * height],
            width,
        }
    }
}

impl<T> core::ops::Index<usize> for Data2D<T> {
    type Output = [T];

    fn index(&self, idx: usize) -> &Self::Output {
        &self.buffer[idx * self.width..(idx + 1) * self.width]
    }
}

impl<T> core::ops::IndexMut<usize> for Data2D<T> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.buffer[idx * self.width..(idx + 1) * self.width]
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

    pub fn iter(&self) -> core::slice::Chunks<T> {
        self.buffer.chunks(self.width)
    }

    pub fn get(&self, idx: usize) -> Option<&[T]> {
        self.buffer.get(idx * self.width..(idx + 1) * self.width)
    }
}

