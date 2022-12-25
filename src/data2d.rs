#[derive(Clone)]
pub struct Data2D<T> {
    buffer: Vec<T>,
    width: usize,
}

impl<T: Clone> Data2D<T> {
    pub fn new_same(width: usize, height: usize, same: T) -> Self {
        Self {
            buffer: vec![same; width * height],
            width,
        }
    }
}

impl<T> std::ops::Index<usize> for Data2D<T> {
    type Output = [T];

    fn index(&self, idx: usize) -> &Self::Output {
        &self.buffer[idx * self.width..(idx + 1) * self.width]
    }
}

impl<T> std::ops::IndexMut<usize> for Data2D<T> {
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

    pub fn iter(&self) -> std::slice::Chunks<T> {
        self.buffer.chunks(self.width)
    }
}

