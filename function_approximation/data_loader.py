import numpy as np


class DataLoader:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_points = args.num_points
        self.input, self.target = self._generate_data()
        self.pointer = 0

    def next_batch(self):
        batch_x = self.input[self.pointer: self.pointer + self.batch_size]
        batch_y = self.target[self.pointer: self.pointer + self.batch_size]
        self.pointer = self.pointer + self.batch_size
        return batch_x, batch_y

    def reset_pointer(self):
        self.pointer = 0

    def _generate_data(self):
        x = np.linspace(-1, 1, self.num_points)
        y = np.sin(x)  # np.random.randn(self.num_points) +
        return x, y
