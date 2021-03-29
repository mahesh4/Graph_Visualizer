import numpy as np


class InstanceObject:
    def __init__(self, value):
        self.value = np.array(value)
        self.skyline = False
        self.virtual = False
        self.probability = None