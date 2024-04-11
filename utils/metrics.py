import matplotlib.pyplot as plt
import numpy as np
from skimage import metrics

coords_a = np.array([[1, 1], [1, 0], [1, 0], [1, 1]])
coords_b = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
distance = metrics.hausdorff_distance(coords_a, coords_b)
print(distance)