'''
VGA
res: 672x376
Center of table: 414, 205 (row, col) -> 53, 0 (x, y)
Pixels per centimeter: 2.3
Pixel window: 40
'''
import numpy as np


cloth_grid_bounds = [[129, 672], [68, 355]]

cloth_grid_window = 20

cloth_grid_input_dim = 10

# Ordered (row, col)
cloth_grid_coordinates = [
    ((338, 150), 4),
    ((338, 200), 4),
    ((338, 250), 4),
    ((278, 150), 4),
    ((278, 200), 4),
    ((278, 250), 1),
    ((278, 300), 1),
    ((278, 650), 1),
    ((228, 200), 1),
    ((228, 250), 1),
    ((228, 300), 1),
    ((228, 350), 1),
    ((228, 500), 1),
    ((228, 550), 1),
    ((228, 650), 1),
    ((178, 250), 1),
    ((178, 300), 1),
    ((178, 350), 1),
    ((178, 400), 1),
    ((178, 450), 1),
    ((178, 500), 1),
    ((178, 550), 1),
    ((178, 600), 1),
    ((178, 650), 1),
    ((128, 300), 1),
    ((128, 350), 1),
    ((128, 400), 1),
    ((128, 450), 1),
    ((128, 500), 1),
    ((128, 550), 1),
    ((128, 600), 1),
    ((128, 650), 1),
]

cloth_grid_ref = np.array([[205, 414], [53, 0]])

pixels_per_cm = 2.3

cloth_net_mean = [117.8504316, 122.86169381, 84.21057003]
cloth_net_std = [49.69568644, 58.26177112, 75.96082055]

basket_net_mean = 0.16303136
basket_net_std = 0.41344904

basket_net_ul = 1.35
basket_net_ll = 1

# Used to match depth images to simulated images
basket_net_bounds = []

bakset_im_dims = [144, 256]
