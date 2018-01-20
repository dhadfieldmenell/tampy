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
    # ((338, 150), 1),
    # ((338, 200), 1),
    # ((338, 250), 1),
    # ((278, 150), 1),
    # ((278, 200), 1),
    # ((278, 250), 1),
    # ((278, 300), 1),
    # ((278, 650), 2),
    # ((228, 200), 1),
    # ((228, 250), 1),
    # ((228, 300), 1),
    # ((228, 350), 3),
    # ((228, 500), 3),
    # ((228, 550), 3),
    # ((228, 650), 2),
    # ((178, 250), 1),
    # ((178, 300), 1),
    # ((178, 350), 3),
    # ((178, 400), 3),
    # ((178, 450), 3),
    # ((178, 500), 3),
    # ((178, 550), 3),
    # ((178, 600), 3),
    # ((178, 650), 2),
    # ((128, 300), 1),
    # ((128, 350), 3),
    # ((128, 400), 3),
    # ((128, 450), 3),
    # ((128, 500), 3),
    # ((128, 550), 3),
    # ((128, 600), 2),
    # ((128, 650), 2),
    ((210, 285), 2),
    ((210, 340), 2),
    ((210, 395), 2),
    ((210, 450), 2),
    ((210, 505), 3),
    # ((210, 560), 3),
    ((155, 285), 2),
    ((155, 340), 2),
    ((155, 395), 2),
    ((155, 450), 2),
    ((155, 505), 3),
    # ((155, 560), 3),
    ((100, 285), 2),
    ((100, 340), 2),
    ((100, 395), 2),
    ((100, 450), 3),
    ((100, 505), 3),
    # ((100, 560), 3)

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

basket_im_dims = [144, 256]

basket_near_pos = [0.65, 0.3,  0.875]
basket_far_pos = [0.5, -0.6,  0.875]
basket_near_rot = [2*np.pi/3, 0, np.pi/2]
basket_far_rot = [np.pi/4, 0, np.pi/2]

cloth_grid_spacing = 0.15 # From center to center is 0.2 + 0.15 + 0.2