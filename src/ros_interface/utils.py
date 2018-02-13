'''
VGA
res: 672x376
Center of table: 200, 350 (row, col) -> 52, 5 (x, y)
Pixels per centimeter: 2.3
Pixel window: 40
'''
import numpy as np


cloth_grid_bounds = [[129, 672], [68, 355]]

cloth_grid_window = 20

cloth_grid_input_dim = 15

# Ordered (row, col)
# Region 5 is the basket near pos, 6 is the basket far pos
cloth_grid_coordinates = [
    ((230, 200), 2),
    ((160, 200), 2),
    ((180, 260), 5),
    ((130, 270), 2),
    ((210, 340), 2),
    ((150, 330), 5),
    ((180, 410), 3),
    ((130, 360), 3),
    ((150, 510), 6),
    ((195, 550), 6),
    ((250, 160), 2),
    ((250, 100), 5),
]

washer_im_locs = [((360, 290), (0.14, 1.19, 0.71)), # -> CENTER FRONT
                  ((230, 300), (0.14, 1.19, 0.71)), # -> LEFT FRONT
                  ((550, 300), (0.14, 1.19, 0.71)), # -> RIGHT FRONT
                  ((320, 370), (0.14, 1.19, 0.71)), # -> CENTER REAR
                  ((275, 275), (0.14, 1.19, 0.71)), # -> LEFT REAR
                  ((275, 475), (0.14, 1.19, 0.71))] # -> RIGHT REAR

cloth_grid_ref = np.array([[200, 350], [55, 5]])

pixels_per_cm = 2.3

# cloth_net_mean = [117.8504316, 122.86169381, 84.21057003]
# cloth_net_std = [49.69568644, 58.26177112, 75.96082055]

cloth_net_mean = [124.45687032, 129.73329028, 101.46781862]
cloth_net_std = [68.81949858, 72.06266253, 80.67131097]

basket_net_mean = 0.16303136
basket_net_std = 0.41344904

basket_net_ul = 1.35
basket_net_ll = 1

# Used to match depth images to simulated images
basket_net_bounds = []

basket_im_dims = [144, 256]

basket_near_pos = [0.65, 0.3,  0.875]
basket_far_pos = [0.6, -0.6,  0.875]
basket_near_rot = [2*np.pi/3, 0, np.pi/2]
basket_far_rot = [np.pi/4, 0, np.pi/2]

# cloth_grid_spacing = 0.15 # From center to center is 0.2 + 0.15 + 0.2
