'''
VGA
res: 672x376
Center of table: 200, 350 (row, col) -> 52, 5 (x, y)
Pixels per centimeter: 2.3
Pixel window: 40
'''
import numpy as np


cloth_grid_bounds = [[129, 672], [68, 355]]

cloth_grid_window = 8 # 20

cloth_grid_input_dim = 15

# Ordered (row, col)
# Region 5 is the basket near pos, 6 is the basket far pos
# cloth_grid_coordinates = [
#     ((230, 200), 2),
#     ((160, 200), 2),
#     ((180, 260), 5),
#     ((130, 270), 2),
#     ((210, 340), 2),
#     ((150, 330), 5),
#     ((180, 410), 3),
#     ((130, 360), 3),
#     ((150, 510), 6),
#     ((195, 550), 6),
#     ((250, 160), 2),
#     ((250, 100), 5),
# ]
cloth_grid_coordinates = [
    # ((175, 180), 1),
    # ((140, 150), 2),
    # ((160, 140), 2),
    # ((190, 200), 2),
    # ((90, 270), 2),
    # ((100, 320), 2),
    # ((100, 335), 3),
    # ((75, 400), 3),
    # ((90, 380), 3),
    # ((250, 595), 4),
    # ((285, 595), 4),
    # ((160, 290), 5),
    # ((150, 320), 5),
    # ((140, 360), 5),
    # ((180, 520), 6),
    # ((150, 470), 6),
    # ((210, 560), 6),
    ((0.15, 0.75), 1),
    ((0.37, 0.67), 2),
    ((0.55, 0.87), 2),
    ((0.66, 0.85), 2),
    ((0.82, 0.62), 2),
    ((0.67, 0.25), 5),
    ((0.65, 0.17), 5),
    ((0.52, 0.37), 5),
    ((0.57, 0.47), 5),
    ((0.6, 0.3), 5),
    ((0.86, -0.15), 3),
    ((0.99, 0.42), 3),
    ((0.87, 0.5), 3),
    ((0.56, -0.29), 7),
    ((0.5, -0.5), 7),
    ((0.42, -0.68), 7),
    ((0.55, -0.38), 7),
    ((0.27, -0.57), 7),
    ((0.58, -0.01), 6),
    ((0.6, 0.6), 6),
]

washer_im_locs = [((180, 240), (0.14, 1.3, 0.73)), # -> CENTER FRONT
                  ((155, 145), (0, 1.15, 0.77)), # -> LEFT FRONT
                  ((180, 300), (0.2, 1., 0.82)), # -> RIGHT FRONT
                  ((160, 215), (0.2, 1.38, 0.77)), # -> CENTER REAR
                  ((130, 110), (0.1, 1.38, 0.77)), # -> LEFT REAR
                  ((130, 290), (0.27, 1.34, 0.76))] # -> RIGHT REAR

wrist_im_offsets = [((0, -50), (0, 5)),
                    ((50, 0), (-5, 0)),
                    ((0, 50), (0, -5)),
                    ((-50, 0), (5, 0)),
                    ((-50, -50), (5, 5)),
                    ((-50, 50), (5, -5)),
                    ((50, 50), (-5, -5)),
                    ((50, -50), (-5, 5)),
                    ((0, -150), (0, -5)),]

# cloth_grid_ref = np.array([[200, 350], [55, 5]])
cloth_grid_ref = np.array([[150, 350], [55, 5]])

pixels_per_cm = 2.3

# cloth_net_mean = [117.8504316, 122.86169381, 84.21057003]
# cloth_net_std = [49.69568644, 58.26177112, 75.96082055]

# cloth_net_mean = [124.45687032, 129.73329028, 101.46781862]
# cloth_net_std = [68.81949858, 72.06266253, 80.67131097]

cloth_net_mean = [124.21902223, 128.37344094, 111.40189499]
cloth_net_std = [77.63351386, 78.496201 , 82.02380836]

# basket_net_mean = 0.16303136
# basket_net_std = 0.41344904
# basket_net_mean = 0.311865
# # basket_net_std = 0.5426525
# basket_net_mean = 0.1102060987658177
# basket_net_std = 0.33788160444680088
# basket_net_mean = 0.1012
# basket_net_std = 0.3319
basket_net_mean = 0.0845566170807589
basket_net_std = 0.29633006991324462


# basket_net_ul = 1.35
# # basket_net_ll = 1
# basket_net_ul = 1.4
# # basket_net_ll = 1.05
# basket_net_ul = 1.25
# basket_net_ll = 1.05
# basket_net_ul = 1.3
# basket_net_ll = 1.1
basket_net_ul = 1.2
basket_net_ll = 1.05

# Used to match depth images to simulated images
basket_net_bounds = []

basket_im_dims = [144, 256]

# basket_near_pos = np.array([0.65, 0.35,  0.875])
basket_near_pos = np.array([0.64, 0.22,  0.88])
basket_far_pos = np.array([0.5, -0.5,  0.88])
basket_near_rot = np.array([2*np.pi/3, 0, np.pi/2])
# basket_near_rot = np.array([np.pi/2, 0, np.pi/2])
basket_far_rot = np.array([np.pi/4, 0, np.pi/2])
# basket_near_rot = [-np.pi/4, 0, np.pi/2]
# basket_far_rot = [np.pi/4, 0, np.pi/2]

# basket_net_zero_pos = [0.53, 0.08, np.pi/2]
basket_net_zero_pos = [0.425, 0.08, np.pi/2]

# cloth_grid_spacing = 0.15 # From center to center is 0.2 + 0.15 + 0.2

left_joints = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
right_joints = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']

MAX_CLOTHS = 4

regions = [[40*np.pi/180], [-2*np.pi/180], [-44*np.pi/180], [-88*np.pi/180]]
