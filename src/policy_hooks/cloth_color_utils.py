import numpy as np

BLUE = 0
WHITE = 1
YELLOW = 2
GREEN = 3

colors = {
            BLUE: "0 0 1 1",
            WHITE: "1 1 1 1",
            YELLOW: "1 1 0 1",
            GREEN: "0 1 0 1"
         }

def get_cloth_color_mapping(cloths):
    splits = []
    for i in range(len(colors.keys()-1)):
        splits.append(np.random.choice(range(num_cloth)))

    splits.sort()
    color_map = {}
    color_list = colors.keys()
    color_ind = 0
    for cloth in cloths:
        while i >= splits[0]:
            splits[0].pop(0)
            color_ind += 1
        color_map[cloth] = (color_list[color_ind], colors[color_list[color_ind]])

    return color_map, colors.copy()
