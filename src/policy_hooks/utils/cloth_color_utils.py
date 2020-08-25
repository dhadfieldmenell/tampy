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
    colors_to_cloth = {}
    cloth_to_colors = {}
    for cloth in cloths:
        color = np.random.choice(list(colors.keys()))
        colors_to_cloth[color] = cloth
        cloth_to_colors[cloth.lower()] = (color, colors[color])

    return cloth_to_colors, colors.copy()
