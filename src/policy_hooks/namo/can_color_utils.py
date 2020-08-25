import numpy as np

BLUE = 0
WHITE = 1
YELLOW = 2
GREEN = 3

colors = {
            "blue": BLUE,
            "white": WHITE,
            "yellow": YELLOW,
            "green": GREEN
         }

def get_can_color_mapping(cans):
    can_to_colors = {}
    for can in cans:
        color = np.random.choice(list(colors.keys()))
        can_to_colors[can.lower()] = (color, colors[color])

    return can_to_colors
