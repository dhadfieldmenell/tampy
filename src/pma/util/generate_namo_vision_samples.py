from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import random
import time
import imageio
from scipy.io import savemat

WALL_ENDPOINTS = [[-1.0,-3.0],[-1.0,4.0],[1.9,4.0],[1.9,8.0],[5.0,8.0],[5.0,4.0],[8.0,4.0],[8.0,-3.0],[-1.0,-3.0]]
ROBOT_RADIUS = 0.4
CAN_RADIUS = 0.3

class Pose:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

def closet_maker(thickness, wall_endpoints, ax):
    rects = []
    for i, (start, end) in enumerate(zip(wall_endpoints[0:-1], wall_endpoints[1:])):
        dim_x, dim_y = 0, 0
        et = thickness / 2.0
        if start[0] == end[0]: # vertical line
            if start[1] > end[1]: #downwards line
                x1 = (start[0] - et, start[1] + et)
                x2 = (end[0] + et, end[1] - et)
            else:
                x1 = (end[0] - et, end[1] + et)
                x2 = (start[0] + et, start[1] - et)
        elif start[1] == end[1]: # horizontal line
            if start[0] < end[0]: #left to right
                x1 = (start[0] - et, start[1] + et)
                x2 = (end[0] + et, end[1] - et)
            else:
                x1 = (end[0] - et, end[1] + et)
                x2 = (start[0] + et, start[1] - et)
        left_bottom = (x1[0], x2[1])
        width = x2[0] - x1[0]
        height = x1[1] - x2[1]
        rects.append([(left_bottom[0], left_bottom[1]), width, height])
    for rect in rects:
        p = patches.Rectangle(rect[0],rect[1],rect[2], lw=0, color="brown")
        ax.add_patch(p)
    return True

def collect_samples(numRobots=1, numCans=2):
        images = []
        labels = []
        for iter in range(10000):
            if iter % 1000 == 0:
                print("Iteration {}".format(iter))
            fig, ax = plt.subplots()
            objList = []
            center = 0
            radius = 0
            circColor = None
            x = None
            y = None
            radius = None
            for _ in range(numRobots):
                x, y, radius = namo_2d_location_generator(objects = objList)
                objList.append(Pose(x, y, radius))
                ax.add_artist(plt.Circle((x, y), ROBOT_RADIUS, color='g'))
            for _ in range(numCans):
                x, y, radius = namo_2d_location_generator(objects = objList)
                objList.append(Pose(x, y, radius))
                ax.add_artist(plt.Circle((x, y), CAN_RADIUS, color='r'))
            closet_maker(1, WALL_ENDPOINTS, ax)
            ax.set_xlim(-3, 10)
            ax.set_ylim(-5, 10)
            plt.gca().set_aspect('equal', adjustable='box')
            fig.canvas.draw()
            plt.axis("off")
            image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            objLocations = []
            for obj in objList:
                objLocations.extend([obj.x, obj.y, obj.radius])
            label = np.array(objLocations)
            labels.append(label)
            # imageio.imwrite('outfile.jpg', data)
            plt.close()
        images = np.array(images)
        labels = np.array(labels)
        dataDict = {'images': images, 'labels': labels}
        savemat("namo_2d_images.mat", dataDict)

def namo_2d_location_generator(radius = None, objects = [], thickness = 0.5):
    '''
    Samples an x, y, radius for a generic can or circle that is collision free
    '''
    if not radius:
        radius = random.uniform(0.3, 0.5)
    # two regions in closet
    point_not_found = True
    x = None
    y = None
    objTemp = None
    while (point_not_found):
        if random.uniform(0, 1) >= 1.0/7.0: # sample from larger area
            x = random.uniform(-1 + thickness + radius, 8 - thickness - radius)
            y = random.uniform(-3 + thickness + radius, 4 - thickness - radius)
        else:
            x = random.uniform(2 + thickness + radius, 5 - thickness - radius)
            y = random.uniform(4 - thickness + radius, 8 - thickness - radius )
        objTemp = Pose(x, y, radius)

        point_not_found = False
        for obj in objects:
            if in_collision(objTemp, obj):
                point_not_found = True
                break
    return x, y, radius

def in_collision(body1, body2):
    """
    Modeled as circles with x, y, radius
    Check if distance between centers is >= sum of radii
    """
    x_diff = body1.x - body2.x
    y_diff = body1.y - body2.y
    squared_diff = x_diff * x_diff + y_diff * y_diff
    radii_sum = body1.radius + body2.radius
    radii_sum_squared = radii_sum * radii_sum
    if squared_diff < radii_sum_squared:
        return True
    else:
        return False
collect_samples()