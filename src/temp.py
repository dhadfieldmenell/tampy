import numpy as np
pr2_disp = np.array([-1, 1])

dx, dy = 1e1 * pr2_disp
zx, zy = 0, 0
x1, y1 = np.array([0., 0.3]) - [0.5*dx, 0.5*dy] - [zx, zy]
x2, y2 = x1 + dx, y1 + dy
dr = np.sqrt(dx**2 + dy**2)
D = x1 * y2 - x2 * y1
r = 1
sy = -1. if dy < 0 else 1.

if dx > 0 and dy > 0:
    x = (D * dy + sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
    y = (-D * dx + np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
elif dx < 0 and dy < 0:
    x = (D * dy - sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
    y = (-D * dx - np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
elif dx > 0 and dy < 0:
    x = (D * dy - sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
    y = (-D * dx - np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
elif dx < 0 and dy > 0:
    x = (D * dy + sy * dx * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)
    y = (-D * dx + np.abs(dy) * np.sqrt(r**2 * dr**2 - D**2)) / (dr**2)

print(x,y)
