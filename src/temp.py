import numpy as np
import time

a = np.ones((40, 1, 37, 1, 37))
b = np.ones((40, 1, 37, 1, 37))
c = np.ones((40, 1, 37, 1, 37))

start_t = time.time()

d = a*b*c

print(time.time()-start_t)
e = np.sum(d, axis=1)
print(time.time()-start_t)
f = np.sum(e, axis=1)
print(time.time()-start_t)


